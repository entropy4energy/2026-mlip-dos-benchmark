#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "ase",
#     "ase-db-backends",
#     "loguru",
#     "tqdm",
#     "pymatgen",
#     "numpy",
#     "scipy",
#     "matplotlib",
# ]
# ///
import os
import sys
import re
import lzma
import csv
import glob
import json
import argparse
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, Callable

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.db import connect
from tqdm import tqdm
from loguru import logger

# pymatgen structure matching
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor

from utils import read_vasp_dos_data, calculate_dos_metrics, read_vasp_xml_maybe_uncompressed, get_rmsd, resolve_file, public_id

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}")

DOS_GRID_POINTS = 1000

# --- StructureMatcher setups ---
SM_DEFAULT = StructureMatcher(
    ltol=0.2, stol=0.3, angle_tol=5.0, scale=True,
    primitive_cell=True, attempt_supercell=False,
)
SM_RELAXED = StructureMatcher(
    ltol=0.3, stol=0.5, angle_tol=10.0, scale=True,
    primitive_cell=True, attempt_supercell=False,
)

def ase_to_pmg(atoms):
    """Convert ASE Atoms to pymatgen Structure."""
    return AseAtomsAdaptor.get_structure(atoms)

def classify_topology(gt_atoms, mlip_atoms):
    """Classify topology relationship between GT and MLIP-relaxed structures."""
    try:
        gt_pmg = ase_to_pmg(gt_atoms)
        mlip_pmg = ase_to_pmg(mlip_atoms)
    except Exception:
        return "error"

    if SM_DEFAULT.fit(gt_pmg, mlip_pmg):
        return "preserved"
    elif SM_RELAXED.fit(gt_pmg, mlip_pmg):
        return "degraded"
    else:
        return "broken"

def get_rmsd_and_max_dist(atoms1, atoms2):
    """Calculate RMSD and max displacement between two ASE Atoms objects using pymatgen."""
    try:
        str1 = AseAtomsAdaptor.get_structure(atoms1)
        str2 = AseAtomsAdaptor.get_structure(atoms2)
        matcher = StructureMatcher()
        result = matcher.get_rms_dist(str1, str2)
        if result is None:
            return np.nan, np.nan
        return result[0], result[1]
    except Exception:
        return np.nan, np.nan

def load_mlip_predictions(mlip_dir: str) -> dict:
    """Load MLIP-predicted energy and forces from .aselmdb files."""
    predictions = {}
    db_paths = glob.glob(os.path.join(mlip_dir, '*.aselmdb'))
    logger.debug(f"Found {len(db_paths)} .aselmdb files in {mlip_dir}")
    for db_path in db_paths:
        try:
            db = connect(db_path)
            for row in db.select():
                if hasattr(row, 'energy') and hasattr(row, 'forces'):
                    fname = row.fname
                    atoms = row.toatoms()
                    energy_per_atom = float(row.energy / len(atoms))
                    force_magnitudes = np.linalg.norm(row.forces, axis=1)
                    fmax = float(np.max(force_magnitudes))
                    fmean = float(np.mean(force_magnitudes))
                    predictions[fname] = {
                        'energy': energy_per_atom,
                        'fmax': fmax,
                        'fmean': fmean,
                        'forces': row.forces.tolist()
                    }
        except Exception as e:
            logger.warning(f"Could not process {db_path}: {e}")
    logger.debug(f"Loaded {len(predictions)} predictions from {mlip_dir}")
    return predictions

def get_core_hours(filepath: str) -> float:
    """Parse core hours from a VASP OUTCAR file."""
    filepath = resolve_file(filepath)
    cores = None
    time_sec = None

    if filepath.endswith('.xz'):
        open_func = lambda p: lzma.open(p, 'rt', encoding='utf-8')
    else:
        open_func = lambda p: open(p, mode='r', encoding='utf-8')

    with open_func(filepath) as f:
        for line in f:
            if cores is None and "running on" in line and "total cores" in line:
                m = re.search(r"running on\s+(\d+)\s+total cores", line)
                if m:
                    cores = int(m.group(1))
            if "Elapsed time (sec):" in line:
                m = re.search(r"Elapsed time \(sec\):\s+([\d\.]+)", line)
                if m:
                    time_sec = float(m.group(1))

    if cores is None or time_sec is None:
        raise ValueError(f"Could not parse VASP runtime fields from {filepath}")
    return (cores * time_sec) / 3600.0

def get_space_group_operations(filepath: str) -> int:
    """Parse number of space group operations from a VASP OUTCAR file."""
    filepath = resolve_file(filepath)
    ops = None

    if filepath.endswith('.xz'):
        open_func = lambda p: lzma.open(p, 'rt', encoding='utf-8')
    else:
        open_func = lambda p: open(p, mode='r', encoding='utf-8')

    with open_func(filepath) as f:
        for line in f:
            if "space group operations" in line:
                m = re.search(r"Found\s+(\d+)\s+space group operations", line)
                if m:
                    ops = int(m.group(1))
                    break
    if ops is None:
        raise ValueError(f"Could not parse space group operations from {filepath}")
    return ops

def get_relative_paths(root_dir: str) -> set[str]:
    """Find all systems by looking for vasprun.xml.static files."""
    root_path = Path(root_dir)
    files = glob.glob(str(root_path / "**" / "vasprun.xml.static"), recursive=True)
    if not files:
        files = glob.glob(str(root_path / "**" / "vasprun.xml.static.xz"), recursive=True)
    if not files:
        raise FileNotFoundError(f"No 'vasprun.xml.static' files found in {root_dir}")
    rels = set()
    for f in files:
        rel_dir = Path(f).parent.relative_to(root_path)
        rels.add(str(rel_dir))
    return rels

def calculate_dos_properties(
    vasprun_path: str,
    doscar_path: str,
    sigma: float,
    smoothing: float,
    dos_grid_points: int = DOS_GRID_POINTS
) -> Tuple[Callable, np.ndarray, np.ndarray, float]:
    _, dos_interpolator, e_fermi_read, _ = read_vasp_dos_data(
        vasprun_path, doscar_path, sigma=sigma, smoothing=smoothing
    )
    if dos_interpolator is None:
        raise ValueError(f"Failed to create DOS interpolator for {vasprun_path}")

    dos_grid_rel = np.linspace(-10, 10, dos_grid_points).reshape(-1, 1)
    dos_grid_abs = dos_grid_rel + e_fermi_read
    dos_on_grid = dos_interpolator(dos_grid_abs)
    dos_on_grid[dos_on_grid < 0] = 0

    return dos_interpolator, dos_grid_rel.flatten(), dos_on_grid.flatten(), e_fermi_read

def process_system(
    rel_path: str,
    gt_dir: str,
    mlip_dirs_map: dict,
    mlip_predictions_map: dict,
    dos_sigma: float,
    dos_smoothing: float,
    dos_grid_points: int = DOS_GRID_POINTS
) -> dict | None:
    # fname (the path) stays internal; the CSV names the entry by catalog,
    # species and prototype
    catalog, species, prototype = public_id(rel_path)
    results = {'fname': rel_path, 'catalog': catalog, 'species': species,
               'prototype': prototype}

    # --- Ground Truth Data ---
    gt_system_path = os.path.join(gt_dir, rel_path)
    try:
        r1_hours = get_core_hours(os.path.join(gt_system_path, "OUTCAR.relax1.xz"))
        r2_hours = get_core_hours(os.path.join(gt_system_path, "OUTCAR.relax2.xz"))
        st_hours = get_core_hours(os.path.join(gt_system_path, "OUTCAR.static.xz"))
        results['gt_core_hours_relax'] = r1_hours + r2_hours
        results['gt_core_hours_static'] = st_hours

        gt_sg_ops = get_space_group_operations(os.path.join(gt_system_path, "OUTCAR.relax2.xz"))
        results['gt_sg_ops'] = gt_sg_ops

        gt_vasprun_path = os.path.join(gt_system_path, "vasprun.xml.static.xz")
        gt_atoms = read_vasp_xml_maybe_uncompressed(gt_vasprun_path, index=-1)
        gt_energy = gt_atoms.get_potential_energy()

        num_atoms = len(gt_atoms)
        results['num_atoms'] = num_atoms
        results['gt_energy'] = gt_energy / num_atoms

        gt_doscar_path = os.path.join(gt_system_path, "DOSCAR.static.xz")
        _, gt_dos_grid_rel, gt_dos_on_grid, gt_e_fermi = calculate_dos_properties(
            gt_vasprun_path, gt_doscar_path, dos_sigma, dos_smoothing, dos_grid_points
        )
        results['gt_e_fermi'] = gt_e_fermi

    except Exception as e:
        logger.warning(f"Skipping system {rel_path} due to error in Ground Truth data: {e}")
        return None

    # --- MLIP Data ---
    for name, mlip_dir in mlip_dirs_map.items():
        fname_to_find = '/' + rel_path
        preds = mlip_predictions_map.get(name, {}).get(fname_to_find)
        if preds:
            results[f'{name}_pred_energy'] = preds['energy']
            results[f'{name}_pred_fmax'] = preds['fmax']
            results[f'{name}_pred_fmean'] = preds['fmean']
            results[f'{name}_pred_forces_flat'] = list(np.array(preds['forces']).flatten()) if 'forces' in preds else None
        else:
            results[f'{name}_pred_energy'] = None
            results[f'{name}_pred_fmax'] = None
            results[f'{name}_pred_fmean'] = None
            results[f'{name}_pred_forces_flat'] = None

        mlip_system_path = os.path.join(mlip_dir, rel_path)

        try:
            mlip_ch = get_core_hours(os.path.join(mlip_system_path, "OUTCAR.static.xz"))
            results[f'{name}_core_hours_static'] = mlip_ch

            mlip_sg_ops = get_space_group_operations(os.path.join(mlip_system_path, "OUTCAR.static.xz"))
            results[f'{name}_sg_ops'] = mlip_sg_ops

            mlip_vasprun_path = os.path.join(mlip_system_path, "vasprun.xml.static.xz")
            mlip_atoms = read_vasp_xml_maybe_uncompressed(mlip_vasprun_path, index=-1)
            mlip_energy = mlip_atoms.get_potential_energy()
            mlip_n_atoms = len(mlip_atoms)
            results[f'{name}_num_atoms'] = mlip_n_atoms
            results[f'{name}_energy'] = mlip_energy / mlip_n_atoms

            rmsd = get_rmsd(gt_atoms, mlip_atoms)
            results[f'{name}_rmsd'] = rmsd

            mlip_doscar_path = os.path.join(mlip_system_path, "DOSCAR.static.xz")
            _, mlip_dos_grid_rel, mlip_dos_on_grid, mlip_e_fermi = calculate_dos_properties(
                mlip_vasprun_path, mlip_doscar_path, dos_sigma, dos_smoothing, dos_grid_points
            )
            results[f'{name}_e_fermi'] = mlip_e_fermi

            reference_dos = (gt_dos_grid_rel, gt_dos_on_grid)
            comparison_dos = (mlip_dos_grid_rel, mlip_dos_on_grid)
            dos_metrics = calculate_dos_metrics(reference_dos, comparison_dos)
            results[f'{name}_dos_cosine_dist'] = dos_metrics["cosine_distance"]
            results[f'{name}_dos_wasserstein_dist'] = dos_metrics["wasserstein_distance"]

            # Structure Matcher Topology & Displacement
            topology = "unknown"
            max_dist = None
            try:
                topology = classify_topology(gt_atoms, mlip_atoms)
                _, max_dist_val = get_rmsd_and_max_dist(gt_atoms, mlip_atoms)
                if not np.isnan(max_dist_val):
                    max_dist = float(max_dist_val)
            except Exception:
                topology = "error"
            results[f'{name}_topology'] = topology
            results[f'{name}_max_dist'] = max_dist

            # Force Residuals
            dft_fmax = None
            dft_frmse = None
            dft_forces_flat = None
            if mlip_atoms is not None:
                dft_forces = mlip_atoms.get_forces()
                dft_force_magnitudes = np.linalg.norm(dft_forces, axis=1)
                dft_fmax = float(np.max(dft_force_magnitudes))
                dft_frmse = float(np.sqrt(np.mean(dft_forces ** 2)))
                dft_forces_flat = dft_forces.flatten().tolist()
            results[f'{name}_dft_fmax'] = dft_fmax
            results[f'{name}_dft_frmse'] = dft_frmse
            results[f'{name}_dft_forces_flat'] = dft_forces_flat

            # Symmetry breaking status
            sym_broken = False
            sym_reason = None
            if gt_sg_ops is not None and mlip_sg_ops is not None:
                if gt_sg_ops != mlip_sg_ops:
                    sym_broken = True
                    sym_reason = f"sg_ops:{gt_sg_ops}->{mlip_sg_ops}"
            results[f'{name}_symmetry_broken'] = sym_broken
            results[f'{name}_symmetry_reason'] = sym_reason

            # Energy MAE
            results[f'{name}_energy_mae'] = abs(results['gt_energy'] - results[f'{name}_energy'])

        except Exception as e:
            logger.warning(f"Error processing MLIP data for {name}/{rel_path}: {e}. Filling with None.")
            results[f'{name}_core_hours_static'] = None
            results[f'{name}_sg_ops'] = None
            results[f'{name}_num_atoms'] = None
            results[f'{name}_energy'] = None
            results[f'{name}_e_fermi'] = None
            results[f'{name}_rmsd'] = None
            results[f'{name}_max_dist'] = None
            results[f'{name}_dos_cosine_dist'] = None
            results[f'{name}_dos_wasserstein_dist'] = None
            results[f'{name}_dft_fmax'] = None
            results[f'{name}_dft_frmse'] = None
            results[f'{name}_dft_forces_flat'] = None
            results[f'{name}_topology'] = "error"
            results[f'{name}_symmetry_broken'] = None
            results[f'{name}_symmetry_reason'] = None
            results[f'{name}_energy_mae'] = None

    return results

# --- Main Driver ---

def main():
    parser = argparse.ArgumentParser(description="Clean merged data collection pipeline for MLIP relaxation benchmarking.")
    parser.add_argument("ground_truth_dir", help="Root directory of the reference (ground truth) dataset.")
    parser.add_argument("mlip_dirs", nargs='+', help="Root directories of the MLIP-relaxed datasets.")
    parser.add_argument("-o", "--output", default="data/data.csv", help="Output CSV path or output directory.")
    parser.add_argument("--test-run", action="store_true", help="Run in test mode, processing only the first 20 systems.")
    parser.add_argument("--dos-sigma", type=float, default=0.05, help="Sigma for DOS broadening.")
    parser.add_argument("--dos-smoothing", type=float, default=1e-5, help="Smoothing parameter for DOS RBFInterpolator.")
    parser.add_argument("--dos-grid-points", type=int, default=DOS_GRID_POINTS, help="Number of points for DOS energy grid.")
    parser.add_argument("--db-dir", default=None, help="Directory containing .aselmdb files (local copies for LMDB compatibility).")
    parser.add_argument("--max-workers", type=int, default=None, help="Maximum number of parallel workers (defaults to min(4, CPU count)).")
    args = parser.parse_args()

    gt_dir = os.path.abspath(args.ground_truth_dir)
    mlip_dirs_map = {Path(p).name: os.path.abspath(p) for p in args.mlip_dirs}
    mlip_names = sorted(mlip_dirs_map.keys())

    # Establish directories and file paths
    output_path = os.path.abspath(args.output)
    if os.path.isdir(output_path) or not output_path.endswith('.csv'):
        output_dir = output_path
        csv_path = os.path.join(output_dir, "data.csv")
    else:
        csv_path = output_path
        output_dir = os.path.dirname(csv_path)

    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading MLIP predictions...")
    mlip_predictions_map = {}
    for name, path in mlip_dirs_map.items():
        logger.info(f"Loading predictions for {name} from .aselmdb files...")
        pred_path = os.path.join(args.db_dir, name) if args.db_dir else path
        mlip_predictions_map[name] = load_mlip_predictions(pred_path)

    logger.info("Indexing MLIP systems to find all systems...")
    all_rels = set()
    for name, path in mlip_dirs_map.items():
        logger.info(f"Indexing {name} systems...")
        try:
            mlip_rels = get_relative_paths(path)
            all_rels.update(mlip_rels)
        except FileNotFoundError as e:
            logger.warning(f"Could not index {name}, skipping: {e}")

    if not all_rels:
        logger.error("Could not find any systems in any MLIP directories.")
        sys.exit(1)

    logger.info(f"Found {len(all_rels)} total systems to process.")
    if args.test_run:
        logger.info("--- TEST RUN MODE: Processing only the first 20 systems. ---")
        all_rels = set(sorted(list(all_rels))[:20])

    # Process all systems in parallel
    all_results = []
    # Limit max workers to prevent OOM
    max_workers = args.max_workers if args.max_workers is not None else min(4, os.cpu_count() or 1)
    logger.info(f"Using {max_workers} parallel workers.")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_system,
                rel,
                gt_dir,
                mlip_dirs_map,
                mlip_predictions_map,
                args.dos_sigma,
                args.dos_smoothing,
                args.dos_grid_points
            ): rel
            for rel in sorted(list(all_rels))
        }

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing systems"):
            res = fut.result()
            if res:
                all_results.append(res)

    if not all_results:
        sys.exit("No systems could be processed successfully. Exiting.")

    # Sort results for consistent CSV output
    all_results.sort(key=lambda x: x['fname'])

    # Re-order results to match original data.csv row order if it exists
    repo_path = Path(__file__).resolve().parents[1]
    ref_csv = os.path.join(repo_path, "data", "data.csv")
    if os.path.exists(ref_csv):
        try:
            with open(ref_csv, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                keys = [(row["catalog"], row["species"], row["prototype"])
                        for row in reader if row.get("catalog")]
            ref_order = {key: i for i, key in enumerate(keys)}
            all_results.sort(key=lambda x: ref_order.get(
                (x['catalog'], x['species'], x['prototype']), len(ref_order)))
            logger.info(f"Sorted results to match {ref_csv} order.")
        except Exception as e:
            logger.warning(f"Could not sort results like {ref_csv}: {e}")

    # Dynamically create CSV header with ALL columns merged
    csv_header = [
        'catalog',
        'species',
        'prototype',
        'num_atoms',
        'gt_core_hours_relax',
        'gt_core_hours_static',
        'gt_sg_ops',
        'gt_energy',
        'gt_e_fermi'
    ]
    for name in mlip_names:
        csv_header.extend([
            f'{name}_core_hours_static',
            f'{name}_sg_ops',
            f'{name}_num_atoms',
            f'{name}_energy',
            f'{name}_e_fermi',
            f'{name}_rmsd',
            f'{name}_max_dist',
            f'{name}_dos_cosine_dist',
            f'{name}_dos_wasserstein_dist',
            f'{name}_dft_fmax',
            f'{name}_dft_frmse',
            f'{name}_pred_energy',
            f'{name}_pred_fmax',
            f'{name}_pred_fmean',
            f'{name}_topology',
            f'{name}_symmetry_broken',
            f'{name}_symmetry_reason',
            f'{name}_energy_mae',
        ])

    # Write the merged CSV file
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_header, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_results)
    logger.info(f"Wrote {len(all_results)} rows to {csv_path}")

if __name__ == "__main__":
    main()
