#!/usr/bin/env python3

import numpy as np
from ase import Atoms
import re
from ase.io import read
from ase.db import connect
from ase.calculators.vasp import Vasp
import subprocess
import time
import os, glob
import ase.db
from loguru import logger
import shutil
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import RBFInterpolator
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import lzma
from typing import Tuple, List, Callable
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor


def resolve_file(path: str) -> str:
    """Resolve a file path, preferring uncompressed version if it exists."""
    if path.endswith('.xz'):
        uncompressed = path[:-3]
        if os.path.exists(uncompressed):
            return uncompressed
    return path


from functools import lru_cache

@lru_cache(maxsize=8)
def _read_vasp_xml_cached(path: str, index: int = -1) -> Atoms:
    resolved_path = resolve_file(path)
    return read(resolved_path, format='vasp-xml', index=index)

def read_vasp_xml_maybe_uncompressed(path: str, **kwargs) -> Atoms:
    """Wrapper for ase.io.read that prefers uncompressed vasprun.xml."""
    if kwargs == {'index': -1} or not kwargs:
        return _read_vasp_xml_cached(path)
    path = resolve_file(path)
    return read(path, format='vasp-xml', **kwargs)


K_B_EV_PER_K = 8.6173333262145e-5

def cosine_distance(d1, d2):
    dot_product = np.dot(d1, d2)
    norm_d1 = np.linalg.norm(d1)
    norm_d2 = np.linalg.norm(d2)
    if norm_d1 == 0 or norm_d2 == 0:
        return 1.0
    similarity = dot_product / (norm_d1 * norm_d2)
    return 1.0 - similarity

def wasserstein_distance(e, d1, d2):
    area1 = np.trapezoid(d1, e)
    area2 = np.trapezoid(d2, e)
    if area1 == 0 or area2 == 0:
        return None
    d1_norm = d1 / area1
    d2_norm = d2 / area2
    cdf1 = cumulative_trapezoid(d1_norm, e, initial=0)
    cdf2 = cumulative_trapezoid(d2_norm, e, initial=0)
    return np.trapezoid(np.abs(cdf1 - cdf2), e)

def calculate_dos_metrics(
    reference_dos: tuple[np.ndarray, np.ndarray],
    comparison_dos: tuple[np.ndarray, np.ndarray]
) -> dict:
    e_ref, d_ref = reference_dos
    _, d_comp = comparison_dos

    distances = {
        "cosine_distance": cosine_distance(d_ref, d_comp),
        "wasserstein_distance": wasserstein_distance(e_ref, d_ref, d_comp)
    }
    return distances

class ZeroExtrapolator:
    def __init__(self, interpolator, x_min, x_max):
        self.interpolator = interpolator
        self.x_min = x_min
        self.x_max = x_max

    def __call__(self, x):
        # x is expected to be a 2D array with one column for RBFInterpolator
        y = self.interpolator(x)
        out_of_bounds = (x[:, 0] < self.x_min) | (x[:, 0] > self.x_max)
        y[out_of_bounds] = 0.0
        return y

def parse_doscar(filepath: str, sigma: float, smoothing: float) -> Tuple[Callable, float]:
    """
    Args:
        filepath: The path to the DOSCAR or DOSCAR.xz file.
        sigma: Broadening width in eV for Gaussian RBF interpolation.
        smoothing: Smoothing parameter for RBFInterpolator.

    Returns:
        A tuple containing:
        - An RBFInterpolator for the DOS.
        - The Fermi energy [eV].
    """
    filepath = resolve_file(filepath)

    if filepath.endswith('.xz'):
        open_func = lzma.open
        mode = 'rt'
    else:
        open_func = open
        mode = 'r'

    with open_func(filepath, mode) as f:
        lines = f.readlines()

    # The 6th line contains NEDOS and the Fermi energy [1].
    header_line_6 = lines[5].strip().split()
    nedos = int(header_line_6[2])
    e_fermi = float(header_line_6[3])


    # The total DOS data block starts from the 7th line and has nedos entries [1].
    dos_lines = lines[6 : 6 + nedos]

    energies = []
    densities = []

    # Check the first data line to see if the calculation is spin-polarized [1].
    first_data_line_cols = len(dos_lines[0].strip().split())

    for line in dos_lines:
        data = [float(x) for x in line.strip().split()]

        energy = data[0]# - e_fermi

        if first_data_line_cols == 5:
            dos = data[1] + data[2]
        else:
            dos = data[1]

        energies.append(energy)
        densities.append(dos)

    # logger.debug(f"file {filepath}, len {len(energies)}, efermi {e_fermi}")

    if not energies:
        return None, e_fermi

    energies_np = np.array(energies).reshape(-1, 1)
    densities_np = np.array(densities)

    # Use Gaussian RBF to interpolate and broaden in one step.
    # Epsilon is related to the Gaussian sigma.
    epsilon = 1 / (np.sqrt(2) * sigma) if sigma > 0 else 0

    # neighbors: avoids numerical artifact with smaller values of sigma; total range is 90 eV, divide it by 900 so that we only consider the points within this 0.1 eV of the point requested to evaluate the gaussian
    rbf_interpolator = RBFInterpolator(energies_np, densities_np, kernel='gaussian', neighbors=len(energies_np)//900, epsilon=epsilon, smoothing=smoothing)

    min_energy = energies_np.min()
    max_energy = energies_np.max()
    interpolator = ZeroExtrapolator(rbf_interpolator, min_energy, max_energy)

    return interpolator, e_fermi

def read_vasp_dos_data(
    vasprun_path: str,
    doscar_path: str,
    sigma: float,
    smoothing: float,
) -> Tuple[float, Callable, float, Atoms]:
    """
    Reads VASP XML and DOSCAR files, extracts total energy, DOS interpolator, and atoms object.
    """
    try:
        # Read total energy from vasprun.xml.static.xz
        atoms = read_vasp_xml_maybe_uncompressed(vasprun_path, index=-1)
        vasp_energy = atoms.get_potential_energy()

        # Read DOS data from DOSCAR.static.xz
        dos_interpolator, e_fermi = parse_doscar(doscar_path, sigma=sigma, smoothing=smoothing)

        if dos_interpolator is None:
            logger.warning(f"No DOS data found for {vasprun_path}. Returning None.")
            return None, None, None, None

        return vasp_energy, dos_interpolator, e_fermi, atoms

    except Exception as e:
        logger.error(f"Error processing VASP files {vasprun_path} and {doscar_path}: {e}")
        return None, None, None, None


def compare_dos(reference_dos, comparison_dos_map):
    distances = {}

    for name, comparison_dos in comparison_dos_map.items():
        metrics = calculate_dos_metrics(reference_dos, comparison_dos)
        distances[f"{name}_cos_dist"] = metrics["cosine_distance"]
        distances[f"{name}_wass_dist"] = metrics["wasserstein_distance"]

    return distances

def compute_boltzmann_weights(energies: np.ndarray, temperature: float) -> np.ndarray:
    """
    Computes Boltzmann weights for a given set of energies at a specified temperature.

    Args:
        energies (np.ndarray): Array of energies.
        temperature (float): Temperature in Kelvin.

    Returns:
        np.ndarray: Array of Boltzmann weights.
    """
    if temperature <= 0:
        logger.warning("Temperature is non-positive. Returning uniform weights.")
        return np.ones_like(energies) / len(energies)

    rel_energies = energies - np.min(energies)
    beta = 1.0 / (K_B_EV_PER_K * temperature)
    raw_weights = np.exp(-beta * rel_energies)
    weight_sum = np.sum(raw_weights)
    if weight_sum == 0:
        logger.warning("All Boltzmann weights are zero; check energies. Returning uniform weights.")
        return np.ones_like(energies) / len(energies)
    return raw_weights / weight_sum


def calculate_weighted_dos(
    dos_on_grid_list: list[np.ndarray],
    weights: np.ndarray,
) -> np.ndarray:
    """
    Calculates a weighted average of multiple DOS already on a common grid.

    Args:
        dos_on_grid_list: A list of DOS arrays, each on the same common grid.
        weights: An array of weights corresponding to each DOS.

    Returns:
        The weighted average DOS array.
    """
    if not dos_on_grid_list:
        return np.array([])

    weighted_dos = np.average(np.array(dos_on_grid_list), axis=0, weights=weights)
    return weighted_dos


def get_kpts_from_nkppra(atoms: Atoms, nkppra=10000) -> list[int]:
    # From aflow_xatom
    n_atoms = len(atoms)
    n_kpts_target = nkppra / n_atoms

    if n_kpts_target <= 1:
        return [1, 1, 1]

    reciprocal_cell = atoms.cell.reciprocal()
    b_norms = np.linalg.norm(reciprocal_cell, axis=1)

    valid_norms = b_norms[b_norms > 1e-9]
    if len(valid_norms) == 0:
        return [1, 1, 1]

    dk = np.min(valid_norms)
    dk_delta = 0.999

    while True:
        kpts_guess = np.maximum(1, np.floor(b_norms / dk)).astype(int)

        if np.prod(kpts_guess) >= n_kpts_target:
            return kpts_guess.tolist()

        dk *= dk_delta
        if dk < 1e-5:
            return [1, 1, 1]

def ase_db_write_row_maybe_replace(outdb, atoms, row):
    # list add row into db but with all the info from row
    # so we don't have to write stuff like
    # relaxed_db.write(atoms, original_unique_id=row.unique_id, original_id=row.id, fname=row.fname, pocc_id=row.pocc_id, degen=row.degen)
    possible_properties = ['fname', 'pocc_id', 'degen', 'uff']
    outdb.write(atoms, **{k: getattr(row, k) for k in possible_properties if hasattr(row, k)})

def public_id(rel_path):
    """Catalog, species and prototype of a calculation, from its path
    .../<catalog>/LIB/<species>/<prototype>; together they name the
    reference calculation (e.g. LIB2, AgCd, A5B8_cI52_217_ce_cg-001.AB)."""
    parts = rel_path.strip('/').split('/')
    i = parts.index('LIB')
    return parts[i - 1], parts[i + 1], parts[i + 2]

def get_setups_from_fname(fname, atoms):
    pp_string = fname.removeprefix('/').split('/')[3].split(':')[0]
    symbols = sorted(list(set(atoms.get_chemical_symbols())), key=len, reverse=True)

    pattern = f"({'|'.join(symbols)})"
    tokens = re.split(pattern, pp_string)

    setups = {}
    for i in range(1, len(tokens), 2):
        symbol = tokens[i]
        suffix = tokens[i + 1]
        if suffix.startswith('_'):
            setups[symbol] = suffix

    logger.debug(f"Pseudopotential string parsed: {pp_string} -> {setups}")
    return setups

def run_vasp_job(row, dirname, nkppra, pocc=True):
    out_base_dir = f'{dirname.removesuffix("/")}'

    if pocc:
        job_id = f"{row.fname}/{row.pocc_id}"
        calc_dir = f'{out_base_dir}/{job_id}'
    else:
        job_id = row.fname
        calc_dir = f'{out_base_dir}/{job_id}'

    # logger.info(f"Job started on {calc_dir}")
    shutil.rmtree(calc_dir, ignore_errors=True)

    atoms = row.toatoms()


    calc_params = {
        'ncore': 24,
        'nsw': 0,
        'ibrion': 2,
        'isif': 2,
        'prec': 'Accurate',
        'encut': 560,
        'ediff': 1e-6,
        'nelm': 120,
        'nelmin': 2,
        'algo': 'Normal',
        'lreal': False,
        'amix': 0.1,
        'bmix': 0.01,
        'ismear': -5,
        'sigma': 0.05,
        'nbands': 124,
        'lorbit': 10,
        'lcharg': True,
        'laechg': True,
        'lwave': False,
        'nedos': 10000,
        'emin': -50.0,
        'emax': 100.0,
        'kpts': tuple(get_kpts_from_nkppra(atoms, nkppra)),
        'gamma': True,
        'xc': "PBE",
        'setups': get_setups_from_fname(row.fname, atoms)
    }

    start_time = time.monotonic()
    try:
        trial_run_params = calc_params.copy()
        trial_run_params['ismear'] = 1
        trial_run_params['sigma'] = 0.1
        trial_run_params['kpts'] = tuple(get_kpts_from_nkppra(atoms, 1000))
        logger.info(f"Starting trial run for {calc_dir} with ISMEAR=0, SIGMA=0.2")
        atoms.calc = Vasp(directory=calc_dir, **trial_run_params)
        atoms.get_potential_energy()
        logger.info(f"Initial trial run for {calc_dir} succeeded, now re-running to obtain DOS")

        dos_params = calc_params.copy()
        dos_params['icharg'] = 11 # read CHGCAR from previous step
        atoms.calc = Vasp(directory=calc_dir, **dos_params)
        atoms.get_potential_energy()
        logger.info(f"Successfully obtained DOS for {calc_dir} in two steps")
    except Exception as ee:
        logger.error(f"Error in {calc_dir}: {repr(ee)}")
        return None

    duration = time.monotonic() - start_time

    subprocess.run(['fd', '--type', 'file', '--search-path', calc_dir, '-x', 'xz'], check=False)

    result_dict = {
        "atoms": atoms,
        "fname": row.fname,
        "time": duration
    }

    if pocc:
        result_dict['pocc_id'] = row.pocc_id
        result_dict['degen'] = row.degen

    return result_dict


def parse_fname_and_pocc_id_from_path(vasprun_path: str, base_dir: str) -> tuple[str, int] | None:
    """
    Parses a vasprun.xml.static.xz path to extract fname_base and pocc_id.
    Assumes path structure like:
    <base_dir>/<fname_base>/<ARUN.POCC_XX_HXCZ>/vasprun.xml.static.xz
    or
    <base_dir>/<fname_base>/<pocc_id_int>/vasprun.xml.static.xz
    """
    relative_path = os.path.relpath(vasprun_path, base_dir)
    parts = relative_path.split(os.sep)

    # The path components are relative to base_dir.
    # So, parts[-1] is 'vasprun.xml.static.xz'
    # parts[-2] is the pocc_id_dir (e.g., '1' or 'ARUN.POCC_01_H0C0')
    # parts[-3] is the fname_base_dir (e.g., 'A_cI2_229_a.A:POCC_P0-0.2xA-0.2xB-0.2xC-0.2xD-0.2xE')

    if len(parts) < 3: # Expect at least fname_base/pocc_id/vasprun.xml.static.xz
        # This warning is now more specific to the parsing logic
        logger.debug(f"Path {vasprun_path} too short to parse fname_base and pocc_id using expected structure. Parts: {parts}")
        return None

    pocc_id_dir = parts[-2]
    fname_base_dir = "/"+ "/".join(parts[0:-2])
    # print(fname_base_dir)

    # Try to parse pocc_id from the directory name
    pocc_id_match = re.match(r"ARUN\.POCC_(\d+)_.*", pocc_id_dir)
    if pocc_id_match:
        pocc_id = int(pocc_id_match.group(1))
    else:
        try:
            pocc_id = int(pocc_id_dir)
        except ValueError:
            logger.debug(f"Could not parse pocc_id from directory name '{pocc_id_dir}' in path {vasprun_path}. Skipping.")
            return None

    fname_base = fname_base_dir
    return fname_base, pocc_id

def get_rmsd(atoms1, atoms2):
    """Calculate RMSD between two ASE Atoms objects using pymatgen."""
    str1 = AseAtomsAdaptor.get_structure(atoms1)
    str2 = AseAtomsAdaptor.get_structure(atoms2)
    matcher = StructureMatcher()
    result = matcher.get_rms_dist(str1, str2)
    if result is None:
        return np.nan
    return result[0]
