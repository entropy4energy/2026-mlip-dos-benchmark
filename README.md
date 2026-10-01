# 2026 MLIP DOS Benchmark

This repository provides data and scripts for the paper:
**"Structural benchmarks overlook density-of-states errors in
machine-learned interatomic potentials"**.

The scripts collect and analyze structural and electronic metrics for
machine-learned interatomic potential relaxations followed by
single-point density-functional calculations.

- **Input:** DFT reference calculations, MLIP-relaxed structures, and
  associated single-point DFT outputs.
- **Structural diagnostics:** RMSD, maximum displacement, topology
  classification, and symmetry-operation changes.
- **Electronic diagnostics:** Fermi-energy residuals and density-of-states
  cosine and Wasserstein distances.
- **Output:** Aggregated CSV data and LaTeX tables used in the
  manuscript and Supplementary Material.

## Repository Structure

```text
data/
  data.csv                         Aggregated benchmark dataset

scripts/
  collect_data.py                  Data collection and metric extraction
  generate_composition_table.py    Composition/prototype table generator
  generate_metrics_table.py        Structural/electronic metrics table generator
  utils.py                         Shared VASP, DOS, and distance utilities
```

## Requirements

- Python 3.13+
- [`uv`](https://docs.astral.sh/uv/) for dependency management

The scripts declare their Python dependencies in their `uv` headers.
No separate package installation is required when using `uv run`.

## Usage

Run commands from the repository root.

### Data Collection

```bash
uv run scripts/collect_data.py \
  -o data/data.csv \
  <ground_truth_dir> \
  <mlip_dir_1> <mlip_dir_2> ...
```

**Positional arguments:**

| Argument | Description |
|---|---|
| `ground_truth_dir` | Root directory containing DFT reference data |
| `mlip_dir_1`, `mlip_dir_2`, ... | Directories containing MLIP-relaxed data |

**Options:**

| Flag | Description |
|---|---|
| `-o OUTPUT`, `--output OUTPUT` | Output CSV path or output directory |
| `--test-run` | Process only the first 20 systems |
| `--dos-sigma FLOAT` | Gaussian broadening for the DOS |
| `--dos-smoothing FLOAT` | Smoothing parameter for DOS interpolation |
| `--dos-grid-points INT` | Number of points for the DOS energy grid |
| `--db-dir DIR` | Directory containing local `.aselmdb` files |
| `--max-workers INT` | Maximum number of parallel workers |

### Table Generation

```bash
uv run scripts/generate_composition_table.py
uv run scripts/generate_metrics_table.py
```

Both table-generation scripts read `data/data.csv` by default.

## Reference Parameters

The parameters used to compile the released `data/data.csv` on the
reference benchmarking system were:

| Parameter | Value |
|---|---|
| `ground_truth_dir` | `<ground_truth_dir>` |
| `mlip_dirs` | `<mlip_dir>/chgnet` |
|  | `<mlip_dir>/esen` |
|  | `<mlip_dir>/mace` |
|  | `<mlip_dir>/uma` |
| `--output` | `data/data.csv` |
| `--dos-sigma` | `0.05` |
| `--dos-smoothing` | `1e-5` |
| `--dos-grid-points` | `1000` |
| `--db-dir` | `<db_dir>` |
| `--max-workers` | `4` |

## Expected Input Layout

```text
ground_truth_dir/
  category/
    LIB/
      MaterialName/
        structure_label/
          vasprun.xml.static          or vasprun.xml.static.xz
          vasprun.xml.relax1          or vasprun.xml.relax1.xz
          vasprun.xml.relax2          or vasprun.xml.relax2.xz
          OUTCAR.static               or OUTCAR.static.xz
          OUTCAR.relax1               or OUTCAR.relax1.xz
          OUTCAR.relax2               or OUTCAR.relax2.xz
          CONTCAR
          POSCAR
          DOSCAR

mlip_dir/
  material_class_model_relaxed.aselmdb
  material_class_model_relaxed/
    structure_label/
      vasprun.xml.static              or vasprun.xml.static.xz
      OUTCAR.static                   or OUTCAR.static.xz
      CONTCAR
      DOSCAR
```

## Output Dataset

`data/data.csv` contains one row per system with columns for the DFT
reference and each MLIP-relaxed workflow.

Representative fields include:

| Field | Description |
|---|---|
| `catalog` | Library of the reference calculation (`LIB2` binaries, `LIB3` ternaries) |
| `species` | Species with their pseudopotential tags (e.g. `Ba_svI`) |
| `prototype` | Prototype label with its parameter-set number and species order (e.g. `A5B8_cI52_217_ce_cg-001.AB`) |
| `num_atoms` | Number of atoms in the structure |
| `gt_energy` | DFT reference energy |
| `<model>_energy` | Single-point DFT energy on the MLIP-relaxed structure |
| `<model>_rmsd` | RMSD between MLIP- and DFT-relaxed structures |
| `<model>_max_dist` | Maximum displacement after structural alignment |
| `<model>_dos_cosine_dist` | Density-of-states cosine distance |
| `<model>_dos_wasserstein_dist` | Density-of-states Wasserstein distance |
| `<model>_topology` | `preserved`, `degraded`, `broken`, or `error` |
| `<model>_symmetry_broken` | Whether symmetry-operation count changes |
| `<model>_symmetry_reason` | Symmetry-operation count comparison |
| `<model>_energy_mae` | Absolute energy residual in eV/atom |

This repository is intended as a minimal working reference and starting
point for MLIP-based density-of-states benchmarking workflows.

## Citation

Published as:

> S.-Y. Tseng, G. Han, T. Li, X. Xu, J. Hu, G. Qiu, and C. Oses,
> *Structural benchmarks overlook density-of-states errors in
> machine-learned interatomic potentials*,
> APL Mach. Learn. **4**, 036117 (2026).
> [doi:10.1063/5.0339982](https://doi.org/10.1063/5.0339982)

Open access; the published version of record is available at the DOI above.

## License

Copyright © 2026 Entropy for Energy Lab, Johns Hopkins University.

Released under the MIT License; see `LICENSE`.
