# Replication Package

This page documents how to reproduce published Res-IRF4 results in a reproducible way.

## Reference publication

> Vivier, L. and Giraudet, L.-G. (2024). *Is France on Track for Decarbonizing Its Residential Sector? Assessing Recent Policy Changes and the Way Forward.* [HAL: hal-04510798](https://hal.science/hal-04510798)

## Software and data availability

| Item | Details |
|------|---------|
| Source code | [github.com/CIRED/Res-IRF4](https://github.com/CIRED/Res-IRF4) |
| Archived release | [10.5281/zenodo.10405492](https://doi.org/10.5281/zenodo.10405492) |
| License | GNU GPL v3 |
| Python | 3.8 |

## Setup

```bash
git clone https://github.com/CIRED/Res-IRF4.git
cd Res-IRF4
conda env create -f environment.yml
conda activate Res-IRF4
```

## Minimum replication run

```bash
python -m project.main -c project/config/config.json
```

Expected results:

- a new folder `project/output/<timestamp>/`
- `summary_run.pdf` in that run folder
- scenario subfolders with `output.csv`
- scenario and comparison figures in `img/`

## Benchmark expectations

Exact runtime depends on hardware and scenario complexity, but for a standard workstation:

- small single-config runs: typically minutes
- scenario bundles and interaction grids: can take substantially longer (often tens of minutes to hours)
- memory: can approach ~2 GB per scenario in detailed settings

## Validation checks after a run

1. No Python exception is printed during execution.
2. `summary_run.pdf` exists in the run directory.
3. Each scenario directory contains `output.csv`.
4. Output CSVs include annual rows up to the configured `end` year.

## Paper-oriented workflow

Use [Paper Replication Map](replication_paper_map.md) to map each result block to:

- exact config files
- preprocessing notebooks (if scenario combinations must be regenerated)
- postprocessing notebooks for final figures/tables

## Input data coverage

All required inputs are versioned in `project/input/`, including:

- stock, technical, macro, climatic and energy inputs
- policy definitions and policy scenario sets

See [Data Dictionary](data_dictionary.md) for schema-level details.

## Troubleshooting

### Run fails immediately

- confirm the active environment: `conda activate Res-IRF4`
- verify Python version: `python --version` should be `3.8.x`

### Run is too slow or memory-heavy

- reduce parallelism (`-cpu` flag)
- use simpler scenario settings (for example quintiles mode in config)

### Outputs look incomplete

- inspect the run log for scenario-specific failures
- rerun a single config first, then scale to larger batches

## Citation

If you use Res-IRF4 outputs, cite both:

- the software metadata in `CITATION.cff`
- the associated publication listed above
