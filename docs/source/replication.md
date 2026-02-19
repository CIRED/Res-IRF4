# Replication Package

This page provides instructions for reproducing the results presented in the associated publication.

## Paper reference

> Vivier, L. & Giraudet, L.-G. (2024). *Is France on Track for Decarbonizing Its Residential Sector? Assessing Recent Policy Changes and the Way Forward.* [HAL: hal-04510798](https://hal.science/hal-04510798)

## Software and data availability

| Item | Details |
|------|---------|
| Source code | [github.com/CIRED/Res-IRF4](https://github.com/CIRED/Res-IRF4) |
| Archived version | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10405492.svg)](https://doi.org/10.5281/zenodo.10405492) |
| License | GNU General Public License v3 |
| Language | Python 3.8 |
| Operating systems | Linux, macOS, Windows |

## Software requirements

All dependencies are pinned in `environment.yml`. Key packages:

- Python 3.8
- pandas 1.4.4, numpy 1.23.3, scipy 1.9.3
- matplotlib 3.5.3, seaborn 0.12.0

## Setup

```bash
git clone https://github.com/CIRED/Res-IRF4.git
cd Res-IRF4
conda env create -f environment.yml
conda activate Res-IRF4
```

## Reproducing results

### Main simulation

```bash
python -m project.main -c project/config/config.json
```

Results are written to `project/output/DDMMYYYY_HHMM/` with a `summary_run.pdf` and per-scenario CSV outputs.

### Paper section mapping

For the paper-oriented workflow (market failures, optimal subsidies, realistic policies, friction variants, distortion checks), use:

- [`replication_paper_map.md`](replication_paper_map.md)

### Configuration files

The `project/config/` directory contains configuration files for different analyses. Each config file defines one or more scenarios that can be run and compared.

### Policy scenarios

Policy definitions are in `project/input/policies/`. Each policy file specifies the instruments, their parameters, and time horizons.

## Input data

All input data are included in the repository under `project/input/`:

| Directory | Content |
|-----------|---------|
| `stock/` | Building stock data (based on SDES 2018) |
| `energy/` | Energy prices, carbon emissions, renewable gas scenarios |
| `investment/` | Investment costs, market shares, discount rates |
| `technical/` | Technical parameters (efficiencies, costs, lifetimes) |
| `macro/` | Macroeconomic data (income, population, construction) |
| `climatic/` | Climate data (heating degree days) |
| `policies/` | Policy instrument definitions and scenarios |
| `resources_dir/` | Reference data and static scenario parameters |

See the [Data Dictionary](data_dictionary.md) for detailed variable descriptions.

## Expected outputs

A successful run produces:

- `summary_run.pdf`: multi-page PDF with key indicators
- Per-scenario `output.csv` files with annual time series of:
  - Energy consumption by fuel and sector
  - Renovation flows and stock performance distribution
  - Policy costs and effectiveness indicators
  - Carbon emissions
- Comparison plots in `img/` directories

## Citation

If you use this software, please cite:

```bibtex
@article{vivierFranceTrack2024,
  author = {Vivier, Lucas and Giraudet, Louis-Ga{\"e}tan},
  title = {Is {France} on Track for Decarbonizing Its Residential Sector? {Assessing} Recent Policy Changes and the Way Forward.},
  year = {2024},
  url = {https://hal.science/hal-04510798},
}
```

See also [`CITATION.cff`](https://github.com/CIRED/Res-IRF4/blob/master/CITATION.cff) for machine-readable citation metadata.
