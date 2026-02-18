# Quickstart

This guide walks you through installing Res-IRF4 and running your first simulation.

## Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/)
- Git

## Installation

```bash
git clone https://github.com/CIRED/Res-IRF4.git
cd Res-IRF4
conda env create -f environment.yml
conda activate Res-IRF4
```

## Running a simulation

Launch from the repository root (not from `/project`):

```bash
python -m project.main -c project/config/config.json
```

### Command-line options

| Flag | Description | Default |
|------|-------------|---------|
| `-c`, `--config` | Path to configuration file | `project/config/test/config.json` |
| `-d`, `--directory` | Path to policies directory | `project/config/policies` |
| `-a`, `--assessment` | Path to assessment config | `None` |
| `-y`, `--year` | Override end year | `None` |
| `-cpu`, `--cpu` | Number of CPUs for parallel runs | `6` |

## Understanding the output

Results are saved in `project/output/DDMMYYYY_HHMM/`:

- `summary_run.pdf` -- aggregated results summary
- One folder per scenario with:
  - `output.csv` -- detailed results (readable with Excel)
  - `img/` -- scenario-specific figures
- `img/` at the root -- cross-scenario comparison plots

## Modifying a scenario

Configuration files are JSON dictionaries in `project/config/`. The default `config.json` references a base scenario:

```json
{
  "Reference": {
    "file": "project/config/reference.json",
    "end": 2020,
    "policies": {
      "file": "project/input/policies/policies_calibration.json"
    }
  }
}
```

Key parameters you can modify:

- `"end"`: simulation end year (e.g., `2050`)
- `"policies"`: path to a policy configuration file
- `"simple"`: simplification options (`"quintiles": true` for faster runs)

## Memory and performance

Each scenario requires up to 2 GB of RAM. For laptops, limit parallel runs to 3 scenarios.
Use `"quintiles": true` and a restricted `"heating_system"` list in the `"simple"` block for lighter simulations.
