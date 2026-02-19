# Quickstart

This page gets you from clone to first successful Res-IRF4 run.

## 1. Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/)
- Git

## 2. Setup

```bash
git clone https://github.com/CIRED/Res-IRF4.git
cd Res-IRF4
conda env create -f environment.yml
conda activate Res-IRF4
```

The expected conda environment name is `Res-IRF4` (from `environment.yml`).

## 3. Run your first simulation

Run from the repository root:

```bash
python -m project.main -c project/config/config.json
```

## 4. Check that the run succeeded

A successful run creates a timestamped folder under `project/output/`:

```bash
ls -1 project/output | tail -n 5
```

Inside the latest run folder, you should see:

- `summary_run.pdf`
- one directory per scenario
- scenario-level `output.csv` files
- `img/` folders with generated figures

## Useful CLI flags

| Flag | Description | Default |
|------|-------------|---------|
| `-c`, `--config` | Path to configuration file | `project/config/test/test.json` |
| `-y`, `--year` | Override simulation end year | unset |
| `-cpu`, `--cpu` | Number of CPUs for parallel runs | `6` |

For batch execution of multiple configs in a folder:

```bash
python -m project.runs -d project/config/policies/realistic
```

## Edit a scenario quickly

Main config files are in `project/config/`.

High-impact fields:

- `"end"`: simulation horizon (for example `2050`)
- `"file"`: inherit from a baseline config
- `"policies"`: inject policy sets
- `"simple"`: run simplifications (`"quintiles": true`, limited heating systems) for faster iterations

## Performance notes

- Memory can reach about 2 GB per scenario depending on detail level.
- On typical laptops, keep parallel runs to about 2-3 scenarios at once.

```{seealso}
- [Replication package](replication.md)
- [Installation help](help_installation.md)
```
