# Replication Package

This page documents how to reproduce published Res-IRF4 results in a reproducible way.

## Reference publication

> Vivier, L. and Giraudet, L.-G. (2024). *Is France on Track for Decarbonizing Its Residential Sector? Assessing Recent Policy Changes and the Way Forward.* [HAL: hal-04510798](https://hal.science/hal-04510798)

## Software and data availability

| Item | Details |
| --- | --- |
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

## Post-simulation indicator calculation

Each scenario produces a standalone `output.csv` (variables × years). Cross-scenario
indicators — cost-benefit analysis, cost-effectiveness, NPV, social welfare decomposition —
require a counterfactual and are therefore computed in a **separate step** after all
scenarios have run.

### What this step produces

Written to `<run_folder>/policies/`:

| File | Content |
| --- | --- |
| `comparison.csv` | Discounted differences vs. Reference for every scenario (variables × scenarios) |
| `indicator.csv` | Compact indicator table: NPV, investment, energy/emission savings, cost-effectiveness ratios, leverage, margins (metrics × scenarios) |
| `summary_assessment.csv` | Aggregated policy assessment summary |
| `social_welfare_*.png` | Stacked-bar CBA decomposition figures (total and annual, standard and horizontal variants) |

### Automatic execution (small runs, ≤ 10 scenarios)

When a run contains **at most 10 scenarios**, `indicator_policies()` is called automatically
at the end of `main.py` ([main.py:453–462](../../project/main.py)), using `Reference` as the
counterfactual:

```python
# project/main.py — triggered when output == 'full' and len(scenarios) <= 10
_, indicator = indicator_policies(result, folder, config_policies, policy_name=policies_name)
```

The `social_welfare_*.png` figures are saved at this point (`figure=True` by default).

### Deferred execution (large runs, > 10 scenarios)

When a run contains **more than 10 scenarios** (e.g. full factorial interaction grids),
`main.py` sets `output_compare = 'none'` ([main.py:439–440](../../project/main.py)) and skips
this step entirely. `indicator_policies()` must then be run explicitly in the
post-processing notebook:

```text
project/analysis/post_processing/policy_decomposition/analyze_factorial_combinations.ipynb
```

Cell 7 calls `indicator_policies(..., figure=False)` to rebuild `indicator.csv`, then reads
it back and merges it with `scenarios_description` for the interaction and sensitivity
analyses. A later cell (Cell 24) calls it again with `figure=True` on grouped subsets to
produce the social welfare figures.

## Factorial interaction runs: scenario description file

Full factorial runs (e.g. `interaction_optimal_pp`, `interaction_mf`,
`interaction_current_pp`) can involve hundreds of scenarios (2^n combinations of n policy
dimensions). Making economic sense of those scenarios requires a **scenario description
file** that maps each scenario ID to the policy features that define it.

### Generating `policies_scenarios_description.csv`

This file is generated **before the simulation run** by one of the pre-processing notebooks:

```text
project/input/policies/interaction_optimal_pp/create_policies_scenarios.ipynb
project/input/policies/interaction_mf/create_policies_scenarios.ipynb
project/input/policies/interaction_current_pp/create_policies_scenarios.ipynb
```

Each notebook builds the full factorial product over the policy dimensions. For example,
with 7 binary dimensions (policy on / off), it produces 2^7 = 128 scenarios:

| Scenario | subsidy_emission | subsidy_health_cost | subsidy_landlord | … |
| --- | --- | --- | --- | --- |
| S0 | subsidy_emission | subsidy_health_cost | subsidy_landlord | … |
| S1 | subsidy_emission | subsidy_health_cost | no_subsidy_landlord | … |
| … | … | … | … | … |
| S127 | no_subsidy_emission | no_subsidy_health_cost | no_subsidy_landlord | … |

The resulting CSV is copied into the run folder alongside `output.csv` files and is
the key input that gives economic meaning to the raw scenario IDs.

### Role in post-processing

In `analyze_factorial_combinations.ipynb`, Cell 6 loads this file:

```python
scenarios_description = pd.read_csv(
    os.path.join(folder, "policies_scenarios_description.csv"), index_col=[0]
)
```

It is then merged with `indicator.csv` (Cell 7) to produce a unified `data` frame
(scenarios × features + indicators) that drives all downstream analysis:
interaction plots, sensitivity (Sobol / Shapley), and cost-effectiveness mappings.

### Specific scenario comparisons

Not all 2^n scenarios are meaningful to display directly. Cell 24 selects interpretable
subsets by filtering on the `Group` column (assigned during pre-processing to label
scenarios of economic interest):

```python
# Keep only labelled scenarios, drop pure interaction combinations
scenarios = data.index[data["Group"] != "Other"]
grouped_outputs = {data.loc[name, "Group"]: output for name, output in dict_output_subset.items()}
```

Assessment-specific exclusions then refine the selection further (e.g. excluding package
scenarios or friction-free benchmarks). The resulting `grouped_outputs` dict — keyed by
human-readable `Group` labels rather than raw scenario IDs — is what gets passed to
`indicator_policies(..., figure=True)` to produce the social welfare and CBA figures.

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
