# Paper Replication Map (Configs + Notebooks)

This page maps each paper result block to:

1. exact simulation config file(s)
2. preprocessing step(s), when scenario files must be regenerated
3. postprocessing notebook(s)

## Execution conventions

Run from repository root:

```bash
python -m project.main -c <config_path>
```

Batch-run all configs in a folder:

```bash
python -m project.runs -d <config_folder>
```

Outputs are written to `project/output/<timestamp>/`.

## 1) Market failures

### 1.1 One-by-one effects + combined case

- Config: `project/config/market_failures/market_failures_bundle.json`
- Policy source: `project/input/policies/policies_mf.json`
- Postprocessing:
  - `project/analysis/post_processing/reporting/build_comparison_figures_from_run.ipynb`

### 1.2 Full interaction space

- Config: `project/config/market_failures/interaction_market_failures.json`
- Policy source: `project/input/policies/interaction_mf/policies_scenarios.json`
- Preprocessing (when variants change):
  - `project/input/policies/interaction_mf/create_policies_scenarios.ipynb`
- Postprocessing:
  - `project/analysis/post_processing/scenario_analysis/summarize_many_scenarios_run.ipynb`
  - `project/analysis/post_processing/scenario_analysis/analyze_policy_packages_across_runs.ipynb` with `assessment = "market_failures"`

### 1.3 Standalone interaction-aligned set

- Config: `project/config/market_failures/standalone/standalone_market_failures.json`
- Policy source: `project/input/policies/interaction_mf/policies_standalone.json`

## 2) Optimal subsidies

### 2.1 One-by-one effects + combined case

- Config: `project/config/policies/optimal/policy_optimal.json`
- Policy source: `project/input/policies/policies_optimal.json`
- Postprocessing:
  - `project/analysis/post_processing/reporting/build_comparison_figures_from_run.ipynb`

### 2.2 Interaction space

- Config: `project/config/policies/optimal/interaction_optimal_policy.json`
- Policy source: `project/input/policies/interaction_optimal_pp/policies_scenarios.json`
- Preprocessing:
  - `project/input/policies/interaction_optimal_pp/create_policies_scenarios.ipynb`
- Postprocessing:
  - `project/analysis/post_processing/scenario_analysis/analyze_policy_packages_across_runs.ipynb` with `assessment = "optimal_pp"`

### 2.3 Standalone interaction-aligned set

- Config: `project/config/policies/optimal/standalone_optimal_policy.json`
- Policy source: `project/input/policies/interaction_optimal_pp/policies_standalone.json`

## 3) Realistic policies

### 3.1 Interaction space

- Config: `project/config/policies/realistic/interaction_policy.json`
- Policy source: `project/input/policies/interaction_current_pp/policies_scenarios_reduced.json`
- Preprocessing:
  - `project/input/policies/interaction_current_pp/create_policies_scenarios_reduced.ipynb`
- Postprocessing:
  - `project/analysis/post_processing/scenario_analysis/analyze_policy_packages_across_runs.ipynb` with `assessment = "policies"`

## 4) Friction assumption comparisons

- Configs:
  - `project/config/market_failures/standalone/policy_friction.json`
  - `project/config/market_failures/standalone/policy_friction_biased.json`
  - `project/config/market_failures/standalone/policy_no_friction.json`
  - `project/config/market_failures/standalone/policy_no_friction_biased.json`
  - `project/config/market_failures/standalone/policy_friction_unbiased.json`
  - `project/config/market_failures/standalone/policy_friction_nocredit.json`
- Policy sources:
  - `project/input/policies/interaction_current_pp/policies_standalone_pp_reduced.json`
  - `project/input/policies/interaction_current_pp/policies_standalone_pp.json`
- Postprocessing:
  - `project/analysis/post_processing/reporting/compare_assumption_sets_npv.ipynb`

## 5) Final five-scenario comparison

- Config: `project/config/policies/realistic/policies.json`
- Postprocessing:
  - `project/analysis/post_processing/reporting/build_comparison_figures_from_run.ipynb`

## 6) Distortion and cross-block comparisons

### 6.1 Distortion-focused run

- Config: `project/config/analysis/policy_distortion_analysis.json`
- Postprocessing:
  - `project/analysis/post_processing/policy_assessment/analyze_subsidy_distortion_runs.ipynb`

### 6.2 Market-failures vs policies comparison

- Run both:
  - `project/config/market_failures/interaction_market_failures.json`
  - `project/config/policies/realistic/interaction_policy.json`
- Compare with:
  - `project/analysis/post_processing/scenario_analysis/analyze_policy_packages_across_runs.ipynb`

## Quick command block

```bash
# Market failures
python -m project.main -c project/config/market_failures/market_failures_bundle.json
python -m project.main -c project/config/market_failures/interaction_market_failures.json

# Optimal subsidies
python -m project.main -c project/config/policies/optimal/policy_optimal.json
python -m project.main -c project/config/policies/optimal/interaction_optimal_policy.json

# Realistic policies
python -m project.main -c project/config/policies/realistic/interaction_policy.json
python -m project.main -c project/config/policies/realistic/policies.json

# Distortion analysis
python -m project.main -c project/config/analysis/policy_distortion_analysis.json
```

## Validation checklist

After each run, verify:

1. a new folder appears in `project/output/`
2. scenario folders contain `output.csv`
3. expected postprocessing notebook can load the run output without missing-file errors
