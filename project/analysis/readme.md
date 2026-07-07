# Analysis

## Overview

```text
project/analysis/
├── pre_processing/     Data preparation and calibration
├── static/             Descriptive statistics and model validation
├── others/             Policy analysis (mitigation, elasticity)
└── post_processing/    Simulation output analysis
```

## Pipeline

```text
pre_processing/  →  static/  →  simulation  →  post_processing/
(prepare inputs)   (validate)    (run model)    (analyze outputs)
```

## Notebooks

### `static/`

| Notebook | Purpose |
|----------|---------|
| `description_stock.ipynb` | Building stock composition and consumption statistics |
| `static_validation.ipynb` | Thermal function validation (DPE A-G), consumption by period, static cost-efficiency |
| `climate.ipynb` | Climate variation impact on heating demand (2010-2019) |
| `hourly_consumption.ipynb` | Hourly heating demand profiles with smoothing |

### `others/`

| Notebook | Purpose |
|----------|---------|
| `mitigation_analysis.ipynb` | Mitigation potential, cost-efficiency, NPV, MAC curves, parameterized analysis, aggregated cost curves |
| `elasticity.ipynb` | Stochastic energy price scenario generation |

### `pre_processing/`

| Folder | Notebooks | Purpose |
|--------|-----------|---------|
| `stock_preparation/` | `prepare_sdes_stock_from_raw_inputs` → `validate_profeel_epc_estimation` → `match_profeel_with_sdes_stock` | Build SDES/Profeel stock inputs |
| `model_calibration/` | `calibrate_ces_utility_function`, `calibrate_hidden_cost_sensitivity`, `calibrate_insulation_market_share`, `calibrate_thermal_comfort_response` | Calibrate model components |
| `scenario_setup/` | `prepare_climate_time_series_inputs`, `visualize_input_assumptions`, `generate_sensitivity_configuration_set`, `evaluate_static_profitability_scenarios` | Prepare scenario assumptions and static analysis |

### `post_processing/`

| Folder | Notebooks | Purpose |
|--------|-----------|---------|
| `policy_decomposition/` | `summarize_scenario_run`, `analyze_factorial_combinations` | Factorial decomposition (Shapley, Sobol) of policy and market-failure interactions |
| `policy_assessment/` | `summarize_policy_portfolio`, `analyze_subsidy_allocation_gap`, `analyze_subsidy_distortion`, `plot_insulation_subsidy_design` | Evaluate policy instruments |
| `reporting/` | `build_comparison_figures_from_run`, `compare_assumption_sets_npv`, `build_stock_transition_figures` | Publication figures and tables |
| `elasticity/` | `estimate_long_term_price_elasticity` | Long-term price elasticity from sensitivity runs |

## Git tracking

Only source code is tracked: notebooks, `.py`, `.md`.

All generated content is gitignored: `data/`, `runs/`, `output/`, `figures/`, `archive/`, `_old/`, `*.pkl`.

Notebook outputs (plots, tables) are automatically stripped on commit via `nbstripout` (configured in `.gitattributes`). Local files keep their outputs for quick review without re-running.
