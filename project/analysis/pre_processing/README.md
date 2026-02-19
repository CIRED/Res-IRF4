# Pre-processing

Data preparation, calibration, and scenario setup.

Each domain folder contains notebooks and local `data/` and `runs/` directories (not tracked in git).

## Notebooks

### `stock_preparation/`

| Notebook | Purpose |
|----------|---------|
| `prepare_sdes_stock_from_raw_inputs.ipynb` | Clean/harmonize SDES raw files and build aggregated stock |
| `visualize_sdes_stock_distribution.ipynb` | Stock distribution figures and tables |
| `validate_profeel_epc_estimation.ipynb` | Validate Profeel EPC transformation logic |
| `match_profeel_with_sdes_stock.ipynb` | Align Profeel with SDES categories |
| `compare_profeel_and_sdes_outputs.ipynb` | Compare SDES vs Profeel-derived indicators |

### `model_calibration/`

| Notebook | Purpose |
|----------|---------|
| `calibrate_ces_utility_function.ipynb` | Calibrate CES utility behavior parameters |
| `calibrate_hidden_cost_sensitivity.ipynb` | Hidden-cost sensitivity calibration scenarios |
| `plot_hidden_cost_sensitivity_results.ipynb` | Plot hidden-cost sensitivity results |
| `calibrate_insulation_market_share.ipynb` | Calibrate insulation market shares (IPF) |
| `calibrate_thermal_comfort_response.ipynb` | Thermal comfort response calibration |

### `scenario_setup/`

| Notebook | Purpose |
|----------|---------|
| `prepare_climate_time_series_inputs.ipynb` | Prepare annual/monthly/daily climate input series |
| `visualize_input_assumptions.ipynb` | Visualize scenario assumptions (energy prices, subsidies) |
| `generate_sensitivity_configuration_set.ipynb` | Generate sensitivity config sets |
| `evaluate_static_profitability_scenarios.ipynb` | Static profitability and subsidy diagnostics |

## Recommended run order

1. `stock_preparation/prepare_sdes_stock_from_raw_inputs.ipynb`
2. `stock_preparation/validate_profeel_epc_estimation.ipynb`
3. `stock_preparation/match_profeel_with_sdes_stock.ipynb`
4. Calibration notebooks in `model_calibration/`
5. Setup notebooks in `scenario_setup/`
