# Changelog

## Version 4.0

Major update from version 3.0, with significant extensions to the model structure, data, and policy coverage.

### New features

- **Component-level insulation modeling**: The thermal module (`thermal.py`) models insulation at the component level (wall, floor, roof, windows) rather than aggregate EPC transitions only.
- **Heating system switching**: Households can now switch between heating technologies (e.g., from gas boiler to heat pump), not only upgrade insulation.
- **Extended policy instruments**: New policy types including MaPrimeRenov (multiple variants: serenity, efficacity, performance), CEE 2024 updates, heater bans, and minimum performance obligations.
- **Multiprocessing support**: Parallel execution of multiple scenarios via Python's `multiprocessing` module.
- **Coupling capabilities**: Module for coupling Res-IRF with external models or running integrated scenario analyses.
- **Space heating utility**: Added utility-based framework for modeling heating comfort decisions.

### Data updates

- **Building stock**: Updated to SDES 2018 data (from Phebus 2012 in v3.0).
- **Policy parameters**: Updated to reflect current French policy landscape (2024).
- **Energy prices**: Updated price trajectories and carbon emission scenarios.

### Technical improvements

- **Flexible configuration system**: JSON-based configs with scenario inheritance, policy composition, and multi-header support.
- **Automated output**: Summary PDF generation, cross-scenario comparison plots.
- **Code structure**: Complete rewrite with clearer separation of concerns across modules.

## Version 3.0

Described in {cite:ps}`giraudetPoliciesLowcarbonAffordable2021`.

- Recoded from Scilab to Python.
- Introduced income heterogeneity (quintiles/deciles).
- Added rebound effect modeling linked to income.
- Calibrated on Phebus 2012 survey data.
- Multi-criteria policy evaluation (effectiveness, cost-effectiveness, leverage, distributional effects).
- Six peer-reviewed publications.

See [Input Res-IRF version 3.0](../legacy/input_2012.md) and [Simulation and sensitivity analysis](../legacy/simulation_2012.md) for detailed v3.0 documentation.

```{bibliography}
:filter: False
```
