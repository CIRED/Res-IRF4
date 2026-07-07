# Config Layout

This folder is organized by use-case so it is easy to find the right run configuration.

## Directory map
- `project/config/config.json`: default baseline entry.
- `project/config/analysis/policy_distortion_analysis.json`, `project/config/validation.json`: analysis/validation runs.
- `project/config/assessment/`: policy-assessment run definitions.
- `project/config/test/`: test-only run entries.
- `project/config/market_failures/`: market-failure and related interaction runs.
- `project/config/market_failures/standalone/`: standalone policy-friction variants.
- `project/config/policies/realistic/`: realistic policy-mimicking runs.
- `project/config/policies/optimal/`: optimal-policy design runs.
- `project/config/sensitivity/`: sensitivity runs and templates.
- `project/config/uncertainty/`: uncertainty bundles by behavioral response level.
- `project/config/hidden_cost/`: hidden-cost calibration/sensitivity configs.
- `project/config/coupling/`: coupling-specific configs.
- `project/config/scenarios/`: S1-S4 scenario building blocks.
- `project/config/reference.json`, `project/config/reference_high_response.json`, `project/config/reference_low_response.json`: shared reference assumptions.

## Example/template files
- `project/config/sensitivity/sensitivity_list.example.json`: example/template parameter list for sensitivity setup.

## Naming convention
- `policy_*.json`: one policy variant run (single focus).
- `policies*.json`: a bundle comparing multiple policy scenarios.
- `policies_response_*.json` (in `uncertainty/`): same uncertainty bundle under different behavioral response calibrations.
- `*_bundle.json`: grouped scenario set generated from a policy-scenarios file.
- `standalone_*.json`: standalone interaction bundle for a specific domain.

## Market-failures files
- `project/config/market_failures/market_failures_bundle.json`: main market-failures scenario bundle.
- `project/config/market_failures/interaction_market_failures.json`: interaction-focused market-failures bundle.
- `project/config/market_failures/standalone/standalone_market_failures.json`: standalone interaction bundle for market-failures.

## Uncertainty response files
- `project/config/uncertainty/policies_response_reference.json`: reference-response calibration.
- `project/config/uncertainty/policies_response_high.json`: high-response calibration.
- `project/config/uncertainty/policies_response_low.json`: low-response calibration.

## Notes
- ADEME config was intentionally removed from this active layout.
- All remaining config JSON files reference existing `project/...` paths.
- VS Code launch profiles are aligned with this structure.
