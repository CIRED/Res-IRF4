# Data Dictionary

This page documents the input data used by Res-IRF4. All data files are located under `project/input/`.

## Building stock (`stock/`)

The building stock is based on SDES (Service des Donnees et Etudes Statistiques) 2018 data.

Each dwelling is characterized by:

| Dimension | Categories | Description |
|-----------|------------|-------------|
| Energy performance | A--G (existing), LE, NZ (new) | EPC label (Diagnostic de Performance Energetique) |
| Housing type | Single-family, Multi-family | Building typology |
| Occupancy status | Owner-occupied, Rented, Social housing | Decision-maker type |
| Heating system | Electricity, Natural gas, Fuel oil, Fuel wood, Heat pump, etc. | Main heating fuel/technology |
| Income class | Deciles (D1--D10) or Quintiles (Q1--Q5) | Household disposable income category |

The model contains up to 1,080 dwelling types from the combination of these dimensions.

## Energy data (`energy/`)

| File | Description | Unit |
|------|-------------|------|
| Energy price trajectories | Fuel-specific price scenarios | EUR/kWh |
| Carbon emissions | Emission factors by fuel | gCO2/kWh |
| Renewable gas | Renewable gas penetration scenarios | Share (%) |

## Investment data (`investment/`)

| Category | Description |
|----------|-------------|
| Market shares | Initial distribution of heating systems and insulation choices |
| Discount rates | Income-differentiated discount rates for private housing; 4% for social housing |
| Renovation costs | Cost matrix linking initial to final EPC label (EUR/m2) |
| Construction costs | Costs for new buildings at LE and NZ levels (EUR/m2) |

## Technical parameters (`technical/`)

| Category | Description |
|----------|-------------|
| Heating system efficiency | Efficiency by technology and vintage |
| Insulation costs | Component-level insulation costs (wall, floor, roof, windows) |
| Lifetimes | Expected lifetimes for heating systems and insulation measures |
| Learning rates | Cost reduction rates for renovation (10%) and construction (15%) technologies |

## Macroeconomic data (`macro/`)

| Variable | Source | Description |
|----------|--------|-------------|
| Population | INSEE projections | Annual population growth (~0.3%/year) |
| Household income | INSEE, extrapolated at 1.2%/year | Disposable income by quintile/decile |
| Construction needs | Derived from population and income | Annual new dwelling construction |

## Climate data (`climatic/`)

Heating degree days (HDD) and temperature profiles used to compute conventional energy consumption.

## Policy definitions (`policies/`)

Policy instruments are defined in JSON files. Each policy specifies:

| Parameter | Description |
|-----------|-------------|
| `start`, `end` | Policy activation period |
| `value` | Subsidy rate, tax rate, or regulatory threshold |
| `target` | Eligibility conditions (income, EPC label, housing type) |

Supported policy instruments include:

- **MaPrimeRenov (MPR)**: Income-targeted renovation subsidies (multiple variants)
- **CEE (Certificats d'Economies d'Energie)**: White certificate obligations
- **CITE**: Tax credit for energy transition
- **Carbon tax**: Applied to natural gas and heating oil
- **Zero-interest loans (ZIL)**: Targeted at deep renovations
- **Reduced VAT**: 5.5% rate on renovation works
- **Building code**: Minimum performance standards for new construction
- **Energy restrictions**: Bans on specific heating systems or EPC labels

## Static scenario parameters (`resources_dir/`)

The file `scenario_static.json` defines assessment parameters including:

- Social and private discount rates
- Policy measure definitions (deep insulation, global renovation)
- Cost-benefit parameters (carbon value, health impacts)

## Calibration data

The model is calibrated to reproduce:

- Total energy consumption by fuel (source: CEREN)
- Renovation rates by housing type (source: OPEN 2016, USH 2017)
- Market shares of energy efficiency upgrades (source: PUCA 2015)

See [Input Res-IRF version 3.0](input_2012.md) for detailed calibration data documentation.
