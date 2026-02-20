# Res-IRF4 Overview

Res-IRF4 is an agent-based simulation model for the French residential building sector, focused on space-heating demand, renovation decisions, heating technology choices, and policy impacts.

## What the model produces

Typical outputs include:

- annual energy consumption by fuel
- renovation and construction flows
- stock performance distribution over time
- policy cost and effectiveness indicators
- emissions and fuel-poverty related indicators

## Repository structure

- `project/`: model code, configs, input data, analysis scripts
- `project/config/`: scenario and experiment configurations
- `project/input/`: policy, technical, energy, and macro input datasets
- `project/output/`: timestamped run results
- `docs/source/`: documentation sources

## Recommended workflow

1. Follow the [Quickstart](quickstart.md) for a first successful run.
2. Read the [Technical documentation](technical_documentation.md) for model structure.
3. Use [Replication package](replication.md) and [Paper replication map](replication_paper_map.md) for publication workflows.
4. Use [API reference](modules.rst) when extending internals.

## Citation

If you use Res-IRF4, cite the software and paper metadata in `CITATION.cff` and the references listed in [Replication package](replication.md).
