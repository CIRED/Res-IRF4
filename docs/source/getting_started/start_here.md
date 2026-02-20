# Start Here

This documentation is organized for **Res-IRF4 (v4)** workflows first.

```{note}
Legacy v3.0 and 2012-calibration documentation is still available in the
[Legacy Archive](../legacy/index.md), but it is intentionally separated from the main v4 workflow.
```

## Choose your path

### I want to run the model quickly

1. Follow the [Quickstart](quickstart.md).
2. Run one config and verify outputs are created under `project/output/`.
3. Use [Installation help](help_installation.md) only if setup issues appear.

### I want to reproduce paper results

1. Read the [Replication Package](../replication/replication.md) for setup and output expectations.
2. Use the [Paper Replication Map](../replication/replication_paper_map.md) for exact config-to-result mapping.
3. Use notebooks in `project/analysis/post_processing/` for figure and table generation.

### I want to develop or extend Res-IRF4

1. Read [API Reference](../developer/modules.rst) for module responsibilities.
2. Follow [Style Guide](../developer/style_guide.md) for documentation standards.
3. Use [Contributing](../developer/contributing.md) for local docs checks and PR expectations.

## Suggested reading order

1. [Quickstart](quickstart.md)
2. [Technical documentation](../model/technical_documentation.md)
3. [Replication package](../replication/replication.md)
4. [API reference](../developer/modules.rst)
