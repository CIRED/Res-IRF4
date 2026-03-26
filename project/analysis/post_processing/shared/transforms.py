"""Common transformations used across post-processing notebooks."""
from __future__ import annotations

import pandas as pd


def add_annualized_rows(df: pd.DataFrame, row_mapping: dict[str, str]) -> pd.DataFrame:
    """Add annualized rows at end-year from cumulative yearly rows.

    Parameters
    ----------
    df:
        Scenario output table indexed by variable names with year columns.
    row_mapping:
        Mapping of {target_row_name: source_row_name}.
    """
    if df.empty:
        return df

    out = df.copy()
    start_year = int(out.columns[0])
    end_col = out.columns[-1]
    duration = int(end_col) - start_year
    if duration <= 0:
        raise ValueError("Cannot annualize rows when end year is not after start year.")

    for target_row, source_row in row_mapping.items():
        if source_row in out.index:
            out.loc[target_row, end_col] = out.loc[source_row, :].sum() / duration

    return out


def extract_end_year_values(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """Extract selected variables at end-year from a (Scenarios, Variables) dataframe."""
    if df.empty:
        return pd.DataFrame(columns=variables)

    end_col = df.columns[-1]
    extract = df[df.index.get_level_values("Variables").isin(variables)].loc[:, end_col].unstack("Variables")
    return extract.reindex(columns=variables)


def end_year_reference_delta(df: pd.DataFrame, variables: list[str], reference_name: str = "Reference") -> pd.DataFrame:
    """Compute end-year deltas versus reference for selected variables.

    Expects a dataframe indexed by (Scenarios, Variables) with year columns.
    """
    end = df.columns[-1]
    extract = df[df.index.get_level_values("Variables").isin(variables)].loc[:, end].unstack("Variables")
    reference = extract[extract.index.get_level_values("Scenarios") == reference_name]
    return extract[variables].sub(reference[variables].values)


def describe_with_extrema(df: pd.DataFrame) -> pd.DataFrame:
    """Return describe() plus min/max scenario names."""
    return pd.concat(
        [
            df.describe(),
            df.idxmax().rename("max_scenario").to_frame().T,
            df.idxmin().rename("min_scenario").to_frame().T,
        ],
        axis=0,
    )


def describe_with_reference_and_extrema(
    df: pd.DataFrame, reference_name: str = "Reference", reference_label: str = "reference"
) -> pd.DataFrame:
    """Return describe(), reference row (if available), plus max/min scenario names."""
    reference_df = pd.DataFrame(columns=df.columns)
    if reference_name in df.index:
        reference_df = df.loc[[reference_name]].rename(index={reference_name: reference_label})

    return pd.concat(
        [
            df.describe(),
            reference_df,
            df.idxmax().rename("max_scenario").to_frame().T,
            df.idxmin().rename("min_scenario").to_frame().T,
        ],
        axis=0,
    )


def keep_scenarios(bundle: dict[str, pd.DataFrame], scenarios: list[str] | None) -> dict[str, pd.DataFrame]:
    """Filter bundle keys while preserving requested order."""
    if not scenarios:
        return bundle
    return {scenario: bundle[scenario] for scenario in scenarios if scenario in bundle}


def add_subsidies_gap(bundle: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Add a ``Subsidies Gap`` column (Distortion - Subsidies) to each table."""
    result = {}
    required = {"Subsidies", "Distortion"}

    for scenario, df in bundle.items():
        missing = required.difference(df.columns)
        if missing:
            raise KeyError("Missing columns for {}: {}".format(scenario, sorted(missing)))

        enriched = df.copy()
        enriched["Subsidies Gap"] = enriched["Distortion"] - enriched["Subsidies"]
        result[scenario] = enriched

    return result


def extract_indicator_values(
    table: pd.DataFrame,
    pattern: str,
    select_value,
    reference: str | None = None,
    reference_multiplier: float = 1e3,
) -> pd.Series:
    """Extract indicator rows matching a regex and return one series per metric."""
    index = table.index.to_series().astype(str)
    mask = index.str.match(pattern)
    filtered = table.loc[mask].copy()

    if filtered.empty:
        return pd.Series(dtype=float)

    filtered.index = index[mask].str.extract(pattern, expand=False)

    if reference is not None:
        ref_mask = index.str.match(reference)
        filtered_ref = table.loc[ref_mask].copy()
        filtered_ref.index = index[ref_mask].str.extract(reference, expand=False)
        filtered = filtered.divide(filtered_ref * reference_multiplier)
        filtered = filtered.dropna(axis=0, how="all")

    if str(select_value).lower() == "avg":
        return filtered.mean(axis=1)

    column = str(select_value)
    if column not in filtered.columns:
        raise KeyError("Column '{}' not found in indicator table.".format(column))

    return filtered[column]


def build_indicator_summary(
    output_bundle: dict[str, pd.DataFrame],
    pattern: str,
    select_value,
    reference: str | None = None,
    reference_multiplier: float = 1e3,
) -> pd.DataFrame:
    """Build a metric x scenario table from output.csv files."""
    result = {}

    for scenario, table in output_bundle.items():
        result[scenario] = extract_indicator_values(
            table=table,
            pattern=pattern,
            select_value=select_value,
            reference=reference,
            reference_multiplier=reference_multiplier,
        )

    final = pd.DataFrame(result)
    final.index.name = "Metric"
    final.columns.name = "Scenario"
    return final


def parse_subsidy_indicator_index(summary: pd.DataFrame) -> pd.DataFrame:
    """Parse the Metric index of a subsidy summary into structured columns.

    Expects index values like ``"Single-family - Owner-occupied - C1"``.
    Returns a DataFrame with columns: Housing, Status, Income, Value (per scenario).
    """
    import re

    records = []
    for metric in summary.index:
        parts = [p.strip() for p in metric.split(" - ")]
        if len(parts) == 3:
            housing, status, income = parts
        elif len(parts) == 2:
            housing, status = parts
            income = "Total"
        else:
            continue
        for scenario in summary.columns:
            records.append({
                "Housing": housing,
                "Status": status,
                "Income": income,
                "Scenario": scenario,
                "Value": summary.loc[metric, scenario],
            })
    return pd.DataFrame(records)


def subsidy_summary_by_housing_status(parsed: pd.DataFrame) -> pd.DataFrame:
    """Pivot to Housing-Status rows x Scenario columns (total only, no income breakdown)."""
    total = parsed[parsed["Income"] == "Total"].copy()
    total["Label"] = total["Housing"] + " - " + total["Status"]
    return total.pivot(index="Label", columns="Scenario", values="Value")


def subsidy_summary_by_income(parsed: pd.DataFrame) -> pd.DataFrame:
    """Pivot to Income rows x Scenario columns (summed across housing-status)."""
    detail = parsed[parsed["Income"] != "Total"].copy()
    grouped = detail.groupby(["Income", "Scenario"])["Value"].sum().reset_index()
    return grouped.pivot(index="Income", columns="Scenario", values="Value")


def subsidy_delta_relative_to_reference(
    parsed: pd.DataFrame,
    reference_scenario: str = "OptimalSubsidies",
) -> pd.DataFrame:
    """Compute subsidy difference of each scenario vs. a reference, by Housing x Status x Income.

    Returns a long-form DataFrame with columns: Housing, Status, Income, Scenario, Delta, DeltaPct.
    Delta = Value - RefValue (positive means scenario spends more than reference).
    DeltaPct = (Value - RefValue) / RefValue * 100.
    """
    detail = parsed[parsed["Income"] != "Total"].copy()
    ref = detail[detail["Scenario"] == reference_scenario].rename(columns={"Value": "RefValue"})
    ref = ref[["Housing", "Status", "Income", "RefValue"]]

    merged = detail.merge(ref, on=["Housing", "Status", "Income"], how="left")
    merged["Delta"] = merged["Value"] - merged["RefValue"]
    merged["DeltaPct"] = (merged["Value"] - merged["RefValue"]) / merged["RefValue"] * 100
    return merged[["Housing", "Status", "Income", "Scenario", "Delta", "DeltaPct"]]


def build_ad_valorem_ratio(
    output_bundle: dict[str, pd.DataFrame],
    subsidy_pattern: str,
    investment_pattern: str,
    select_value,
    subsidy_to_investment_scale: float = 1e-3,
) -> pd.DataFrame:
    """Compute an implicit ad valorem rate: subsidies / investment by Housing x Status.

    Parameters
    ----------
    output_bundle:
        Scenario name -> output table.
    subsidy_pattern:
        Regex matching subsidy rows.  Must capture the Housing-Status label.
    investment_pattern:
        Regex matching investment rows.  Must capture the same Housing-Status label.
    select_value:
        ``"avg"`` or a year column.
    subsidy_to_investment_scale:
        Factor to align units.  Default ``1e-3`` converts Million€ subsidies
        to Billion€ so the ratio with Billion€ investment is dimensionless.

    Returns
    -------
    DataFrame with Housing-Status rows x Scenario columns, values in [0, 1] range
    (fraction, not percent).
    """
    subsidy_df = build_indicator_summary(
        output_bundle, pattern=subsidy_pattern, select_value=select_value,
    )
    investment_df = build_indicator_summary(
        output_bundle, pattern=investment_pattern, select_value=select_value,
    )
    # Align indices (Housing - Status labels captured by the regex)
    common = subsidy_df.index.intersection(investment_df.index)
    ratio = (subsidy_df.loc[common] * subsidy_to_investment_scale) / investment_df.loc[common]
    ratio.index.name = "Metric"
    return ratio
