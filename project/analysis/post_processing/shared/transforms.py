"""Common transformations used across post-processing notebooks."""

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
