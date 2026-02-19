"""Common transformations used across post-processing notebooks."""

import pandas as pd


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
