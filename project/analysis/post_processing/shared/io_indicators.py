"""Helpers to load policy indicator tables from run folders."""

from pathlib import Path
import os
import pandas as pd


def load_indicator(folder: str | Path, relative_file: str = "policies/indicator.csv") -> pd.DataFrame:
    """Load a policy indicator file from a run folder."""
    folder = Path(folder)
    return pd.read_csv(folder / relative_file, index_col=[0])


def load_indicators_from_children(folder: str | Path) -> dict[str, pd.DataFrame]:
    """Load indicator tables from all direct child directories."""
    folder = Path(folder)
    result: dict[str, pd.DataFrame] = {}
    for child in [i for i in os.listdir(folder) if i != ".DS_Store" and (folder / i).is_dir()]:
        indicator = folder / child / "policies" / "indicator.csv"
        if indicator.is_file():
            result[child] = pd.read_csv(indicator, index_col=[0])
    return result
