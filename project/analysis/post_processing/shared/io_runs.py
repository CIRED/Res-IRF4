"""IO helpers for scenario run outputs."""
from __future__ import annotations

from pathlib import Path
import os
import pandas as pd


def load_output_bundle(folder: str | Path, ignore: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Load all scenario output.csv files from child folders.

    Parameters
    ----------
    folder:
        Path to a run folder that contains scenario subfolders.
    ignore:
        Folder/file names to skip while scanning.
    """
    ignore = set(ignore or ["root_log.log", "img", ".DS_Store", "comparison.csv"])
    folder = Path(folder)

    data: dict[str, pd.DataFrame] = {}
    for child in [i for i in os.listdir(folder) if i not in ignore]:
        csv_path = folder / child / "output.csv"
        if csv_path.is_file():
            data[child] = pd.read_csv(csv_path, index_col=[0])
    return data


def load_child_csv_bundle(folder: str | Path, filename: str, index_col=None) -> dict[str, pd.DataFrame]:
    """Load one CSV file from each direct child folder.

    Parameters
    ----------
    folder:
        Parent folder whose children may contain *filename*.
    filename:
        CSV file name to look for inside each child folder.
    index_col:
        Column(s) to use as the row index (passed to ``pd.read_csv``).
    """
    folder = Path(folder)
    result = {}
    for child in sorted(folder.iterdir()):
        csv_path = child / filename
        if child.is_dir() and csv_path.is_file():
            result[child.name] = pd.read_csv(csv_path, index_col=index_col)
    return result


def concat_output_bundle(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Concatenate scenario-level outputs into a multi-indexed dataframe."""
    if not data:
        return pd.DataFrame()
    return pd.concat(data, axis=0).rename_axis(["Scenarios", "Variables"], axis=0)
