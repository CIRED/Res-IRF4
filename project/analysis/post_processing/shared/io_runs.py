"""IO helpers for scenario run outputs."""

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


def concat_output_bundle(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Concatenate scenario-level outputs into a multi-indexed dataframe."""
    if not data:
        return pd.DataFrame()
    return pd.concat(data, axis=0).rename_axis(["Scenarios", "Variables"], axis=0)
