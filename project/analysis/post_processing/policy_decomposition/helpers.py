"""Helpers for policy decomposition post-processing notebooks."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from project.analysis.post_processing.shared.io_runs import load_output_bundle
from project.utils import format_ax


def resolve_run_folder(run_id: str, description_filename: str, start_dir: str | Path | None = None) -> Path:
    """Resolve a policy-decomposition run folder and validate raw inputs."""
    cwd = Path(start_dir or Path.cwd()).resolve()
    repo_root = next((path for path in [cwd, *cwd.parents] if (path / "project").is_dir()), None)
    if repo_root is None:
        raise FileNotFoundError(
            f"Could not locate the repository root from the current working directory: {cwd}"
        )

    candidate_roots = [
        repo_root / "project" / "analysis" / "post_processing" / "policy_decomposition" / "runs" / "complete_interactions",
        cwd / "runs" / "complete_interactions",
    ]

    checked: list[Path] = []
    incomplete: list[tuple[Path, list[str]]] = []
    for root in dict.fromkeys(path.resolve() for path in candidate_roots):
        candidate = root / run_id
        checked.append(candidate)
        if not candidate.is_dir():
            continue

        missing = []
        if not (candidate / description_filename).is_file():
            missing.append(description_filename)
        if not any(candidate.glob("*/output.csv")):
            missing.append("*/output.csv")

        if not missing:
            return candidate

        incomplete.append((candidate, missing))

    if incomplete:
        details = "\n\n".join(
            f"{path}\nMissing required inputs: {', '.join(missing)}"
            for path, missing in incomplete
        )
        raise FileNotFoundError(
            f"Run folder found for '{run_id}', but it is missing required raw inputs:\n{details}"
        )

    searched = "\n".join(str(path) for path in checked)
    raise FileNotFoundError(f"Run folder not found for '{run_id}'. Checked:\n{searched}")


def load_scenario_outputs(folder: str | Path) -> dict[str, pd.DataFrame]:
    """Load scenario output tables and coerce year columns to integers."""
    bundle = load_output_bundle(folder)
    for output_df in bundle.values():
        output_df.columns = [int(column) for column in output_df.columns]
    return bundle


def make_mapping_scatter(
    data: pd.DataFrame,
    color_dict: dict[str, str],
    file: str | os.PathLike[str],
    x: str = "NPV annual (Billion euro/year)",
    y: str = "Cumulated energy saving (TWh)",
    hue: str = "Group",
    format_y: Callable[[float, int], str] | None = None,
    format_x: Callable[[float, int], str] | None = None,
    annotate: list[str] | None = None,
    x_label: str | None = None,
    remove_legend: list[str] | None = None,
    xmin: float | None = None,
    xmax: float | None = None,
    legend_title: str = "",
    size: str | None = None,
    hline: float | None = None,
    ymax: float | None = None,
) -> None:
    """Build a scatter plot used for policy mapping visualizations."""
    format_y = format_y or (lambda value, _: f"{value:.0f}")
    format_x = format_x or (lambda value, _: f"{value:.0f}")
    annotate = [] if annotate is None else [item for item in annotate if item in data[hue].values]

    if size is None:
        point_sizes = 100
        df = data.loc[:, [x, y, hue]].copy()
    else:
        df = data.loc[:, [x, y, hue, size]].dropna().copy()
        point_sizes = df[size]

    df = df.sort_values(hue)

    csv_path = os.path.splitext(file)[0] + ".csv"
    df.to_csv(csv_path)

    fig, ax = plt.subplots(figsize=(12.8, 9.6))
    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        ax=ax,
        s=point_sizes,
        hue=hue,
        palette=color_dict,
        style=hue,
    )

    format_ax(
        ax,
        title=y,
        format_y=format_y,
        y_label="",
        format_x=format_x,
        ymin=None,
        xmin=xmin,
        xmax=xmax,
        ymax=ymax,
    )
    if x_label is not None:
        ax.set_xlabel(x_label)

    y_max = df[y].max()
    if hline is not None:
        y_max = max(y_max, hline)

    x_min = df[x].min()
    x_max = df[x].max()
    y_min = df[y].min()

    margin_x = (x_max - x_min) * 0.1 if x_max > x_min else 1.0
    margin_y = (y_max - y_min) * 0.1 if y_max > y_min else 1.0
    ax.set_xlim([x_min - margin_x, x_max + margin_x])
    ax.set_ylim([y_min - margin_y, y_max + margin_y])

    ax.axhline(0, color="black", lw=1)
    if hline is not None:
        ax.axhline(hline, color="red", lw=1, linestyle="--")

    for label in annotate:
        row = df[df[hue] == label]
        if row.empty:
            continue
        ax.annotate(
            label,
            (float(row[x].iloc[0]), float(row[y].iloc[0])),
            fontsize=12,
            color=color_dict.get(label, "black"),
            fontweight="bold",
        )

    handles, labels = ax.get_legend_handles_labels()
    hidden_labels = set(annotate)
    if remove_legend is not None:
        hidden_labels.update(remove_legend)

    filtered_legend = [
        (handle, label)
        for handle, label in zip(handles, labels)
        if label not in hidden_labels
    ]

    if filtered_legend:
        handles, labels = zip(*filtered_legend)
        ax.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(1, 1),
            frameon=False,
            title=legend_title,
        )
    elif ax.get_legend() is not None:
        ax.get_legend().remove()

    fig.savefig(file, bbox_inches="tight")
    plt.close(fig)


def _ordered_interaction_matrix(temp: pd.DataFrame, orders: list[str] | None = None) -> pd.DataFrame:
    matrix = temp.pivot(index="policy", columns="group", values="value")
    if orders is None:
        return matrix

    row_order = [item for item in orders if item in matrix.index] + [
        item for item in matrix.index if item not in orders
    ]
    col_order = [item for item in orders if item in matrix.columns] + [
        item for item in matrix.columns if item not in orders
    ]
    return matrix.loc[row_order, col_order]


def make_interaction_heatmap(
    temp: pd.DataFrame,
    file: str | os.PathLike[str],
    fmt: str = ".1f",
    xlabel: str = "Social welfare (billion euro per year)",
    orders: list[str] | None = None,
) -> None:
    """Plot the interaction heatmap and export the underlying matrix."""
    fig, ax = plt.subplots(figsize=(12.8, 9.6))

    matrix = _ordered_interaction_matrix(temp, orders=orders)

    cmap = sns.diverging_palette(220, 20, as_cmap=True, center="light")
    sns.heatmap(
        matrix,
        annot=False,
        fmt=fmt,
        ax=ax,
        cbar_kws={"label": xlabel},
        center=0,
        cmap=cmap,
    )

    for i, row in enumerate(matrix.index):
        for j, col in enumerate(matrix.columns):
            value = matrix.at[row, col]
            if not np.isnan(value):
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    format(value, fmt),
                    ha="center",
                    va="center",
                    fontweight="bold" if row == col else "normal",
                )

    ax.set_xlabel("In interaction with")
    ax.set_ylabel("Policies")
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", labelrotation=90)
    ax.xaxis.set_label_position("top")

    csv_path = os.path.splitext(file)[0] + ".csv"
    matrix.to_csv(csv_path)

    fig.savefig(file, bbox_inches="tight")
    plt.close(fig)


def make_interaction_heatmap_percent(
    temp: pd.DataFrame,
    file: str | os.PathLike[str],
    xlabel: str = "Social welfare (billion euro per year)",
    orders: list[str] | None = None,
) -> None:
    """Plot interaction changes relative to the standalone baseline."""
    fig, ax = plt.subplots(figsize=(12.8, 9.6))

    matrix = _ordered_interaction_matrix(temp, orders=orders)
    cmap = sns.diverging_palette(20, 150, as_cmap=True, center="light")

    for row in matrix.index:
        baseline = matrix.at[row, row] if row in matrix.columns else np.nan
        for col in matrix.columns:
            if row == col:
                continue
            if pd.isna(baseline) or baseline == 0:
                matrix.at[row, col] = np.nan
            else:
                matrix.at[row, col] = ((matrix.at[row, col] - baseline) / baseline) * 100

    sns.heatmap(
        matrix,
        annot=False,
        fmt=".1f",
        ax=ax,
        cbar=False,
        center=0,
        cmap=cmap,
        mask=np.eye(len(matrix), dtype=bool),
    )

    for i, row in enumerate(matrix.index):
        for j, col in enumerate(matrix.columns):
            value = matrix.at[row, col]
            if not np.isnan(value):
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{value:.1f}" if row == col else f"{value:.0f}%",
                    ha="center",
                    va="center",
                    fontweight="bold" if row == col else "normal",
                    color="black",
                )

    ax.set_xlabel("In interaction with")
    ax.set_ylabel("Policies")
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", labelrotation=90)
    ax.xaxis.set_label_position("top")

    csv_path = os.path.splitext(file)[0] + ".csv"
    matrix.to_csv(csv_path)

    fig.savefig(file, bbox_inches="tight")
    plt.close(fig)


def make_swarmplot(
    df: pd.DataFrame,
    file: str | os.PathLike[str],
    xlabel: str = "Social welfare (billion euro per year)",
    orders: list[str] | None = None,
    xmin: float | None = None,
    xmax: float | None = None,
) -> None:
    """Plot interaction results as a swarmplot and export the pivot table."""
    fig, ax = plt.subplots(figsize=(12.8, 9.6))

    if orders is not None:
        present = df["policy"].unique().tolist()
        ordered = [item for item in orders if item in present] + [
            item for item in present if item not in orders
        ]
        rank = {name: idx for idx, name in enumerate(ordered)}
        df = df.sort_values(by="policy", key=lambda series: series.map(rank))

    sns.swarmplot(ax=ax, data=df, x="value", y="policy", hue="hue", size=5)

    if xmin is not None and xmax is not None:
        ax.set_xlim(xmin, xmax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    csv_path = os.path.splitext(file)[0] + ".csv"
    df.pivot(index="policy", columns="group", values="value").to_csv(csv_path)

    fig.savefig(file, bbox_inches="tight")
    plt.close(fig)
