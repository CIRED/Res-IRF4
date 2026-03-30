"""Reusable plotting functions for post-processing notebooks."""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# Consistent colour mapping across all plots
SCENARIO_COLORS = {
    "OptimalSubsidies": "#1f77b4",  # blue
    "Package2024": "#ff7f0e",       # orange
    "Reference": "#ff7f0e",         # orange
}

_DEFAULT_FALLBACK_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

# Default subplot ordering: Single-family first, then Multi-family.
# Groups whose Housing component appears earlier in this list are shown first.
DEFAULT_HOUSING_ORDER = ["Single-family", "Multi-family"]
DEFAULT_STATUS_ORDER = ["Owner-occupied", "Privately rented", "Social-housing"]

# Occupancy statuses to exclude by default (Social-housing has a single income
# class, making the income-axis breakdown uninformative).
DEFAULT_EXCLUDE_STATUS = ["Social-housing"]
PREFERRED_SCENARIO_ORDER = ["Package2024", "OptimalSubsidies", "Reference"]


def _order_scenarios(scenarios):
    """Return scenarios in preferred display order, preserving unknown names."""
    scenarios = list(scenarios)
    preferred = [name for name in PREFERRED_SCENARIO_ORDER if name in scenarios]
    remaining = [name for name in scenarios if name not in preferred]
    return preferred + remaining


def _scenario_color_list(scenarios):
    """Return a list of colours aligned with *scenarios*, using the fixed palette."""
    scenarios = _order_scenarios(scenarios)
    return [
        SCENARIO_COLORS.get(s, _DEFAULT_FALLBACK_COLORS[i % len(_DEFAULT_FALLBACK_COLORS)])
        for i, s in enumerate(scenarios)
    ]


def _flatten_axes(axes):
    if isinstance(axes, np.ndarray):
        return list(axes.flatten())
    return [axes]


def _is_single_reference(scenarios):
    """Return True when the plot contains only the Reference scenario."""
    return list(scenarios) == ["Reference"]


def _has_single_scenario(scenarios):
    """Return True when the plot contains exactly one scenario."""
    return len(list(scenarios)) == 1


def _grid_ncols(panel_count):
    """Return the preferred number of subplot columns for grid-style figures."""
    if panel_count == 4:
        return 2
    return min(3, max(1, panel_count))


def _order_groups(groups, housing_order):
    """Sort *groups* (``"Housing - Status"`` strings) so that housing types
    listed earlier in *housing_order* come first."""
    def _sort_key(label):
        housing = label.split(" - ")[0]
        try:
            rank = housing_order.index(housing)
        except ValueError:
            rank = len(housing_order)
        return (rank, label)
    return sorted(groups, key=_sort_key)


def _order_segments(segments, housing_order=None, status_order=None):
    """Sort ``Housing - Status`` labels using preferred housing and status order."""
    housing_order = housing_order or DEFAULT_HOUSING_ORDER
    status_order = status_order or DEFAULT_STATUS_ORDER

    def _sort_key(label):
        parts = str(label).split(" - ", 1)
        housing = parts[0]
        status = parts[1] if len(parts) > 1 else ""
        try:
            housing_rank = housing_order.index(housing)
        except ValueError:
            housing_rank = len(housing_order)
        try:
            status_rank = status_order.index(status)
        except ValueError:
            status_rank = len(status_order)
        return (housing_rank, status_rank, str(label))

    return sorted(segments, key=_sort_key)


def _housing_status_subplot_grid(
    parsed,
    scenarios,
    value_col,
    ylabel,
    suptitle,
    label_size=14,
    same_axis_limits=True,
    save=None,
    hline=None,
    exclude_status=None,
    housing_order=None,
    n_cols=None,
    xlabel=None,
    legend_position="top",
):
    """Generic clustered-bar grid: one subplot per Housing x Status, x-axis = Income, one bar per scenario."""
    if exclude_status is None:
        exclude_status = DEFAULT_EXCLUDE_STATUS
    if housing_order is None:
        housing_order = DEFAULT_HOUSING_ORDER

    data = parsed[parsed["Income"] != "Total"].copy()
    scenarios = _order_scenarios(scenarios)
    data = data[data["Scenario"].isin(scenarios)]
    if exclude_status:
        data = data[~data["Status"].isin(exclude_status)]
    data["Label"] = data["Housing"] + " - " + data["Status"]
    groups = _order_groups(data["Label"].unique(), housing_order)
    incomes = sorted(data["Income"].unique())
    single_reference = _is_single_reference(scenarios)
    single_scenario = _has_single_scenario(scenarios)

    n_groups = len(groups)
    n_cols = min(n_cols or _grid_ncols(n_groups), max(1, n_groups))
    n_rows = math.ceil(n_groups / n_cols)

    colors = _scenario_color_list(scenarios)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.2 * n_rows), squeeze=False)
    flat_axes = [ax for row in axes for ax in row]
    figure_legend = None

    for idx, group in enumerate(groups):
        ax = flat_axes[idx]
        subset = data[data["Label"] == group]
        pivot = subset.pivot(index="Income", columns="Scenario", values=value_col)
        pivot = pivot.reindex(index=incomes, columns=scenarios)
        pivot.plot.bar(ax=ax, rot=0, color=colors, edgecolor="white")
        ax.set_title(group, fontsize=label_size)
        ax.set_xlabel("" if xlabel in (None, "") else xlabel, fontsize=label_size - 2)
        ax.set_ylabel(ylabel, fontsize=label_size - 2)
        ax.tick_params(axis="both", labelsize=label_size - 2)
        legend = ax.get_legend()
        if legend is not None:
            if not single_reference and not single_scenario and figure_legend is None:
                figure_legend = ax.get_legend_handles_labels()
            legend.remove()
        if hline is not None:
            ax.axhline(hline, color="grey", linestyle="--", linewidth=0.8)

    if same_axis_limits:
        all_vals = data[value_col].dropna().values
        if len(all_vals):
            ymin = float(np.nanmin(all_vals))
            ymax = float(np.nanmax(all_vals))
            margin = (ymax - ymin) * 0.05 if ymax > ymin else 0.05
            for i in range(len(groups)):
                flat_axes[i].set_ylim(ymin - margin, ymax + margin)

    for idx in range(len(groups), len(flat_axes)):
        flat_axes[idx].set_visible(False)

    fig.suptitle(suptitle, fontsize=label_size + 2)
    if figure_legend is not None:
        handles, labels = figure_legend
        if legend_position == "bottom":
            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.01),
                ncol=len(labels),
                frameon=False,
                fontsize=label_size - 2,
            )
            plt.tight_layout(rect=(0, 0.08, 1, 0.95))
        else:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.98),
                ncol=len(labels),
                frameon=False,
                fontsize=label_size - 2,
            )
            plt.tight_layout(rect=(0, 0, 1, 0.92))
    else:
        plt.tight_layout()

    if save is not None:
        plt.savefig(save, bbox_inches="tight")
    plt.show()


def plot_indicator_bar_grid(
    summary,
    same_axis_limits,
    label_size,
    xlabel="Scenario",
    value_formatter=None,
    title=None,
    save=None,
):
    """Plot one bar chart per indicator row.

    Parameters
    ----------
    summary:
        DataFrame with indicator rows x scenario columns.
    same_axis_limits:
        Enforce the same y-limits across all subplots.
    label_size:
        Font size for labels.
    xlabel:
        X-axis label shown on every subplot.
    value_formatter:
        Optional callable used to annotate each bar. Receives the numeric value
        and must return the display label.
    title:
        Optional figure-level title.
    """
    if summary.empty:
        raise ValueError("Summary dataframe is empty; no bar chart to plot.")

    scenarios = _order_scenarios(summary.columns)
    summary = summary.reindex(columns=scenarios)
    colors = _scenario_color_list(scenarios)
    single_reference = _is_single_reference(scenarios)

    panel_count = len(summary.index)
    n_cols = _grid_ncols(panel_count)
    n_rows = math.ceil(panel_count / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.2 * n_rows), squeeze=False)
    flat_axes = _flatten_axes(axes)
    x_positions = np.arange(len(scenarios))

    if same_axis_limits:
        values = summary.values.flatten()
        ymin = float(np.nanmin(values))
        ymax = float(np.nanmax(values))
        span = ymax - ymin
        base_margin = span * 0.05 if span > 0 else max(abs(ymax) * 0.05, 0.05)
        top_margin = max(base_margin, span * 0.12 if value_formatter is not None else base_margin)
        ylim = (ymin - base_margin, ymax + top_margin)
    else:
        ylim = None

    for idx, indicator in enumerate(summary.index):
        ax = flat_axes[idx]
        values = summary.loc[indicator, scenarios]
        bars = ax.bar(x_positions, values.values, color=colors, edgecolor="white")
        ax.set_title(str(indicator), fontsize=label_size)
        ax.set_xlabel(xlabel, fontsize=label_size - 2)
        ax.set_ylabel("")
        ax.tick_params(axis="both", labelsize=label_size)
        if single_reference:
            ax.set_xticklabels([])
            ax.set_xticks([])
        else:
            ax.set_xticks(x_positions)
            ax.set_xticklabels(scenarios, rotation=0)
        if ylim is not None:
            ax.set_ylim(*ylim)
        elif value_formatter is not None:
            local_min = float(np.nanmin(values.values))
            local_max = float(np.nanmax(values.values))
            span = local_max - local_min
            margin = span * 0.12 if span > 0 else max(abs(local_max) * 0.12, 0.05)
            lower = local_min - margin * 0.4 if local_min < 0 else 0
            ax.set_ylim(lower, local_max + margin)

        if value_formatter is not None:
            for bar, value in zip(bars, values.values):
                if pd.isna(value):
                    continue
                offset = 3 if value >= 0 else -3
                va = "bottom" if value >= 0 else "top"
                ax.annotate(
                    value_formatter(value),
                    (bar.get_x() + bar.get_width() / 2, value),
                    ha="center",
                    va=va,
                    fontsize=label_size - 3,
                    xytext=(0, offset),
                    textcoords="offset points",
                )

    for ax in flat_axes[panel_count:]:
        ax.set_visible(False)

    if title is not None:
        fig.suptitle(title, fontsize=label_size + 2)
        plt.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        plt.tight_layout()

    if save is not None:
        plt.savefig(save, bbox_inches="tight")
    plt.show()


def plot_subsidies_by_housing_status(
    parsed,
    scenarios,
    label_size=14,
    same_axis_limits=True,
    save=None,
    exclude_status=None,
    housing_order=None,
    suptitle=None,
    xlabel=None,
):
    """Clustered bar: subsidy levels by income class, one subplot per Housing x Status.

    Compares current vs. optimal subsidy levels side by side.
    """
    _housing_status_subplot_grid(
        parsed=parsed,
        scenarios=scenarios,
        value_col="Value",
        ylabel="Subsidies (M\u20ac)",
        suptitle=suptitle or "Public spending on subsidies by income class and housing segment",
        label_size=label_size,
        same_axis_limits=same_axis_limits,
        save=save,
        exclude_status=exclude_status,
        housing_order=housing_order,
        xlabel=xlabel,
        legend_position="bottom",
    )


def plot_subsidy_delta_by_housing_status(
    delta_df,
    scenario,
    optimal_scenario,
    label_size=14,
    same_axis_limits=True,
    save=None,
    exclude_status=None,
    housing_order=None,
    suptitle=None,
    xlabel=None,
    legend_position="top",
):
    """Bar chart of subsidy gap vs. optimal (in M euros), one subplot per Housing x Status.

    Positive bars = scenario spends more than optimal; negative = spends less.
    """
    scenarios = [scenario] if isinstance(scenario, str) else list(scenario)
    _housing_status_subplot_grid(
        parsed=delta_df,
        scenarios=scenarios,
        value_col="Delta",
        ylabel="Difference vs. optimal (M\u20ac)",
        suptitle=suptitle or "Misallocation of subsidies: {} vs. {}".format(", ".join(scenarios), optimal_scenario),
        label_size=label_size,
        same_axis_limits=same_axis_limits,
        save=save,
        hline=0.0,
        exclude_status=exclude_status,
        housing_order=housing_order,
        xlabel=xlabel,
        legend_position=legend_position,
    )


def plot_subsidy_delta_pct_by_housing_status(
    delta_df,
    scenario,
    optimal_scenario,
    label_size=14,
    same_axis_limits=True,
    save=None,
    exclude_status=None,
    housing_order=None,
    suptitle=None,
    xlabel=None,
    legend_position="top",
):
    """Bar chart of subsidy gap vs. optimal (in %), one subplot per Housing x Status.

    Positive bars = scenario spends more than optimal; negative = spends less.
    """
    scenarios = [scenario] if isinstance(scenario, str) else list(scenario)
    _housing_status_subplot_grid(
        parsed=delta_df,
        scenarios=scenarios,
        value_col="DeltaPct",
        ylabel="Difference vs. optimal (%)",
        suptitle=suptitle or "Misallocation of subsidies (%): {} vs. {}".format(", ".join(scenarios), optimal_scenario),
        label_size=label_size,
        same_axis_limits=same_axis_limits,
        save=save,
        hline=0.0,
        exclude_status=exclude_status,
        housing_order=housing_order,
        xlabel=xlabel,
        legend_position=legend_position,
    )


def plot_ad_valorem_ratio_by_housing_status(
    ratio_df,
    label_size=14,
    same_axis_limits=True,
    save=None,
    exclude_status=None,
    housing_order=None,
    suptitle=None,
):
    """Clustered bar chart of implicit ad valorem subsidy rate by Housing x Status.

    Parameters
    ----------
    ratio_df:
        DataFrame with Housing-Status rows x Scenario columns (values as fractions).
    exclude_status:
        Occupancy statuses to drop (default: Social-housing).
    housing_order:
        Housing types in display order (default: Single-family first).
    """
    if exclude_status is None:
        exclude_status = DEFAULT_EXCLUDE_STATUS
    if housing_order is None:
        housing_order = DEFAULT_HOUSING_ORDER

    if ratio_df.empty:
        raise ValueError("Ratio dataframe is empty; nothing to plot.")

    # Filter out excluded statuses
    filtered = ratio_df.copy()
    if exclude_status:
        filtered = filtered[
            ~filtered.index.to_series().apply(lambda x: any(s in x for s in exclude_status))
        ]

    scenarios = list(filtered.columns)
    scenarios = _order_scenarios(scenarios)
    filtered = filtered.reindex(columns=scenarios)
    colors = _scenario_color_list(scenarios)
    groups = _order_groups(list(filtered.index), housing_order)
    single_reference = _is_single_reference(scenarios)

    n_groups = len(groups)
    n_cols = _grid_ncols(n_groups)
    n_rows = math.ceil(n_groups / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.2 * n_rows), squeeze=False)
    flat_axes = [ax for row in axes for ax in row]

    for idx, group in enumerate(groups):
        ax = flat_axes[idx]
        values = filtered.loc[group] * 100  # convert to percent
        values.plot.bar(ax=ax, rot=0, color=colors, edgecolor="white")
        ax.set_title(group, fontsize=label_size)
        ax.set_xlabel("")
        ax.set_ylabel("Subsidy / Investment (%)", fontsize=label_size - 2)
        ax.tick_params(axis="both", labelsize=label_size - 2)
        if single_reference:
            ax.set_xticklabels([])

        for patch in ax.patches:
            height = patch.get_height()
            if pd.isna(height):
                continue
            ax.annotate(
                "{:,.0f}%".format(height),
                (patch.get_x() + patch.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=label_size - 3,
                xytext=(0, 3),
                textcoords="offset points",
            )

    for i in range(len(groups)):
        flat_axes[i].set_ylim(0, 100)

    for idx in range(len(groups), len(flat_axes)):
        flat_axes[idx].set_visible(False)

    fig.suptitle(
        suptitle or "Implicit ad valorem subsidy rate by housing segment",
        fontsize=label_size + 2,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))

    if save is not None:
        plt.savefig(save, bbox_inches="tight")
    plt.show()


def plot_subsidies_time_series_by_segment(
    timeseries: pd.DataFrame,
    label_size: int = 14,
    save=None,
    scenario_title_labels: dict[str, str] | None = None,
    ylabel: str = "M€",
    suptitle: str = "Total renovation subsidies over time by housing segment (Million €)",
    xlabel: str = "",
    housing_order=None,
    status_order=None,
    n_cols: int = 3,
):
    """Plot yearly subsidy series in a grid of housing-segment subplots."""
    required_cols = {"Scenario", "Segment", "Year", "Value"}
    missing = required_cols.difference(timeseries.columns)
    if missing:
        raise KeyError("Missing required time-series columns: {}".format(sorted(missing)))
    if timeseries.empty:
        raise ValueError("Time-series dataframe is empty; no plot to draw.")

    scenario_title_labels = scenario_title_labels or {}
    data = timeseries.copy()
    data["Year"] = data["Year"].astype(int)
    scenarios = _order_scenarios(data["Scenario"].dropna().unique().tolist())
    segments = _order_segments(
        data["Segment"].dropna().unique().tolist(),
        housing_order=housing_order,
        status_order=status_order,
    )

    colors = _scenario_color_list(scenarios)
    linestyles = ["-", "--", "-.", ":"]
    scenario_color = {scenario: colors[i] for i, scenario in enumerate(scenarios)}
    scenario_ls = {scenario: linestyles[i % len(linestyles)] for i, scenario in enumerate(scenarios)}

    n_rows = math.ceil(len(segments) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(15, 3.8 * n_rows),
        squeeze=False,
        sharey="row",
    )
    flat_axes = _flatten_axes(axes)

    for ax, segment in zip(flat_axes, segments):
        subset = data[data["Segment"] == segment]
        for scenario in scenarios:
            series = subset[subset["Scenario"] == scenario].sort_values("Year")
            series = series.dropna(subset=["Value"])
            if series.empty:
                continue
            ax.plot(
                series["Year"],
                series["Value"],
                label=scenario_title_labels.get(scenario, scenario),
                color=scenario_color[scenario],
                linestyle=scenario_ls[scenario],
                linewidth=1.8,
                marker="o",
                markersize=3,
            )
        ax.set_title(segment, fontsize=label_size - 1, pad=6)
        ax.set_xlabel(xlabel, fontsize=label_size - 2)
        ax.set_ylabel(ylabel, fontsize=label_size - 2)
        ax.tick_params(labelsize=label_size - 3)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in flat_axes[len(segments):]:
        ax.set_visible(False)

    legend_source = next((ax for ax in flat_axes if ax.lines), None)
    if legend_source is not None:
        handles, labels = legend_source.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=len(labels),
            frameon=False,
            fontsize=label_size - 1,
        )

    fig.suptitle(suptitle, fontsize=label_size + 1, y=1.01)
    fig.tight_layout()

    if save is not None:
        plt.savefig(save, bbox_inches="tight")
    plt.show()


def plot_subsidies_gap_boxplot(bundle, output_file, title=None):
    """Draw and save scenario-wise boxplots for Subsidies Gap."""
    data = pd.concat(bundle, names=["Scenario"]).reset_index(level=0)
    if "Subsidies Gap" not in data.columns:
        raise KeyError("`Subsidies Gap` column not found in distortion tables.")

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12.8, 9.6))
    box = sns.boxplot(data=data, x="Scenario", y="Subsidies Gap", ax=ax)
    single_reference = _is_single_reference(data["Scenario"].dropna().unique().tolist())

    medians = data.groupby("Scenario")["Subsidies Gap"].median()
    offset = data["Subsidies Gap"].median() * 0.05

    for tick, label in enumerate(box.get_xticklabels()):
        scenario = label.get_text()
        if scenario in medians.index:
            value = medians.loc[scenario]
            box.text(
                tick,
                value + offset,
                "{:.0f}".format(value),
                ha="center",
                size="x-small",
                color="w",
                weight="semibold",
            )

    ax.set_xlabel("")
    ax.set_ylabel("Subsidies Gap")
    ax.tick_params(axis="x", rotation=45)
    if single_reference:
        ax.set_xticklabels([])
    if title is not None:
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()


def plot_subsidy_vs_distortion(
    bundle,
    output_folder,
    technology_filter,
    make_scatter_fn,
    title_template=None,
    x_label="",
):
    """Save one scatter plot per scenario using shared axis limits."""
    xmin = min(df["Subsidies"].min() for df in bundle.values())
    xmax = max(df["Subsidies"].max() for df in bundle.values())
    ymin = min(df["Distortion"].min() for df in bundle.values())
    ymax = max(df["Distortion"].max() for df in bundle.values())

    for scenario, table in bundle.items():
        plot_df = table.reset_index(drop=True)

        if "Technology" in plot_df.columns:
            plot_df = plot_df[~plot_df["Technology"].isin(technology_filter)]

        if plot_df.empty:
            continue

        make_scatter_fn(
            plot_df,
            "Subsidies",
            "Distortion",
            x_label,
            (title_template or "Subsidies and distortion by technology: {scenario}").format(scenario=scenario),
            annotate=False,
            save=output_folder / "subsidies_distortion_{}.png".format(scenario),
            format_y=lambda y, _: "{:.0f}".format(y / 1e3),
            format_x=lambda x, _: "{:.0f}".format(x / 1e3),
            s=10,
            diagonal_line=True,
            col_colors=None,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
        )
