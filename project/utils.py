# Copyright 2020-2021 Ecole Nationale des Ponts et Chaussées
#
# This file is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Original author Lucas Vivier <vivier@centre-cired.fr>

import numpy as np
import pandas as pd
from math import floor, ceil
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import defaultdict
from functools import wraps
from time import time
from importlib import resources
from pathlib import Path

COLOR = 'dimgrey'
SMALL_SIZE = 10
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE, titlecolor=COLOR, titleweight='bold', labelsize=BIGGER_SIZE, labelcolor=COLOR,
       labelweight='bold')  # fontsize of the axes title of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE, color=COLOR)  # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE, color=COLOR)  # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('lines', lw=3.5)
plt.rc('axes', lw=3.5, edgecolor=COLOR)

STYLES = ['-', '--', ':', 's-', 'o-', '^-', '*-', 's-', 'o-', '^-', '*-'] * 10


def get_pandas(path, func=lambda x: pd.read_csv(x)):
    path = Path(path)
    with resources.path(str(path.parent).replace('/', '.'), path.name) as df:
        return func(df)


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f'Function {f.__name__} took {te - ts:2.4f} seconds')
        return result
    return wrap


def reverse_dict(data):
    flipped = defaultdict(dict)
    for key, val in data.items():
        for subkey, subval in val.items():
            flipped[subkey][key] = subval
    return dict(flipped)


def reindex_mi(df, mi_index, levels=None, axis=0):
    """Return re-indexed DataFrame based on miindex using only few labels.

    Parameters
    -----------
    df: pd.DataFrame, pd.Series
        data to reindex
    mi_index: pd.MultiIndex, pd.Index
        master to index to reindex df
    levels: list, default df.index.names
        list of levels to use to reindex df
    axis: {0, 1}, default 0
        axis to reindex df

    Returns
    --------
    pd.DataFrame, pd.Series

    Example
    -------
        reindex_mi(surface_ds, segments, ['Occupancy status', 'Housing type']))
        reindex_mi(cost_invest_ds, segments, ['Heating energy final', 'Heating energy']))
    """

    if isinstance(df, (float, int)):
        return pd.Series(df, index=mi_index)

    if levels is None:
        if axis == 0:
            levels = df.index.names
        else:
            levels = df.columns.names

    if len(levels) > 1:
        tuple_index = (mi_index.get_level_values(level).tolist() for level in levels)
        new_miindex = pd.MultiIndex.from_tuples(list(zip(*tuple_index)))
        if axis == 0:
            df = df.reorder_levels(levels)
        else:
            df = df.reorder_levels(levels, axis=1)
    else:
        new_miindex = mi_index.get_level_values(levels[0])
    df_reindex = df.reindex(new_miindex, axis=axis)
    if axis == 0:
        df_reindex.index = mi_index
    elif axis == 1:
        df_reindex.columns = mi_index
    else:
        raise AttributeError('Axis can only be 0 or 1')

    return df_reindex


def format_ax(ax, y_label=None, title=None, format_y=lambda y, _: y, ymin=0, xinteger=True):
    """

    Parameters
    ----------
    y_label: str
    format_y: function
    ymin: float or None
    xinteger: bool
    title: str, optional

    Returns
    -------

    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    # ax.spines['bottom'].set_linewidth(2)

    ax.spines['left'].set_visible(True)
    # ax.spines['left'].set_linewidth(2)
    ax.xaxis.set_tick_params(which=u'both', length=0)
    ax.yaxis.set_tick_params(which=u'both', length=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y))

    if y_label is not None:
        ax.set_ylabel(y_label)

    if title:
        t = title.split(' (')[0]
        unit = title.split(' (')[1].split(')')[0]
        ax.set_title('{}\n{}'.format(t, unit), loc='left')

    if ymin is not None:
        ax.set_ylim(ymin=0)
        _, ymax = ax.get_ylim()
        ax.set_ylim(ymax=ymax * 1.1)

    if xinteger:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    return ax


def format_legend(ax, ncol=3, offset=1, labels=None, loc='upper', left=1.04):
    try:
        leg = None
        if loc == 'upper':
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1,
                             box.width, box.height * 0.9])

            # Put a legend below current axis
            if labels is not None:
                leg = ax.legend(labels, loc='upper center', bbox_to_anchor=(0.5, -0.07 * offset),
                                frameon=False, shadow=True, ncol=ncol)
            else:
                leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07 * offset),
                                frameon=False, shadow=False, ncol=ncol)
        elif loc == 'left':
            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])

            # Put a legend to the right of the current axis
            if labels is not None:
                leg = ax.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5),
                                frameon=False, shadow=False)
            else:
                handles, labels = ax.get_legend_handles_labels()
                leg = ax.legend(handles[::-1], labels[::-1], loc='upper center', bbox_to_anchor=(left, 0.7),
                                frameon=False, shadow=True)
        texts = leg.get_texts()
        for text in texts:
            text.set_color(COLOR)

    except AttributeError:
        pass


def save_fig(fig, save=None, bbox_inches='tight'):
    if save is not None:
        fig.savefig(save, bbox_inches=bbox_inches)
        plt.close(fig)
    else:
        plt.show()


def make_plot(df, y_label, colors=None, format_y=lambda y, _: y, save=None, scatter=None, legend=True, integer=True):
    """Make plot.

    Parameters
    ----------
    df: pd.DataFrame
    y_label: str
    colors: dict
    format_y: function
    save: str, optional
    scatter: pd.Series, default None
    """
    if integer:
        df.index = df.index.astype(int)
    fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))

    if colors is None:
        df.plot(ax=ax, style=STYLES)
    else:
        df.plot(ax=ax, color=colors, style=STYLES)

    if scatter is not None:
        scatter.plot(ax=ax, style='.', ms=15, c='red')

    ax = format_ax(ax, title=y_label, format_y=format_y, ymin=0, xinteger=True)
    if legend:
        format_legend(ax)
    save_fig(fig, save=save)


def make_grouped_subplots(dict_df, n_columns=3, format_y=lambda y, _: y, n_bins=2, save=None, scatter=None, order=None):
    """ Plot a line for each index in a subplot.

    Parameters
    ----------
    dict_df: dict
        df_dict values are pd.DataFrame (index=years, columns=scenario)
    format_y: function, optional
    n_columns: int, default 3
    n_bins: int, default None
    save: str, default None
    scatter: dict, default None
    """
    list_keys = list(dict_df.keys())
    if order is not None:
        list_keys = order

    sns.set_palette(sns.color_palette('husl', dict_df[list_keys[0]].shape[1]))
    y_max = max([i.fillna(0).to_numpy().max() for i in dict_df.values()]) * 1.1

    n_axes = int(len(list_keys))
    n_rows = ceil(n_axes / n_columns)
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(12.8, 9.6), sharex='all', sharey='all')
    handles, labels = None, None
    for k in range(n_rows * n_columns):

        row = floor(k / n_columns)
        column = k % n_columns
        if n_rows == 1:
            ax = axes[column]
        else:
            ax = axes[row, column]
        try:
            key = list_keys[k]
            dict_df[key].sort_index().plot(ax=ax, style=STYLES, ms=3)
            if scatter is not None:
                scatter[key].plot(ax=ax, style='.', ms=8, color=sns.color_palette('bright', scatter[key].shape[1]))

            ax = format_ax(ax, format_y=format_y, ymin=0, xinteger=True)
            ax.spines['left'].set_visible(False)
            ax.set_ylim(ymax=y_max)

            plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
            if n_bins is not None:
                plt.locator_params(axis='x', nbins=n_bins)

            title = key
            if isinstance(key, tuple):
                title = '{}-{}'.format(key[0], key[1])
            ax.set_title(title, fontweight='bold', color='dimgrey', pad=-1.6)

            if k == 0:
                handles, labels = ax.get_legend_handles_labels()
                labels = [l.replace('_', ' ') for l in labels]
            ax.get_legend().remove()
        except IndexError:
            ax.axis('off')

    fig.legend(handles, labels, loc='lower center', frameon=False, ncol=3,
               bbox_to_anchor=(0.5, -0.1))

    save_fig(fig, save=save)


def make_area_plot(df, y_label, colors=None, format_y=lambda y, _: y, save=None, ncol=3, total=True, offset=1,
                   ymin=None, loc='upper', scatter=None, left=1.04):

    df.index = df.index.astype(int)
    fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))
    if colors is None:
        df.plot.area(ax=ax, stacked=True)
    else:
        df.plot.area(ax=ax, stacked=True, color=colors)

    if total:
        df.sum(axis=1).rename('Total').plot(ax=ax, color='black')

    if scatter is not None:
        scatter.plot(ax=ax, style='.', ms=15, c='red')
    ax = format_ax(ax, title=y_label, xinteger=True, format_y=format_y, ymin=ymin)
    format_legend(ax, ncol=ncol, offset=offset, loc=loc, left=left)

    save_fig(fig, save=save)


def make_stackedbar_plot(df, y_label, colors=None, format_y=lambda y, _: y, save=None, ncol=3, offset=1):
    """Make stackedbar plot.

    Parameters
    ----------
    df: pd.DataFrame
    y_label: str
    colors: dict
    format_y: function
    save: str, optional
    """
    df.index = df.index.astype(int)
    fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))

    if colors is None:
        df.plot(ax=ax, kind='bar', stacked=True)
    else:
        df.plot(ax=ax, kind='bar', stacked=True, color=colors)

    ax = format_ax(ax, y_label=y_label, format_y=format_y, ymin=0, xinteger=True)
    format_legend(ax, ncol=ncol, offset=offset)
    save_fig(fig, save=save, bbox_inches=None)


def waterfall_chart(df, title=None, save=None, colors=None, figsize=(12.8, 9.6)):
    """Make waterfall chart. Used for Social Economic Assessment.

    Parameters
    ----------
    df: pd.Series
    title: str, optional
    figsize

    Returns
    -------

    """

    # color = {'Investment': 'firebrick', 'Embodied emission additional': 'darkgreen', 'Cofp': 'grey',
    #          'Energy saving': 'darkorange', 'Emission saving': 'forestgreen',
    #          'Well-being benefit': 'royalblue', 'Health savings': 'blue',
    #          'Mortality reduction benefit': 'lightblue',  'Total': 'black'}


    data = df.copy()
    if colors is not None:
        color = [colors[key] for key in list(data.index) + ['Social NPV']]

    data.rename(index={'Energy saving': 'Energy',
                       'Emission saving': 'Emission',
                       'Embodied emission additional': 'Embodied emission',
                       'Well-being benefit': 'Well-being',
                       'Mortality reduction benefit': 'Mortality',
                       'Cofp': 'COFP'
                       }, inplace=True)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    blank = data.cumsum().shift(1).fillna(0)

    # Get the net total number for the final element in the waterfall
    total = data.sum()
    blank.loc["Social NPV"] = total
    data.loc["Social NPV"] = total
    # The steps graphically show the levels as well as used for label placement
    step = blank.reset_index(drop=True).repeat(3).shift(-1)
    step[1::3] = np.nan

    # When plotting the last element, we want to show the full bar,
    # Set the blank to 0
    blank.loc["Social NPV"] = 0

    # Plot and label
    if colors is None:
        data.plot(kind='bar', stacked=True, bottom=blank, legend=None,
                  title=title, ax=ax, edgecolor=None)
    else:
        data.plot(kind='bar', stacked=True, bottom=blank, legend=None,
              title=title, ax=ax, color=color, edgecolor=None)
    # my_plot.plot(step.index, step.values, 'k')

    plt.axhline(y=0, color='black', linestyle='--', linewidth=0.3)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_tick_params(which=u'both', length=0, labelsize=16)
    ax.yaxis.set_tick_params(which=u'both', length=0, labelsize=16)

    # Get the y-axis position for the labels
    y_height = data.cumsum().shift(1).fillna(0)

    # Get an offset so labels don't sit right on top of the bar
    max = data.max()
    min = data.min()
    neg_offset, pos_offset = max / 10, max / 50
    plot_offset = int(max / 15)
    ax.set_ylim(top=max + max/3, bottom=min + min/3 )

    # Start label loop
    loop = 0
    for index, val in data.iteritems():
        # For the last item in the list, we don't want to double count
        if val == total:
            y = y_height[loop]
        else:
            y = y_height[loop] + val
        # Determine if we want a neg or pos offset
        if val > 0:
            y += pos_offset
        else:
            y -= neg_offset
        ax.annotate("{:,.1f}".format(val), (loop, y), ha="center")
        loop += 1
    labels = [string.replace(" ", "\n") for string in data.index]
    ax.set_xticklabels(labels, rotation=15)
    save_fig(fig, save=save)


def assessment_scenarios(df, save=None, colors=None, figsize=(12.8, 9.6)):
    """Compare social NPV between scenarios and one reference.

    Stacked bar chart.

    Parameters
    ----------
    df
    save
    figsize

    Returns
    -------

    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    data = df.copy()

    data.rename(index={'Energy saving': 'Energy',
                       'Emission saving': 'Emission',
                       'Embodied emission additional': 'Embodied emission',
                       'Well-being benefit': 'Well-being',
                       'Mortality reduction benefit': 'Mortality',
                       'Cofp': 'COFP'
                       }, inplace=True)

    total = data.sum(axis=1).reset_index()
    total.columns = ['Scenarios', 'NPV']

    pd.DataFrame(total).plot(kind='scatter', x='Scenarios', y='NPV', legend=False, zorder=10, ax=ax, color='black',
                             s=50, xlabel=None)

    if colors is None:
        data.plot(kind='bar', stacked=True, ax=ax)
    else:
        data.plot(kind='bar', stacked=True, ax=ax, color=colors)

    plt.axhline(y=0, color='black', linestyle='--', linewidth=0.3)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_tick_params(which=u'both', length=0, labelsize=16)
    ax.yaxis.set_tick_params(which=u'both', length=0, labelsize=16)

    ax.xaxis.label.set_visible(False)
    ax.yaxis.label.set_visible(False)
    y_range = abs(ax.get_ylim()[1] - ax.get_ylim()[0])

    for _, y in total.iterrows():
        ax.annotate("{:,.1f} B€".format(y['NPV']), (y['Scenarios'], y['NPV'] + y_range /20 ), ha="center")

    ax.set_xticklabels(data.index, rotation=0)

    format_legend(ax)
    save_fig(fig, save=save)


def make_uncertainty_plot(df, title, detailed=False, format_y=lambda y, _: y, ymin=0, save=None, scatter=None,
                          columns=None, ncol=3, offset=1, loc='upper', left=1.04):
    """Plot multi scenarios and uncertainty area between lower value and higher value of scenarios.

    Parameters
    ----------
    df: pd.DataFrame
        Columns represent one scenario
    title: str
    detailed: bool, default False
    format_y: func
    ymin: float or int
    """

    subset = df.loc[:, columns]
    others = df.loc[:, [c for c in df.columns if c not in columns]]

    df_min = subset.min(axis=1)
    df_max = subset.max(axis=1)
    df_ref = df.loc[:, 'Reference']

    fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))
    fig.subplots_adjust(top=0.85)
    if not others.empty:
        others.plot(ax=ax)

    if detailed:
        subset.plot(ax=ax)
    df_ref.plot(ax=ax, c='black')
    plt.fill_between(df_min.index, df_min.values, df_max.values, alpha=0.4)

    if scatter is not None:
        scatter.plot(ax=ax, style='.', ms=15, c='red')

    format_ax(ax, title=title, xinteger=True, format_y=format_y, ymin=ymin)
    format_legend(ax, ncol=ncol, offset=offset, loc=loc, left=left)

    save_fig(fig, save=save)

