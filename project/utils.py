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

STYLES = ['-', '--', ':', 's-', 'o-', '^-', '*-', 's-', 'o-', '^-', '*-'] * 10


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


def make_plot(df, y_label, colors=None, format_y=lambda y, _: y, save=None, scatter=None):
    """Make plot.

    Parameters
    ----------
    df: pd.DataFrame
    y_label: str
    colors: dict
    format_y: {'percent', 'million', 'billion'}
    save: str, optional
    scatter: pd.Series, default None
    """
    df.index = df.index.astype(int)
    fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))

    if colors is None:
        df.plot(ax=ax, style=STYLES)
    else:
        df.plot(ax=ax, color=colors, style=STYLES)

    if scatter is not None:
        scatter.plot(ax=ax, style='.', ms=8)

    ax.set_ylabel(y_label)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_tick_params(which=u'both', length=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

    ax.yaxis.set_tick_params(which=u'both', length=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y))
    ax.set_ylim(ymin=0)

    try:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                  frameon=False, shadow=True, ncol=5)
    except AttributeError:
        pass

    if save is not None:
        fig.savefig(save, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


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
    y_max = max([i.fillna(0).to_numpy().max() for i in dict_df.values()])

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
            dict_df[key].sort_index().plot(ax=ax, linewidth=1, style=STYLES, ms=3)
            if scatter is not None:
                scatter[key].plot(ax=ax, style='.', ms=8, color=sns.color_palette('bright', scatter[key].shape[1]))

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)

            plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
            ax.xaxis.set_tick_params(which=u'both', length=0)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if n_bins is not None:
                plt.locator_params(axis='x', nbins=n_bins)

            ax.yaxis.set_tick_params(which=u'both', length=0)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y))
            ax.set_ylim(ymin=0, ymax=y_max)

            if isinstance(key, tuple):
                ax.set_title('{}-{}'.format(key[0], key[1]), pad=-1.6)
            else:
                ax.set_title(key, pad=-1.6)
            if k == 0:
                handles, labels = ax.get_legend_handles_labels()
                labels = [l.replace('_', ' ') for l in labels]
            ax.get_legend().remove()
        except IndexError:
            ax.axis('off')

    fig.legend(handles, labels, loc='lower center', frameon=False, ncol=3,
               bbox_to_anchor=(0.5, -0.1))

    if save is not None:
        fig.savefig(save, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def make_area_plot(df, y_label, colors=None, save=None):

    df.index = df.index.astype(int)
    fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))

    if colors is None:
        df.plot.area(ax=ax, stacked=True)
    else:
        df.plot.area(ax=ax, stacked=True, color=colors)

    df.sum(axis=1).rename('Total').plot(ax=ax, color='black')
    ax.set_ylabel(y_label)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_tick_params(which=u'both', length=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_tick_params(which=u'both', length=0)

    try:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                  frameon=False, shadow=True, ncol=2)
    except AttributeError:
        pass

    if save is not None:
        fig.savefig(save, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def make_stackedbar_plot(df, y_label, colors=None, format_y=lambda y, _: y, save=None):
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

    ax.set_ylabel(y_label)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_tick_params(which=u'both', length=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

    ax.yaxis.set_tick_params(which=u'both', length=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y))
    ax.set_ylim(ymin=0)

    try:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                  frameon=False, shadow=True, ncol=df.shape[1])
    except AttributeError:
        pass

    if save is not None:
        fig.savefig(save, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def waterfall_chart(df, title=None, save=None, figsize=(12.8, 9.6)):
    """Make waterfall chart. Used for Social Economic Assessment.

    Parameters
    ----------
    df: pd.Series
    title: str, optional
    figsize

    Returns
    -------

    """

    color = {'Investment': 'firebrick', 'Energy saving': 'darkorange', 'Emission saving': 'forestgreen',
             'Health benefit': 'royalblue', 'Total': 'black'}

    data = df.copy()

    data.rename(index={'Energy saving': 'Energy',
                       'Emission saving': 'Emission',
                       'Health benefit': 'Health'
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
    data.plot(kind='bar', stacked=True, bottom=blank, legend=None,
              title=title, ax=ax, color=color.values(), edgecolor=None)
    # my_plot.plot(step.index, step.values, 'k')

    plt.axhline(y=0, color='black', linestyle='--')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_tick_params(which=u'both', length=0, labelsize=16)
    ax.yaxis.set_tick_params(which=u'both', length=0, labelsize=16)

    # Get the y-axis position for the labels
    y_height = data.cumsum().shift(1).fillna(0)

    # Get an offset so labels don't sit right on top of the bar
    max = data.max()
    neg_offset, pos_offset = max / 20, max / 50
    plot_offset = int(max / 15)

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
        ax.annotate("{:,.0f}".format(val), (loop, y), ha="center")
        loop += 1

    ax.set_xticklabels(data.index, rotation=0)

    if save is not None:
        fig.savefig(save, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def assessment_scenarios(df, save=None, figsize=(12.8, 9.6)):
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
    color = {'Investment': 'firebrick', 'Energy': 'darkorange', 'Emission': 'forestgreen',
             'Health': 'royalblue'}

    data = df.copy()

    data.rename(columns={'Energy saving': 'Energy',
                         'Emission saving': 'Emission',
                         'Health benefit': 'Health'
                         }, inplace=True)

    total = data.sum(axis=1).reset_index()
    total.columns = ['Scenarios', 'NPV']

    pd.DataFrame(total).plot(kind='scatter', x='Scenarios', y='NPV', legend=False, zorder=10, ax=ax, color='black',
                             s=50, xlabel=None)
    data.plot(kind='bar', stacked=True, ax=ax, color=color)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_tick_params(which=u'both', length=0, labelsize=16)
    ax.yaxis.set_tick_params(which=u'both', length=0, labelsize=16)

    ax.xaxis.label.set_visible(False)
    ax.yaxis.label.set_visible(False)

    for _, y in total.iterrows():
        ax.annotate("{:,.1f} B€".format(y['NPV']), (y['Scenarios'], y['NPV'] + 3), ha="center")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
              frameon=False, shadow=True, ncol=2)
    ax.set_xticklabels(data.index, rotation=0)

    if save is not None:
        fig.savefig(save, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
