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
from pathlib import Path, PosixPath, WindowsPath
import sys
import json
import re
DECILES2QUINTILES = {'D1': 'C1', 'D2': 'C1',
                     'D3': 'C2', 'D4': 'C2',
                     'D5': 'C3', 'D6': 'C3',
                     'D7': 'C4', 'D8': 'C4',
                     'D9': 'C5', 'D10': 'C5'}


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


def size_dict(dict_vars, n=30, display=True):
    def sizeof_fmt(num, suffix='B'):
        for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
            if abs(num) < 1024.0:
                return "%3.1f %s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f %s%s" % (num, 'Yi', suffix)

    temp = dict()
    for name, size in sorted(((name, get_size(value)) for name, value in list(
                              dict_vars.items())), key=lambda x: -x[1])[:n]:
        if display:
            print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
        temp.update({name: sizeof_fmt(size)})
    return temp


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def get_pandas(path, func=lambda x: pd.read_csv(x)):
    path = Path(path)
    if isinstance(path, WindowsPath):
        with resources.path(str(path.parent).replace('\\', '.'), path.name) as df:
            return func(df)
    else:
        with resources.path(str(path.parent).replace('/', '.'), path.name) as df:
            return func(df)


def get_json(path):
    path = Path(path)
    if isinstance(path, WindowsPath):
        with resources.path(str(path.parent).replace('\\', '.'), path.name) as f:
            with open(f) as file:
                return json.load(file)
    else:
        with resources.path(str(path.parent).replace('/', '.'), path.name) as f:
            with open(f) as file:
                return json.load(file)


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


def select(df, dict_levels):
    idx = np.array([True] * df.shape[0])
    for level, value in dict_levels.items():
        if not isinstance(value, list):
            value = [value]
        idx *= df.index.get_level_values(level).isin(value)
    if isinstance(df, pd.DataFrame):
        return df.loc[idx, :]
    elif isinstance(df, pd.Series):
        return df.loc[idx]


def deciles2quintiles_pandas(data, func='mean'):
    if isinstance(data, (pd.DataFrame, pd.Series)):
        level_income = []
        for key in ['Income owner', 'Income tenant', 'Income']:
            if key in data.index.names:
                level_income += [key]

        for level in level_income:
            names = None
            if isinstance(data.index, pd.MultiIndex):
                names = data.index.names

            data = data.rename(index=DECILES2QUINTILES, level=level)

            if func == 'mean':
                data = data.groupby(data.index).mean()
            elif func == 'sum':
                data = data.groupby(data.index).sum()

            if names:
                data.index = pd.MultiIndex.from_tuples(data.index)
                data.index.names = names

    return data


def deciles2quintiles_list(item):
    new_item = []
    for i in item:
        if i in DECILES2QUINTILES.keys():
            i = DECILES2QUINTILES[i]
        new_item.append(i)

    return list(set(new_item))


def deciles2quintiles_dict(inputs):

    for key, item in inputs.items():
        if isinstance(item, (pd.Series, pd.DataFrame)):
            inputs[key] = deciles2quintiles_pandas(item)
        elif isinstance(item, list):
            inputs[key] = deciles2quintiles_list(item)
        elif isinstance(item, dict):
            for k, i in item.items():
                if isinstance(i, (pd.Series, pd.DataFrame)):
                    inputs[key][k] = deciles2quintiles_pandas(i)
                elif isinstance(i, list):
                    inputs[key][k] = deciles2quintiles_list(i)
                elif isinstance(i, dict):
                    for kk, ii in i.items():
                        if isinstance(ii, (pd.Series, pd.DataFrame)):
                            inputs[key][k][kk] = deciles2quintiles_pandas(ii)
                        elif isinstance(ii, list):
                            inputs[key][k][kk] = deciles2quintiles_list(ii)
                        elif isinstance(ii, dict):
                            for kkk, iii in ii.items():
                                if isinstance(iii, (pd.Series, pd.DataFrame)):
                                    inputs[key][k][kk][kkk] = deciles2quintiles_pandas(iii)

    return inputs


def calculate_annuities(capex, lifetime=50, discount_rate=0.032):
    return capex * discount_rate / (1 - (1 + discount_rate) ** (-lifetime))


def make_policies_tables(policies, path, plot=True):
    sub_replace = {'subsidy_target': 'Subsidy, per unit',
                   'subsidy_ad_valorem': 'Subsidy, ad valorem',
                   'bonus': 'Subsidy, bonus',
                   'obligation': 'Retrofitting obligation',
                   'premature_heater': 'Premature replacement',
                   'reduced_vta': 'Reduced VTA',
                   'restriction_heater': 'Restriction heater',
                   'restriction_energy': 'Restriction energy',
                   'subsidies_cap': 'Subsidy, cap'
                   }

    heater_replace = {'Electricity-Heat pump air': 'HP-air',
                      'Electricity-Heat pump water': 'HP-water',
                      'Natural gas-Performance boiler': 'GasBoiler',
                      'Natural gas-Standard boiler': 'GasBoiler',
                      'Natural gas-Collective boiler': 'CollectiveGasBoiler',
                      'Wood fuel-Performance boiler': 'WoodBoiler',
                      }

    tables_policies = list()
    for p in policies:
        temp = {'Name': '{} \n {}'.format(p.name.capitalize().replace('_', ' '), p.gest.capitalize()),
                'Date': '{} - {}'.format(p.start, p.end),
                'Policy': '{}'.format(sub_replace[p.policy])
                }

        value = p.value
        growth = False
        if isinstance(value, dict):
            value = value[list(value.keys())[0]]
            growth = True

        if isinstance(value, pd.DataFrame):
            t = value.mean()
        else:
            t = value
        if isinstance(t, pd.Series):
            if p.policy == 'obligation':
                t = t[t.ne(t.shift())] # only for retrofitting obligation
            else:
                t = t[t > 0]

            if isinstance(t.index, pd.MultiIndex):
                t.index = ['-'.join(col) for col in t.index.values]
            t = t.rename_axis(None)
            t = t.rename(None)
            if p.policy in ['subsidy_ad_valorem', 'subsidies_cap']:
                t = t.map('{:,.0%}'.format)
            elif p.policy == 'subsidy_target':
                t = t.map('{:,.0f}'.format)

            if p.gest == 'heater':
                t = t.rename(index=heater_replace)

            t = t.to_string(name=None).replace('\n', ';')
            t = re.sub(' +', ':', t)

        elif isinstance(t, list):
            t = ', '.join(t)
        else:
            t = value

        details = 'Value: {}'.format(t)
        if growth:
            details = details + ',\nGrowth: true'
        if p.target is not None:
            t = p.target
            if isinstance(t, list):
                t = ', '.join(t)
            details = details + ',\nTarget: {}'.format(t)
        if p.cap is not None:
            cap = p.cap
            if isinstance(cap, dict):
                cap = cap[list(cap.keys())[0]]
            if isinstance(cap, pd.Series):
                cap = cap[cap > 0]
                if isinstance(cap.index, pd.MultiIndex):
                    cap.index = ['-'.join(col) for col in cap.index.values]
                cap = cap.rename_axis(None)
                cap = cap.rename(None)
                cap = cap.map('{:,.0f}'.format)
                cap = cap.to_string(name=None).replace('\n', ';')
                cap = re.sub(' +', ':', cap)

            details = details + ',\nCap: {}'.format(cap)

        temp.update({'Details': details})
        tables_policies.append(temp)
    tables_policies = pd.DataFrame(tables_policies).set_index('Name').sort_index()
    tables_policies.to_csv(path)
    if plot:
        plot_table(tables_policies, path)


def plot_table(tables_policies, path):
    ax = plt.subplot(111, frame_on=False)  # no visible frame
    ax.axis('tight')  # turns off the axis lines and labels
    ax.axis('off') # hide the y axis

    cell_text = []
    number_max = 50
    for row in range(len(tables_policies)):
        temp = tables_policies.iloc[row].copy()
        t = temp.loc['Details'].split('\n')
        if [i for i in t if len(i) > number_max]:
            new = []
            for i in temp.loc['Details'].split('\n'):
                if len(i) > number_max:
                    new.append(i[:number_max] + '\n' + i[number_max:])
                else:
                    new.append(i)
            temp.loc['Details'] = '\n'.join(new)

        cell_text.append(temp)

    table = plt.table(cellText=cell_text, colLabels=tables_policies.columns,
                      rowLabels=tables_policies.index,
                      loc='center', colWidths=[0.15, 0.25, 0.65],
                      cellLoc='left')
    plt.axis('off')
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 4)
    plt.savefig(path.replace('.csv', '.png'), dpi=200, bbox_inches='tight')
    plt.close()





def format_ax(ax, y_label=None, title=None, format_y=lambda y, _: y, ymin=0, ymax=None, xinteger=True):
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
        _, y_max = ax.get_ylim()
        ax.set_ylim(ymax=y_max * 1.1)

    if ymax is not None:
        ax.set_ylim(ymax=ymax)

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


def make_plot(df, y_label, colors=None, format_y=lambda y, _: y, save=None, scatter=None, legend=True, integer=True,
              ymin=0, ymax=None):
    """Make plot.

    Parameters
    ----------
    df: pd.DataFrame
    y_label: str
    colors: dict
    format_y: function
    save: str, optional
    scatter: pd.Series, default None
    ymin: float, optional
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

    ax = format_ax(ax, title=y_label, format_y=format_y, ymin=ymin, xinteger=integer, ymax=ymax)
    if legend:
        format_legend(ax)
    # plt.ticklabel_format(style='plain', axis='x')

    save_fig(fig, save=save)


def make_plots(dict_df, y_label, colors=None, format_y=lambda y, _: y, save=None, scatter=None, legend=True,
               integer=False, loc='upper', left=1.04):
    """Make plot.

    Parameters
    ----------
    dict_df: dict
    y_label: str
    colors: dict
    format_y: function
    save: str, optional
    scatter: pd.Series, default None
    """
    sns.set_palette(sns.color_palette('husl', len(dict_df.keys())))

    fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))


    for key, df in dict_df.items():

        df = df.rename(key)

        if integer:
            df.index = df.index.astype(int)

        if colors is None:
            df.plot(ax=ax, style=STYLES)
        else:
            df.plot(ax=ax, color=colors, style=STYLES)

        if scatter is not None:
            scatter.plot(ax=ax, style='.', ms=15, c='red')

    ax = format_ax(ax, title=y_label, format_y=format_y, ymin=0, xinteger=True)
    if legend:
        format_legend(ax, loc=loc, left=left)
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
    try:
        sns.set_palette(sns.color_palette('husl', dict_df[list_keys[0]].shape[1]))
    except:
        print('break')
    try:
        y_max = max([i.fillna(0).to_numpy().max() for i in dict_df.values()]) * 1.1
    except ValueError:
        print('break')

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

            plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
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
                   ymin=None, loc='upper', scatter=None, left=1.04, xinteger=True):

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
    ax = format_ax(ax, title=y_label, xinteger=xinteger, format_y=format_y, ymin=ymin)
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


def plot_attribute(stock, attribute, dict_order=None, suptitle=None, percent=False, dict_color=None,
                   width=0.3, save=None, figsize=(12.8, 9.6)):
    """Make bar plot for 1 stock dataframe for one attribute in order to graphically compare.

    Parameters
    ----------
    stock: pd.Series
    attribute: str
        Level name of stock.
    dict_order: dict, optional
    suptitle: str, optional
    percent: bool
    dict_color: dict, optional
    width: float, default 0.3
    """

    fig, ax = plt.subplots(figsize=figsize)

    stock_total = stock.sum()

    if suptitle:
        fig.suptitle(suptitle, fontsize=20, fontweight='bold')

    stock_attribute = stock.groupby(attribute).sum()

    if dict_order:
        if attribute in dict_order.keys():
            stock_attribute = stock_attribute.loc[dict_order[attribute]]

    if percent:
        stock_attribute = stock_attribute / stock_total
        format_y = lambda y, _: '{:,.0f}%'.format(y * 100)
    else:
        format_y = lambda y, _: '{:,.0f}M'.format(y / 1000000)

    if dict_color is not None:
        stock_attribute.plot.bar(ax=ax, color=[dict_color[key] for key in stock_attribute.index], width=width)

    else:
        stock_attribute.plot.bar(ax=ax, width=width)

    ax.xaxis.set_tick_params(which=u'both', length=0)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y))
    ax.yaxis.set_tick_params(which=u'both', length=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

    if save is not None:
        fig.savefig(save, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def subplots_attributes(stock, dict_order={}, suptitle=None, percent=False, dict_color=None,
                        n_columns=3, sharey=False, save=None):
    """Multiple bar plot of stock by attributes.

    Parameters
    ----------
    stock: pd.Series
    dict_order: dict
    suptitle: str
    percent: bool
    dict_color: dict
    n_columns: int
    sharey: bool
    """
    labels = list(stock.index.names)
    stock_total = stock.sum()

    n_axes = int(len(stock.index.names))
    n_rows = ceil(n_axes / n_columns)
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(12.8, 9.6), sharey=sharey)

    if suptitle:
        fig.suptitle(suptitle, fontsize=20, fontweight='bold')

    for k in range(n_rows * n_columns):

        try:
            label = labels[k]
        except IndexError:
            ax.remove()
            break

        stock_label = stock.groupby(label).sum()
        if label in dict_order.keys():
            stock_label = stock_label.loc[dict_order[label]]

        if percent:
            stock_label = stock_label / stock_total
            format_y = lambda y, _: '{:,.0f}%'.format(y * 100)
        else:
            format_y = lambda y, _: '{:,.0f}M'.format(y / 1000000)

        row = floor(k / n_columns)
        column = k % n_columns
        if n_rows == 1:
            ax = axes[column]
        else:
            ax = axes[row, column]

        if dict_color is not None:
            stock_label.plot.bar(ax=ax, color=[dict_color[key] for key in stock_label.index])
        else:
            stock_label.plot.bar(ax=ax)

        ax.xaxis.set_tick_params(which=u'both', length=0)
        ax.xaxis.label.set_size(12)

        ax.tick_params(axis='x', which='major', labelsize=8)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y))
        ax.tick_params(axis='y', which='major', labelsize=12)

        ax.yaxis.set_tick_params(which=u'both', length=0)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

    if save is not None:
        fig.savefig(save, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def subplots_pie(stock, dict_order={}, pie={}, suptitle=None, percent=False, dict_color=None,
                 n_columns=3, save=None):
    """Multiple bar plot of stock by attributes.

    Parameters
    ----------
    stock: pd.Series
    dict_order: dict
    pie: dict
    suptitle: str
    percent: bool
    dict_color: dict
    n_columns: int
    sharey: bool
    """
    labels = list(stock.index.names)
    stock_total = stock.sum()

    n_axes = int(len(stock.index.names))
    n_rows = ceil(n_axes / n_columns)
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(12.8, 9.6))

    if suptitle:
        fig.suptitle(suptitle, fontsize=20, fontweight='bold')

    for k in range(n_rows * n_columns):

        try:
            label = labels[k]
        except IndexError:
            ax.remove()
            break

        stock_label = stock.groupby(label).sum()
        if label in dict_order.keys():
            stock_label = stock_label.loc[dict_order[label]]

        if percent:
            stock_label = stock_label / stock_total
            format_y = lambda y, _: '{:,.0f}%'.format(y * 100)
        else:
            format_y = lambda y, _: '{:,.0f}M'.format(y / 1000000)

        row = floor(k / n_columns)
        column = k % n_columns
        if n_rows == 1:
            ax = axes[column]
        else:
            ax = axes[row, column]
        if label in pie:
            if dict_color is not None:
                lab = [string.replace(" ", "\n").replace("-", "\n") for string in stock_label.index]
                stock_label.plot.pie(ax=ax, explode=None, labels=lab, colors=[dict_color[key] for key in
                                                                              stock_label.index],
                                     autopct='%1.1f%%', shadow=False, textprops={'fontsize': 10},
                                     ylabel='', xlabel=stock_label.index.name)
                ax.set_title(stock_label.index.name, fontsize=12)
            else:
                stock_label.plot.pie(ax=ax, explode=None, labels=stock_label.index, autopct='%1.1f%%', shadow=False,
                                     textprops = {'fontsize': 10},  ylabel='', xlabel=stock_label.index.name)
        else:
            if dict_color is not None:
                stock_label.plot.bar(ax=ax, color=[dict_color[key] for key in stock_label.index])
            else:
                stock_label.plot.bar(ax=ax)

        ax.xaxis.set_tick_params(which=u'both', length=0)
        ax.xaxis.label.set_size(12)

        ax.tick_params(axis='x', which='major', labelsize=8)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y))
        ax.tick_params(axis='y', which='major', labelsize=12)

        ax.yaxis.set_tick_params(which=u'both', length=0)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

    if save is not None:
        fig.savefig(save, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_attribute2attribute(stock, attribute1, attribute2, suptitle=None, dict_order={}, dict_color={}, percent=False,
                             save=None):
    fig, ax = plt.subplots(figsize=(12.8, 9.6))
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=20, fontweight='bold')

    df = stock.groupby([attribute1, attribute2]).sum().unstack(attribute2)
    if percent:
        df = (df.T * df.sum(axis=1) ** -1).T
        format_y = lambda y, _: '{:,.0f}%'.format(y * 100)
    else:
        format_y = lambda y, _: '{:,.0f}M'.format(y / 1000000)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y))

    df = df.loc[dict_order[attribute1], dict_order[attribute2]]

    df.plot(ax=ax, kind='bar', stacked=True, color=dict_color)

    ax.xaxis.set_tick_params(which=u'both', length=0)
    ax.yaxis.set_tick_params(which=u'both', length=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.legend(loc='best', frameon=False)

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

    if save is not None:
        fig.savefig(save, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def cumulated_plot(x, y, plot=True, format_y=lambda y, _: y):
    """Y by cumulated x.

    Use for marginal abatement cost curve.

    Parameters
    ----------
    x: Series
    y: Series

    Returns
    -------

    """
    df = pd.concat((x, y), axis=1)
    df.sort_values(y.name, inplace=True)
    df['{} cumulated'.format(x.name)] = df[x.name].cumsum()
    df = df.set_index('{} cumulated'.format(x.name))[y.name]
    if plot:
        make_plot(df, y_label=y.name, legend=False, format_y=format_y)
    else:
        return df


def cumulated_plots(dict_df, y_label, legend=True, format_y=lambda y, _: y, save=None, ylim=None, ymin=0):
    fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))
    for k, df in dict_df.items():
        df.rename(k).plot(ax=ax)

    ax = format_ax(ax, title=y_label, format_y=format_y, ymin=ymin, xinteger=False)
    if legend:
        format_legend(ax, loc='left', left=1.1)

    if ylim:
        ax.set_ylim(top=ylim)
    save_fig(fig, save=save)


def compare_bar_plot(df, y_label, legend=True, format_y=lambda y, _: y, save=None):

    fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))

    df.plot(ax=ax, kind='bar')

    ax = format_ax(ax, title=y_label, format_y=format_y)
    if legend:
        format_legend(ax)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

    # plt.ticklabel_format(style='plain', axis='x')

    save_fig(fig, save=save)


def make_hist(df, x, hue, y_label, legend=True, format_y=lambda y, _: y, save=None, kde=False, palette=None,
              bins=20):

    fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))

    if palette is None:
        ax = sns.histplot(data=df, x=x, kde=kde, weights='Stock', hue=hue, bins=bins,
                     palette=palette, ax=ax, legend=legend)
    else:
        ax = sns.histplot(data=df, x=x, kde=kde, weights='Stock', hue=hue, bins=bins,
                     palette=palette, ax=ax, legend=legend)

    ax = format_ax(ax, title=y_label, format_y=format_y)
    ax.yaxis.label.set_visible(False)

    # plt.legend(bbox_to_anchor=(1.2, 0.5))

    # plt.ticklabel_format(style='plain', axis='x')

    save_fig(fig, save=save)
