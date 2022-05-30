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

import matplotlib.pyplot as plt
from math import ceil, floor
import pandas as pd
import seaborn as sns

SMALL_SIZE = 10
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


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