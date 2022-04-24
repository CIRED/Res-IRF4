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

import pandas as pd
import numpy as np


def stock_need(population, pop_housing_ini, pop_housing_min, start, factor):
    """

    Parameters
    ----------
    population
    pop_housing_ini
    pop_housing_min
    start
    factor

    Returns
    -------
    Stock need
    """
    population_housing = dict()
    population_housing[start] = pop_housing_ini
    max_year = max(population.index)

    stock_needed = dict()
    stock_needed[start] = population[start] / population_housing[start]

    for year in range(start + 1, max_year + 1):
        if year > max_year:
            break
        population_housing[year] = population_housing_dynamic(population_housing[year - 1],
                                                              pop_housing_min,
                                                              pop_housing_ini,
                                                              factor)
        stock_needed[year] = population[year] / population_housing[year]

    return pd.Series(stock_needed), pd.Series(population_housing)


def population_housing_dynamic(pop_housing_prev, pop_housing_min, pop_housing_ini, factor):
    """Returns number of people by building.

    Number of people by housing decrease over the time.

    Parameters
    ----------
    pop_housing_prev: int
    pop_housing_min: int
    pop_housing_ini: int
    factor: int

    Returns
    -------
    int
    """
    eps_pop_housing = (pop_housing_prev - pop_housing_min) / (
            pop_housing_ini - pop_housing_min)
    eps_pop_housing = max(0, min(1, eps_pop_housing))
    factor_pop_housing = factor * eps_pop_housing
    return max(pop_housing_min, pop_housing_prev * (1 + factor_pop_housing))


def share_multi_family(stock_need, factor):
    """Calculate share of multi-family buildings in the total stock

    In Res-IRF 2.0, the share of single- and multi-family dwellings was held constant in both existing and new
    housing stocks, but at different levels; it therefore evolved in the total stock by a simple composition
    effect. These dynamics are now more precisely parameterized in Res-IRF 3.0 thanks to recent empirical
    work linking the increase in the share of multi-family housing in the total stock to the rate of growth of
    the total stock housing growth (Fisch et al., 2015).
    This relationship in particular reflects urbanization effects.

    Parameters
    ----------
    stock_need: pd.Series
    factor: float

    Returns
    -------
    dict
        Dictionary with year as keys and share of multi-family in the total stock as value.
        {2012: 0.393, 2013: 0.394, 2014: 0.395}
    """

    def func(stock, stock_ini, f):
        """Share of multi-family dwellings as a function of the growth rate of the dwelling stock.

        Parameters
        ----------
        stock: float
        stock_ini: float
        f: float

        Returns
        -------
        float
        """
        trend_housing = (stock - stock_ini) / stock_ini * 100
        share = 0.1032 * np.log(10.22 * trend_housing / 10 + 79.43) * f
        return share

    share_multi_family_tot = {}
    stock_needed_ini = stock_need.iloc[0]
    for year in stock_need.index:
        share_multi_family_tot[year] = func(stock_need.loc[year], stock_needed_ini, factor)

    return pd.Series(share_multi_family_tot)


def evolution_surface_built(surface_ini, surface_max, elasticity_surface, income):
    """Evolution of new buildings area based on total available income. Function represents growth.

    Parameters
    ----------
    surface_ini: pd.Series
    surface_max: pd.Series
    elasticity_surface: pd.Series
    income: pd.Series

    Returns
    -------
    pd.DataFrame
    """

    surface = {income.index[0]: surface_ini}
    for year in income.index[1:]:

        surface_max = surface_max.reorder_levels(surface_ini.index.names)

        eps_area_new = (surface_max - surface[year - 1]) / (surface_max - surface_ini)
        eps_area_new = eps_area_new.apply(lambda x: max(0, min(1, x)))
        elasticity = eps_area_new.multiply(elasticity_surface)

        factor = elasticity * max(0.0, income[year] / income.iloc[0] - 1.0)

        surface[year] = pd.concat([surface_max, surface[year - 1] * (1 + factor)], axis=1).min(axis=1)

    return pd.DataFrame(surface)


def share_type_built(stock_need, share_multi_family, flow_built):
    """Calculate multi-family buildings in the construction.

    Parameters
    ----------
    stock_need: pd.Series
    share_multi_family: pd.Series
    flow_built: pd.Series

    Returns
    -------
    pd.Series
    """
    share_multi_family_built = (stock_need * share_multi_family - stock_need.shift(1) * share_multi_family.shift(
        1)) / flow_built

    return pd.concat((share_multi_family_built, 1 - share_multi_family_built), axis=1,
                     keys=['Multi-family', 'Single-family'], names=['Housing type']).dropna().T
