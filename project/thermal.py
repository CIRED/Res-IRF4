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
from utils import reindex_mi

"""LOGISTIC_COEFFICIENT = pd.read_csv('project/input/logistic_regression_coefficient_epc.csv', index_col=[0])
LOGISTIC_COEFFICIENT.columns = ['Intercept', 'Proxy_conso_square']
LOGISTIC_COEFFICIENT.index.names = ['Performance']"""
CONVERSION = 2.3
DHH = 55706
CERTIFICATE_3USES_BOUNDARIES = {
    'A': [0, 50],
    'B': [50, 90],
    'C': [90, 150],
    'D': [150, 230],
    'E': [230, 330],
    'F': [330, 450],
    'G': [450, 1000],
}


def model_heating_consumption(df, a=0.921323, b=0.634717):
    """
    X = (Deper_mur + Deper_baie + Deper_plancher + Deper_plafond) / Efficiency * DDH
    Conso = X ** 1.746973 * exp(1.536192)
    """
    return (df ** a) * np.exp(b)


def model_3uses_consumption(df, a=0.846102, b=1.029880):
    """
    X = Primary space hating energy consumption (kWh EP / m2)
    Conso = X ** 0.846102 * exp(1.029880)
    """
    return (df ** a) * np.exp(b)


def heating_consumption(u_wall, u_floor, u_roof, u_windows, dh, efficiency, ratio_surface):
    """Calculate space heating consumption in kWh/m2.year based on insulation performance and heating system efficiency.

    Function simulates the 3CL-method, and use parameters to estimate unobserved variables.

    Parameters
    ----------
    u_wall: float or pd.Series
    u_floor: float or pd.Series
    u_roof: float or pd.Series
    u_windows: float or pd.Series
    dh: float or pd.Series
    efficiency: float or pd.Series
    ratio_surface: pd.Series

    Returns
    -------
    float or pd.Series or pd.DataFrame
        Standard space heating consumption.
    """
    data = pd.concat([u_wall, u_floor, u_roof, u_windows], axis=1, keys=['Wall', 'Floor', 'Roof', 'Windows'])
    partial_losses = (reindex_mi(ratio_surface, data.index) * data).sum(axis=1)

    if isinstance(partial_losses, (pd.Series, pd.DataFrame)):
        if partial_losses.index.equals(efficiency.index):
            indicator_losses = partial_losses / efficiency * dh / 1000
            consumption = model_heating_consumption(indicator_losses).rename('Consumption')

        else:
            indicator_losses = (partial_losses * dh / 1000).to_frame().dot(
                (1 / efficiency).to_frame().T)
            consumption = model_heating_consumption(indicator_losses)

    else:
        indicator_losses = partial_losses / efficiency * dh / 1000
        consumption = model_heating_consumption(indicator_losses).rename('Consumption')

    return consumption


def final2primary(heat_consumption, energy, conversion=CONVERSION):
    if isinstance(heat_consumption, pd.Series):
        primary_heat_consumption = heat_consumption.copy()
        primary_heat_consumption[energy == 'Electricity'] = primary_heat_consumption * conversion
        return primary_heat_consumption

    elif isinstance(heat_consumption, float):
        if energy == 'Electricity':
            return heat_consumption * conversion
        else:
            return heat_consumption

    if isinstance(heat_consumption, pd.DataFrame):
        # index
        if energy.index.equals(heat_consumption.index):
            primary_heat_consumption = heat_consumption.copy()
            primary_heat_consumption.loc[energy == 'Electricity', :] = primary_heat_consumption * conversion
            return primary_heat_consumption
        # columns
        elif energy.index.equals(heat_consumption.columns):
            primary_heat_consumption = heat_consumption.copy()
            primary_heat_consumption.loc[:, energy == 'Electricity'] = primary_heat_consumption * conversion
            return primary_heat_consumption
        else:
            raise 'Energy DataFrame do not match indexes and columns'


def primary_heating_consumption(u_wall, u_floor, u_roof, u_windows, dh, efficiency, energy, ratio_surface,
                                conversion=CONVERSION):
    """Convert final to primary heating consumption.

    Parameters
    ----------
    u_wall
    u_floor
    u_roof
    u_windows
    dh
    efficiency
    energy
    ratio_surface
    conversion

    Returns
    -------
    """
    heat_consumption = heating_consumption(u_wall, u_floor, u_roof, u_windows, dh, efficiency, ratio_surface)
    return final2primary(heat_consumption, energy, conversion=conversion)


def certificate(df):
    """Returns energy performance certificate based on space heating energy consumption.

    Parameters
    ----------
    df: float or pd.Series or pd.DataFrame
        Space heating energy consumption (kWh PE / m2.year)
    Returns
    -------
    float or pd.Series or pd.DataFrame
        Energy performance certificate.
    """
    primary_consumption = model_3uses_consumption(df)

    if isinstance(primary_consumption, pd.Series):
        certificate = pd.Series(dtype=str, index=primary_consumption.index)
        for key, item in CERTIFICATE_3USES_BOUNDARIES.items():
            cond = (primary_consumption > item[0]) & (primary_consumption <= item[1])
            certificate[cond] = key
        return certificate

    elif isinstance(primary_consumption, pd.DataFrame):
        certificate = pd.DataFrame(dtype=str, index=primary_consumption.index, columns=primary_consumption.columns)
        for key, item in CERTIFICATE_3USES_BOUNDARIES.items():
            cond = (primary_consumption > item[0]) & (primary_consumption <= item[1])
            certificate[cond] = key
        return certificate

    elif isinstance(primary_consumption, float):
        for key, item in CERTIFICATE_3USES_BOUNDARIES.items():
            if (primary_consumption > item[0]) & (primary_consumption <= item[1]):
                return key


def certificate_buildings(u_wall, u_floor, u_roof, u_windows, dh, efficiency, energy, ratio_surface):
    """Returns energy performance certificate.

    Parameters
    ----------
    u_wall
    u_floor
    u_roof
    u_windows
    dh
    efficiency
    energy
    param

    Returns
    -------
    pd.Series
        Primary heating consumption for all buildings in the stock.
    pd.Series
        Certificates for all buildings in the stock.

    """
    primary_heat_consumption = primary_heating_consumption(u_wall, u_floor, u_roof, u_windows, dh, efficiency,
                                                           energy, ratio_surface, conversion=CONVERSION)
    return primary_heat_consumption, certificate(primary_heat_consumption)


def _certificate(primary_consumption, single_family=None):
    if single_family is None:
        single_family = pd.Series(primary_consumption.index.get_level_values('Housing type') == 'Single-family',
                                  index=primary_consumption.index, name='Single-family')
    else:
        if single_family:
            single_family = pd.Series(True, index=primary_consumption.index, name='Single-family')
        else:
            single_family = pd.Series(False, index=primary_consumption.index, name='Single-family')

    df = pd.concat((primary_consumption.rename('Consumption'), single_family), axis=1)

    def func(ds):
        LOGISTIC_COEFFICIENT = None
        y = LOGISTIC_COEFFICIENT['Intercept'] + LOGISTIC_COEFFICIENT['Proxy_conso_square'] * ds['Consumption']
        proba = np.exp(y) / (np.exp(y).sum())
        return proba.idxmax()

    performance = df.apply(func, axis=1)
    return performance


def _heating_consumption(u_wall, u_floor, u_roof, u_windows, dh, efficiency, param):
    """Calculate space heating consumption in kWh/m2.year based on insulation performance and heating system efficiency.

    Function simulates the 3CL-method, and use parameters to estimate unobserved variables.

    Parameters
    ----------
    u_wall: float or pd.Series
    u_floor: float or pd.Series
    u_roof: float or pd.Series
    u_windows: float or pd.Series
    dh: float or pd.Series
    efficiency: float or pd.Series
    param: dict

    Returns
    -------
    float or pd.Series or pd.DataFrame
        Standard space heating consumption.
    """
    data = pd.concat([u_wall, u_floor, u_roof, u_windows], axis=1, keys=['Wall', 'Floor', 'Roof', 'Windows'])
    partial_losses = (reindex_mi(param['ratio_surface'], data.index) * data).sum(axis=1)

    if isinstance(partial_losses, (pd.Series, pd.DataFrame)):
        if partial_losses.index.equals(efficiency.index):
            indicator_losses = partial_losses / efficiency * dh / 1000
            consumption = (reindex_mi(param['coefficient'], indicator_losses.index) * indicator_losses).rename('Consumption')

        else:
            indicator_losses = (partial_losses * dh / 1000).to_frame().dot(
                (1 / efficiency).to_frame().T)

            consumption = (reindex_mi(param['coefficient'], indicator_losses.index) * indicator_losses.T).T

    else:
        indicator_losses = partial_losses / efficiency * dh / 1000
        consumption = (reindex_mi(param['coefficient'], indicator_losses.index) * indicator_losses).rename('Consumption')

    return consumption


def _certificate(conso, certificate_bounds):
    """Returns energy performance certificate based on space heating energy consumption.

    Parameters
    ----------
    conso: float or pd.Series or pd.DataFrame
        Space heating energy consumption.
    certificate_bounds: dict
        Energy consumption bounds that define certificate.

    Returns
    -------
    float or pd.Series or pd.DataFrame
        Energy performance certificate.
    """

    if isinstance(conso, pd.Series):
        certificate = pd.Series(dtype=str, index=conso.index)
        for key, item in certificate_bounds.items():
            cond = (conso > item[0]) & (conso <= item[1])
            certificate[cond] = key
        return certificate

    elif isinstance(conso, pd.DataFrame):
        certificate = pd.DataFrame(dtype=str, index=conso.index, columns=conso.columns)
        for key, item in certificate_bounds.items():
            cond = (conso > item[0]) & (conso <= item[1])
            certificate[cond] = key
        return certificate

    elif isinstance(conso, float):
        for key, item in certificate_bounds.items():
            if (conso > item[0]) & (conso <= item[1]):
                return key