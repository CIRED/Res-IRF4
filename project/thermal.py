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


def certificate(conso, certificate_bounds):
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


def heating_consumption(u_wall, u_floor, u_roof, u_windows, dh, efficiency, param):
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
    float or pd.Series
        Partial losses.
    float or pd.Series
        Envelope losses.
    float or pd.Series
        Heating need.
    float or pd.Series
        Standard space heating consumption.
    """

    partial_losses = param['ratio_surface_wall'] * u_wall + param['ratio_surface_roof'] * u_roof + param[
        'ratio_surface_floor'] * u_floor + param['ratio_surface_windows'] * u_windows
    envelope_losses = param['partial_losses_surface'] * partial_losses + param['const partial_losses_surface']
    heating_need = param['envelope_losses_surface'] * envelope_losses + param['const envelope_losses_surface']

    if isinstance(heating_need, (pd.Series, pd.DataFrame)):
        if heating_need.index.equals(efficiency.index):
            heat_consumption = param['heating_consumption_calcul'] * heating_need * dh / 1000 * (1 / efficiency) + param[
                'const heating_consumption_calcul']
        else:
            heat_consumption = (param['heating_consumption_calcul'] * heating_need * dh / 1000).to_frame().dot(
                (1 / efficiency).to_frame().T) + param['const heating_consumption_calcul']

    else:
        heat_consumption = param['heating_consumption_calcul'] * heating_need * dh / 1000 * (1 / efficiency) + param[
            'const heating_consumption_calcul']

    return partial_losses, envelope_losses, heating_need, heat_consumption


def primary_heating_consumption(u_wall, u_floor, u_roof, u_windows, dh, efficiency, energy, param, conversion=2.58):
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
    param
    conversion

    Returns
    -------

    """
    heat_consumption = heating_consumption(u_wall, u_floor, u_roof, u_windows, dh, efficiency, param)[3]

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


def certificate_buildings(u_wall, u_floor, u_roof, u_windows, dh, efficiency, energy, param, conversion=2.58):
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
    conversion

    Returns
    -------
    pd.Series
        Primary heating consumption for all buildings in the stock.
    pd.Series
        Certificates for all buildings in the stock.

    """
    primary_heat_consumption = primary_heating_consumption(u_wall, u_floor, u_roof, u_windows, dh, efficiency,
                                                           energy, param, conversion=conversion)
    return primary_heat_consumption, certificate(primary_heat_consumption, param['certificate_bounds'])
