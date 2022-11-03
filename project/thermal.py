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

from project.utils import reindex_mi

"""LOGISTIC_COEFFICIENT = pd.read_csv('project/input/logistic_regression_coefficient_epc.csv', index_col=[0])
LOGISTIC_COEFFICIENT.columns = ['Intercept', 'Proxy_conso_square']
LOGISTIC_COEFFICIENT.index.names = ['Performance']"""
CONVERSION = 2.3
HDD = 55706
CERTIFICATE_3USES_BOUNDARIES = {
    'A': [0, 50],
    'B': [50, 90],
    'C': [90, 150],
    'D': [150, 230],
    'E': [230, 330],
    'F': [330, 450],
    'G': [450, 1000],
}


# Transmission heat losses
THERMAL_BRIDGING = {'Minimal': 0,
                    'Low': 0.05,
                    'Medium': 0.1,
                    'High': 0.15}
FACTOR_SOIL = 0.5

# Ventilation heat losses
HEAT_CAPACITY_AIR = 0.34 # Wh/m3.K

AIR_TIGHTNESS_INFILTRATION = {'Minimal': 0.05,
                              'Low': 0.1,
                              'Medium': 0.2,
                              'High': 0.5}

ROOM_HEIGHT = 2.5 # m
VENTILATION_TYPES = {'Ventilation naturelle': 0.4,
                     'VMC SF auto et VMC double flux': 0.3,
                     'VMC SF hydrogérable': 0.2}

# Heat transfer
FACTOR_NON_UNIFORM = 0.9
TEMP_INT = 19 #°C

# Solar heat load
FACTOR_SHADING = 0.6
FACTOR_FRACTION = 0.3
FACTOR_NON_PERPENDICULAR = 0.9

# Internal heat sources
INTERNAL_HEAT_SOURCES = 4.17 # W/m2

# Gain factor
INTERNAL_HEAT_CAPACITY = 45 # Wh/m2.K
A_0 = 0.8
TAU_0 = 30

FACTOR_TABULA_3CL = 0.9

# Climatic data
TEMP_EXT = 7.1 #°C
DAYS_HEATING_SEASON = 209
HDD_EQ = (TEMP_INT - TEMP_EXT) * 24 * DAYS_HEATING_SEASON
SOLAR_ENERGY_TRANSMITTANCE = 0.62
SOLAR_RADIATION = 306.4 # kWh/m2.an

DHW_NEED = pd.Series([15.3, 19.8], index=pd.Index(['Single-family',	'Multi-family'], name='Housing type')) # kWh/m2.a
DHW_EFFICIENCY = {'Electricity-Performance boiler': 0.7,
                  'Electricity-Heat pump': 2.5,
                  'Natural gas-Performance boiler': 0.6,
                  'Natural gas-Standard boiler': 0.6,
                  'Oil fuel-Performance boiler': 0.6,
                  'Oil fuel-Standard boiler': 0.6,
                  'Wood fuel-Performance boiler': 0.6,
                  'Wood fuel-Standard boiler': 0.6
                  }


C_LIGHT = 0.9
P_LIGHT = 1.4 # W/m2
HOURS_LIGHT = 2123 # h


def conventional_heating_need(u_wall, u_floor, u_roof, u_windows, ratio_surface,
                              th_bridging='Medium', vent_types='Ventilation naturelle', infiltration='Medium',
                              air_rate=None, unobserved=None):
    """Monthly stead-state space heating need.

    Parameters
    ----------
    u_wall: pd.Series
        Index should include Housing type {'Single-family', 'Multi-family'}.
    u_floor: pd.Series
    u_roof: pd.Series
    u_windows: pd.Series
    ratio_surface: pd.Series
    th_bridging: {'Minimal', 'Low', 'Medium', 'High'}
    vent_types: {'Ventilation naturelle', 'VMC SF auto et VMC double flux', 'VMC SF hydrogérable'}
    infiltration: {'Minimal', 'Low', 'Medium', 'High'}
    air_rate: default None
    unobserved: {'Minimal', 'High'}, default None

    Returns
    -------
    Conventional heating need (kWh/m2.a)
    """

    if unobserved == 'Minimal':
        th_bridging = 'Minimal'
        vent_types = 'VMC SF hydrogérable'
        infiltration = 'Minimal'
    elif unobserved == 'High':
        th_bridging = 'High'
        vent_types = 'Ventilation naturelle'
        infiltration = 'High'

    data = pd.concat([u_wall, u_floor, u_roof, u_windows], axis=1, keys=['Wall', 'Floor', 'Roof', 'Windows'])
    ratio_surface.loc[:, 'Floor'] *= FACTOR_SOIL
    surface_components = reindex_mi(ratio_surface, data.index)

    coefficient_transmission_transfer = (surface_components * data).sum(axis=1)
    coefficient_transmission_transfer += surface_components.sum(axis=1) * THERMAL_BRIDGING[th_bridging]

    if air_rate is None:
        air_rate = VENTILATION_TYPES[vent_types] + AIR_TIGHTNESS_INFILTRATION[infiltration]

    coefficient_ventilation_transfer = HEAT_CAPACITY_AIR * air_rate * ROOM_HEIGHT

    coefficient_climatic = 24 / 1000 * FACTOR_NON_UNIFORM * (TEMP_INT - TEMP_EXT) * DAYS_HEATING_SEASON

    heat_transfer = (coefficient_ventilation_transfer + coefficient_transmission_transfer) * coefficient_climatic

    solar_load = FACTOR_SHADING * (1 - FACTOR_FRACTION) * FACTOR_NON_PERPENDICULAR * SOLAR_ENERGY_TRANSMITTANCE * SOLAR_RADIATION * surface_components.loc[:, 'Windows']

    internal_heat_sources = 24 / 1000 * INTERNAL_HEAT_SOURCES * DAYS_HEATING_SEASON

    time_constant = INTERNAL_HEAT_CAPACITY / (coefficient_transmission_transfer + coefficient_ventilation_transfer)
    a_h = A_0 + time_constant / TAU_0
    heat_balance_ratio = (internal_heat_sources + solar_load) / heat_transfer
    gain_utilization_factor = (1 - heat_balance_ratio**a_h) / (1 - heat_balance_ratio**(a_h + 1))

    heat_gains = solar_load + internal_heat_sources

    heat_need = (heat_transfer - heat_gains * gain_utilization_factor) * FACTOR_TABULA_3CL
    return heat_need


def conventional_heating_final(u_wall, u_floor, u_roof, u_windows, ratio_surface, efficiency,
                               th_bridging='Medium', vent_types='Ventilation naturelle', infiltration='Medium',
                               air_rate=None, unobserved=None
                               ):
    """Monthly stead-state space heating final energy delivered.

    Parameters
    ----------
    u_wall: pd.Series
    u_floor: pd.Series
    u_roof: pd.Series
    u_windows: pd.Series
    ratio_surface: pd.Series
    efficiency: pd.Series
    th_bridging: {'Minimal', 'Low', 'Medium', 'High'}
    vent_types: {'Ventilation naturelle', 'VMC SF auto et VMC double flux', 'VMC SF hydrogérable'}
    infiltration: {'Minimal', 'Low', 'Medium', 'High'}
    air_rate: default None
    unobserved: {'Minimal', 'High'}, default None

    Returns
    -------

    """
    heat_need = conventional_heating_need(u_wall, u_floor, u_roof, u_windows, ratio_surface,
                                          th_bridging=th_bridging, vent_types=vent_types,
                                          infiltration=infiltration, air_rate=air_rate, unobserved=unobserved,
                                          )
    return heat_need / efficiency


def conventional_dhw_final(index):
    """Calculate dhw final energy consumption.

    Parameters
    ----------
    index: pd.MultiIndex

    Returns
    -------

    """
    efficiency = pd.Series(index.get_level_values('Heating system')).astype('object').replace(DHW_EFFICIENCY).set_axis(index, axis=0)
    dhw_need = DHW_NEED.reindex(index.get_level_values('Housing type')).set_axis(index, axis=0)
    return dhw_need / efficiency


def conventional_energy_3uses(u_wall, u_floor, u_roof, u_windows, ratio_surface, efficiency, index,
                              th_bridging='Medium', vent_types='Ventilation naturelle', infiltration='Medium',
                              air_rate=None, unobserved=None
                              ):
    """Space heating conventional, and energy performance certificate.

    Method before july 2021.

    Parameters
    ----------
    u_wall: pd.Series
    u_floor: pd.Series
    u_roof: pd.Series
    u_windows: pd.Series
    ratio_surface: pd.Series
    efficiency: pd.Series
    index: pd.MultiIndex or pd.Index
        Index should include Housing type and Energy.
    th_bridging: {'Minimal', 'Low', 'Medium', 'High'}
    vent_types: {'Ventilation naturelle', 'VMC SF auto et VMC double flux', 'VMC SF hydrogérable'}
    infiltration: {'Minimal', 'Low', 'Medium', 'High'}
    air_rate: default None
    unobserved: {'Minimal', 'High'}, default None

    Returns
    -------

    """

    heating_final = conventional_heating_final(u_wall, u_floor, u_roof, u_windows, ratio_surface, efficiency,
                                               th_bridging=th_bridging, vent_types=vent_types,
                                               infiltration=infiltration, air_rate=air_rate, unobserved=unobserved
                                               )
    dhw_final = conventional_dhw_final(index)
    ac_final = 0
    energy_final = heating_final + dhw_final + ac_final
    if 'Energy' not in index.names:
        energy_carrier = index.get_level_values('Heating system').str.split('-').str[0].rename('Energy')
    else:
        energy_carrier = index.get_level_values('Energy')

    energy_primary = final2primary(energy_final, energy_carrier)
    performance = find_certificate(energy_primary)
    return performance, energy_primary


def find_certificate(primary_consumption):
    """Returns energy performance certificate from A to G.

    Parameters
    ----------
    primary_consumption: float or pd.Series or pd.DataFrame
        Space heating energy consumption (kWh PE / m2.year)
    Returns
    -------
    float or pd.Series or pd.DataFrame
        Energy performance certificate.
    """

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


def stat_model_heating_consumption(df, a=0.921323, b=0.634717):
    """Statistical model based on ADEME DPE data.

    X = (Deper_mur + Deper_baie + Deper_plancher + Deper_plafond) / Efficiency * DDH
    Conso = X ** 1.746973 * exp(1.536192)
    """
    return (df ** a) * np.exp(b)


def stat_model_3uses_consumption(df, a=0.846102, b=1.029880):
    """Statistical model based on ADEME DPE data.

    X = Primary space hating energy consumption (kWh EP / m2)
    Conso = X ** 0.846102 * exp(1.029880)
    """
    return (df ** a) * np.exp(b)


def stat_heating_consumption(u_wall, u_floor, u_roof, u_windows, efficiency, ratio_surface, hdd):
    """Calculate space heating consumption in kWh/m2.year based on insulation performance and heating system efficiency.

    Function simulates the 3CL-method, and use parameters to estimate unobserved variables.

    Parameters
    ----------
    u_wall: float or pd.Series
    u_floor: float or pd.Series
    u_roof: float or pd.Series
    u_windows: float or pd.Series
    hdd: float or pd.Series
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
            indicator_losses = partial_losses / efficiency * hdd / 1000
            consumption = stat_model_heating_consumption(indicator_losses).rename('Consumption')

        else:
            indicator_losses = (partial_losses * hdd / 1000).to_frame().dot(
                (1 / efficiency).to_frame().T)
            consumption = stat_model_heating_consumption(indicator_losses)

    else:
        indicator_losses = partial_losses / efficiency * hdd / 1000
        consumption = stat_model_heating_consumption(indicator_losses).rename('Consumption')

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


def primary_heating_consumption(u_wall, u_floor, u_roof, u_windows, efficiency, energy, ratio_surface, hdd,
                                conversion=CONVERSION):
    """Convert final to primary heating consumption.

    Parameters
    ----------
    u_wall
    u_floor
    u_roof
    u_windows
    hdd
    efficiency
    energy
    ratio_surface
    conversion

    Returns
    -------
    """
    # data = pd.concat([u_wall, u_floor, u_roof, u_windows], axis=1, keys=['Wall', 'Floor', 'Roof', 'Windows'])
    heat_consumption = stat_heating_consumption(u_wall, u_floor, u_roof, u_windows, efficiency, ratio_surface, hdd)
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
    primary_consumption = stat_model_3uses_consumption(df)

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


def certificate_buildings(u_wall, u_floor, u_roof, u_windows, hdd, efficiency, energy, ratio_surface):
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
    primary_heat_consumption = primary_heating_consumption(u_wall, u_floor, u_roof, u_windows, hdd, efficiency,
                                                           energy, ratio_surface, conversion=CONVERSION)
    return primary_heat_consumption, certificate(primary_heat_consumption)

