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

from numpy import log
from project.utils import reindex_mi, get_pandas
import os
from datetime import timedelta

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
TEMP_INDOOR = 19 #°C

# Solar heat load
FACTOR_SHADING = 0.6
FACTOR_FRAME = 0.3
FACTOR_NON_PERPENDICULAR = 0.9
# see Methode 3CL
ORIENTATION_FACTOR = {'South': 1.1, 'West': 0.57, 'Est': 0.57, 'North': 0.2}
ORIENTATION_FACTOR['Mean'] = (ORIENTATION_FACTOR['South'] + ORIENTATION_FACTOR['West'] + ORIENTATION_FACTOR['Est'] + ORIENTATION_FACTOR['North']) / 4

# Internal heat sources
INTERNAL_HEAT_SOURCES = 4.17 # W/m2

# Gain factor
INTERNAL_HEAT_CAPACITY = 45 # Wh/m2.K
A_0 = 0.8
TAU_0 = 30

FACTOR_TABULA_3CL = 0.9

# Climatic data
TEMP_BASE = 12 # °C
# HDD_EQ = (temp_indoor - TEMP_EXT_3CL) * 24 * DAYS_HEATING_SEASON
SOLAR_ENERGY_TRANSMITTANCE = 0.62 # data to check

TEMP_EXT_3CL = 7.1 # °C
DAYS_HEATING_SEASON_3CL = 209
SOLAR_RADIATION_3CL = 306.4 # kWh/m2.an

DHW_NEED = pd.Series([15.3, 19.8], index=pd.Index(['Single-family',	'Multi-family'], name='Housing type')) # kWh/m2.a
DHW_EFFICIENCY = {'Electricity-Performance boiler': 0.7,
                  'Electricity-Heat pump air': 0.7,
                  'Electricity-Heat pump water': 2.5,
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

HOURLY_PROFILE_POWER = pd.Series(
    [0.035, 0.039, 0.041, 0.042, 0.046, 0.05, 0.055, 0.058, 0.053, 0.049, 0.045, 0.041, 0.037, 0.034,
     0.03, 0.033, 0.037, 0.042, 0.046, 0.041, 0.037, 0.034, 0.033, 0.042], index=pd.TimedeltaIndex(range(0, 24), unit='h'))

HOURLY_PROFILE_FOSSIL = pd.Series(
    [0.04, 0.038, 0.039, 0.04, 0.041, 0.042, 0.044, 0.046, 0.048, 0.045, 0.043, 0.04, 0.037, 0.033, 0.034, 0.036, 0.039,
     0.043, 0.045, 0.046, 0.047, 0.046, 0.045, 0.043], index=pd.TimedeltaIndex(range(0, 24), unit='h'))

TEMP_SINK = 55

CLIMATE_DATA = {'year': os.path.join('project', 'input', 'climatic', 'climatic_data.csv'),
                'month': os.path.join('project', 'input', 'climatic', 'climatic_data_month.csv'),
                'day': os.path.join('project', 'input', 'climatic', 'climatic_data_daily.csv'),
                'hour': os.path.join('project', 'input', 'climatic', 'climatic_data_daily.csv'),
                'smooth_day': os.path.join('project', 'input', 'climatic', 'climatic_data_smooth_daily.csv')
                }


GAIN_UTILIZATION_FACTOR = False


def conventional_heating_need(u_wall, u_floor, u_roof, u_windows, ratio_surface,
                              th_bridging='Medium', vent_types='Ventilation naturelle', infiltration='Medium',
                              air_rate=None, unobserved=None, climate=None, smooth=False, freq='year',
                              hourly_profile=None, temp_indoor=None, gain_utilization_factor=GAIN_UTILIZATION_FACTOR,
                              ):
    """Seasonal method for space heating need.


    We apply a seasonal method according to EN ISO 13790 to estimate annual space heating demand by building type.
    The detailed calculation can be found in the TABULA project documentation (Loga, 2013).
    In a nutshell, the energy need for heating is the difference between the heat losses and the heat gain.
    The total heat losses result from heat transfer by transmission and ventilation during the heating season
    respectively proportional to the heat transfer coefficient $H_tr$ and $H_ve$.

    To not consider gain_utilization_factor create a difference of 5%. For consistency between results and

    Parameters
    ----------
    u_wall: pd.Series
        Index should include Housing type {'Single-family', 'Multi-family'}.
    u_floor: pd.Series
    u_roof: pd.Series
    u_windows: pd.Series
    ratio_surface: pd.Series
    th_bridging: {'Minimal', 'Low', 'Medium', 'High'}, default None
    vent_types: {'Ventilation naturelle', 'VMC SF auto et VMC double flux', 'VMC SF hydrogérable'}, default None
    infiltration: {'Minimal', 'Low', 'Medium', 'High'}, default None
    air_rate: pd.Series, default None
    unobserved: {'Minimal', 'High'}, default None
    climate: int, default None
        Climatic year to use to calculate heating need.
    smooth: bool, default False
        Use smooth daily data to calculate heating need.
    freq
    hourly_profile: optional, pd.Series
    temp_indoor: optional, default temp_indoor
    gain_utilization_factor: bool, default False
        If False, for simplification we use gain_utilization_factor = 1.

    Returns
    -------
    Conventional heating need (kWh/m2.a)
    """

    temp_ext = TEMP_EXT_3CL
    days_heating_season = DAYS_HEATING_SEASON_3CL
    solar_radiation = SOLAR_RADIATION_3CL
    if temp_indoor is None:
        temp_indoor = TEMP_INDOOR

    if climate is not None:
        if freq == 'year':
            data = get_pandas(CLIMATE_DATA['year'],
                              func=lambda x: pd.read_csv(x, index_col=[0], parse_dates=True))

            temp_ext = float(data.loc[data.index.year == climate, 'TEMP_EXT'])
            days_heating_season = float(data.loc[data.index.year == climate, 'DAYS_HEATING_SEASON'])
            solar_radiation = float(data.loc[data.index.year == climate, 'SOLAR_RADIATION'])
            # print(temp_ext, days_heating_season, solar_radiation)

        else:
            path = CLIMATE_DATA[freq]
            if smooth:
                path = CLIMATE_DATA['smooth_day']

            data = get_pandas(path, func=lambda x: pd.read_csv(x, index_col=[0], parse_dates=True))
            temp_ext = data.loc[data.index.year == climate, 'TEMP_EXT'].rename(None)
            days_heating_season = data.loc[data.index.year == climate, 'DAYS_HEATING_SEASON'].rename(None)
            days_heating_season = days_heating_season.replace({True: 1, False: float('nan')})
            solar_radiation = data.loc[data.index.year == climate, 'SOLAR_RADIATION'].rename(None)

    if unobserved == 'Minimal':
        th_bridging = 'Minimal'
        vent_types = 'VMC SF hydrogérable'
        infiltration = 'Minimal'
    elif unobserved == 'High':
        th_bridging = 'High'
        vent_types = 'Ventilation naturelle'
        infiltration = 'High'

    surface_components = ratio_surface.copy()

    df = pd.concat([u_wall, u_floor, u_roof, u_windows], axis=1, keys=['Wall', 'Floor', 'Roof', 'Windows'])
    surface_components.loc[:, 'Floor'] *= FACTOR_SOIL
    surface_components = reindex_mi(surface_components, df.index)

    coefficient_transmission_transfer = (surface_components * df).sum(axis=1)
    coefficient_transmission_transfer += surface_components.sum(axis=1) * THERMAL_BRIDGING[th_bridging]

    if air_rate is None:
        air_rate = VENTILATION_TYPES[vent_types] + AIR_TIGHTNESS_INFILTRATION[infiltration]

    coefficient_ventilation_transfer = HEAT_CAPACITY_AIR * air_rate * ROOM_HEIGHT
    heat_transfer_coefficient = coefficient_ventilation_transfer + coefficient_transmission_transfer

    solar_coefficient = FACTOR_SHADING * (1 - FACTOR_FRAME) * FACTOR_NON_PERPENDICULAR * SOLAR_ENERGY_TRANSMITTANCE * surface_components.loc[:, 'Windows']

    coefficient = 24 / 1000 * FACTOR_NON_UNIFORM * days_heating_season
    coefficient_climatic = coefficient * (temp_indoor - temp_ext)
    internal_heat_sources = 24 / 1000 * INTERNAL_HEAT_SOURCES * days_heating_season

    if freq == 'year':

        heat_transfer = heat_transfer_coefficient * coefficient_climatic
        solar_load = solar_coefficient * solar_radiation
        heat_gains = solar_load + internal_heat_sources

        if gain_utilization_factor is True:
            time_constant = INTERNAL_HEAT_CAPACITY / (coefficient_transmission_transfer + coefficient_ventilation_transfer)
            a_h = A_0 + time_constant / TAU_0
            heat_balance_ratio = (internal_heat_sources + solar_load) / heat_transfer
            gain_utilization_factor = (1 - heat_balance_ratio ** a_h) / (1 - heat_balance_ratio ** (a_h + 1))
        else:
            gain_utilization_factor = 1

        heat_need = (heat_transfer - heat_gains * gain_utilization_factor) * FACTOR_TABULA_3CL

        return heat_need

    else:
        heat_transfer = heat_transfer_coefficient.rename(None).to_frame().dot(coefficient_climatic.to_frame().T)
        solar_load = solar_coefficient.rename(None).to_frame().dot(solar_radiation.to_frame().T)
        heat_gains = solar_load + internal_heat_sources

        if gain_utilization_factor is True:
            time_constant = INTERNAL_HEAT_CAPACITY / (coefficient_transmission_transfer + coefficient_ventilation_transfer)
            a_h = A_0 + time_constant / TAU_0
            # average over the month
            heat_balance_ratio = heat_gains.groupby(heat_gains.columns.month, axis=1).sum() / heat_transfer.groupby(heat_transfer.columns.month, axis=1).sum()
            gain_utilization_factor = (1 - (heat_balance_ratio.T ** a_h).T) / (1 - (heat_balance_ratio.T ** (a_h + 1)).T)
            temp = []
            for i in gain_utilization_factor.columns.astype(str):
                if len(i) == 1:
                    i = '0{}'.format(i)
                temp.append(i)
            gain_utilization_factor.columns = [np.datetime64('{}-{}'.format(climate, i), 'D') for i in temp]
            gain_utilization_factor = gain_utilization_factor.reindex(heat_gains.columns, axis=1, method='pad')
        else:
            gain_utilization_factor = 1

        heat_need = ((heat_transfer - heat_gains * gain_utilization_factor) * FACTOR_TABULA_3CL).fillna(0)

        heat_need = heat_need.stack(heat_need.columns.names)

        if freq == 'hour':
            if hourly_profile is None:
                hourly_profile = HOURLY_PROFILE_POWER

            heat_need = heat_need.to_frame().dot(hourly_profile.to_frame().T)
            heat_need = heat_need.unstack(['time'])
            heat_need.columns = heat_need.columns.get_level_values(None) + heat_need.columns.get_level_values('time')
        else:
            heat_need = heat_need.unstack(['time'])

        return heat_need.sort_index(axis=1)


def conventional_heating_final(u_wall, u_floor, u_roof, u_windows, ratio_surface, efficiency,
                               th_bridging='Medium', vent_types='Ventilation naturelle', infiltration='Medium',
                               air_rate=None, unobserved=None, climate=None, freq='year', smooth=False,
                               temp_indoor=None, gain_utilization_factor=GAIN_UTILIZATION_FACTOR,
                               efficiency_hour=False):
    """Monthly stead-state space heating final energy delivered.


    Heat-pump formula come from Stafell et al., 2012.

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
    climate: int, default None
        Climatic year to use to calculate heating need.
    freq
    smooth: bool, default False
        Use smooth daily data to calculate heating need.
    temp_indoor
    gain_utilization_factor: bool, default False
        If False, for simplification we use gain_utilization_factor = 1.
    efficiency_hour

    Returns
    -------

    """
    heat_need = conventional_heating_need(u_wall, u_floor, u_roof, u_windows, ratio_surface,
                                          th_bridging=th_bridging, vent_types=vent_types,
                                          infiltration=infiltration, air_rate=air_rate, unobserved=unobserved,
                                          climate=climate, freq=freq, smooth=smooth,
                                          temp_indoor=temp_indoor, gain_utilization_factor=gain_utilization_factor)

    if (freq == 'hour') and (efficiency_hour is True):
        path = CLIMATE_DATA[freq]
        if smooth:
            path = CLIMATE_DATA['smooth_day']

        data = get_pandas(path, func=lambda x: pd.read_csv(x, index_col=[0], parse_dates=True))
        temp_ext = data.loc[data.index.year == climate, 'TEMP_EXT'].rename(None)
        delta_temperature = TEMP_SINK - temp_ext
        # TODO replace 0 by nan
        efficiency_hp = 6.81 - 0.121 * delta_temperature + 0.00063 * (delta_temperature ** 2)

        heat_pumps = ['Electricity-Heat pump air', 'Electricity-Heat pump water']
        efficiency = pd.concat([efficiency] * efficiency_hp.shape[0], axis=1, keys=efficiency_hp.index, names=efficiency_hp.index.names)
        idx = efficiency[efficiency.index.get_level_values('Heating system').isin(heat_pumps)].index
        for i in idx:
            efficiency.loc[i, :] = efficiency_hp
        efficiency = efficiency.reindex(heat_need.columns, method='pad', axis=1)

    if isinstance(heat_need, pd.Series):
        return heat_need / efficiency
    if isinstance(heat_need, pd.DataFrame):
        if isinstance(efficiency, pd.Series):
            return (heat_need.T / efficiency).T
        elif isinstance(efficiency, pd.DataFrame):
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
        Should include Housing type and Energy.
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
    ratio_surface: pd.DataFrame

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
    hdd
    efficiency
    energy
    ratio_surface

    Returns
    -------
    pd.Series
        Primary heating consump_tion for all buildings in the stock.
    pd.Series
        Certificates for all buildings in the stock.

    """
    primary_heat_consumption = primary_heating_consumption(u_wall, u_floor, u_roof, u_windows, hdd, efficiency,
                                                           energy, ratio_surface, conversion=CONVERSION)
    return primary_heat_consumption, certificate(primary_heat_consumption)


def heat_intensity(budget):
    return -0.191 * budget.apply(log) + 0.1105


if __name__ == '__main__':
    from project.model import get_inputs
    output = get_inputs(variables=['buildings'])
    buildings = output['buildings']

    heating_need_year = buildings.heating_need(freq='year', climate=2006)
    energy = heating_need_year.index.get_level_values('Heating system').str.split('-').str[0]
    heating_need_year = heating_need_year.groupby(energy).sum()

    heating_need_month = buildings.heating_need(freq='month', climate=2006)
    heating_need_month = heating_need_month.sum(axis=1)
    energy = heating_need_month.index.get_level_values('Heating system').str.split('-').str[0]
    heating_need_month = heating_need_month.groupby(energy).sum()

    heating_need_day = buildings.heating_need(freq='day', climate=2006)
    heating_need_day = heating_need_day.sum(axis=1)
    energy = heating_need_day.index.get_level_values('Heating system').str.split('-').str[0]
    heating_need_day = heating_need_day.groupby(energy).sum()

    heating_need_hour = buildings.heating_need(freq='hour', climate=2006)
    heating_need_hour = heating_need_hour.sum(axis=1)
    energy = heating_need_hour.index.get_level_values('Heating system').str.split('-').str[0]
    heating_need_hour = heating_need_hour.groupby(energy).sum()

    print('break')

    print('break')
