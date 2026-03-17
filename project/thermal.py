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
    'G': [450, 9999],
}

# kWhPE/m2.a
CERTIFICATE_5USES_BOUNDARIES_ENERGY = {
    'A': [0, 70],
    'B': [70, 110],
    'C': [110, 180],
    'D': [180, 250],
    'E': [250, 330],
    'F': [330, 420],
    'G': [420, 9999],
}
# kgCO2/m2.a
CERTIFICATE_5USES_BOUNDARIES_EMISSION = {
    'A': [0, 6],
    'B': [6, 11],
    'C': [11, 30],
    'D': [30, 50],
    'E': [50, 70],
    'F': [70, 100],
    'G': [100, 1000],
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
TEMP_OUTDOOR_PEAK = -6

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
DHW_EFFICIENCY = {'Electricity-Direct electric': 0.95,
                  'Electricity-Wood stove': 0.95,
                  'Electricity-Heat pump air': 0.95, # previously at 0.95
                  'Electricity-Heat pump': 2.5,
                  'Electricity-Heat pump water': 2.5,
                  'Natural gas-Performance boiler': 0.6,
                  'Natural gas-Standard boiler': 0.6,
                  'Natural gas-Collective boiler': 0.6,
                  'Oil fuel-Performance boiler': 0.6,
                  'Oil fuel-Standard boiler': 0.6,
                  'Oil fuel-Collective boiler': 0.6,
                  'Wood fuel-Performance boiler': 0.6,
                  'Wood fuel-Standard boiler': 0.6,
                  'Heating-District heating': 0.6
                  }

# kWh/m2.a from EDF
AUXILIARY_CONSUMPTION = {'Electricity-Direct electric': 1.4,
                         'Electricity-Wood stove': 1.4,
                         'Electricity-Heat pump air': 2.17,
                         'Electricity-Heat pump': 2.17,
                         'Electricity-Heat pump water': 2.17,
                         'Natural gas-Performance boiler': 3.65,
                         'Natural gas-Standard boiler': 3.65,
                         'Natural gas-Collective boiler': 3.65,
                         'Oil fuel-Performance boiler': 3.83,
                         'Oil fuel-Standard boiler': 3.83,
                         'Oil fuel-Collective boiler': 3.83,
                         'Wood fuel-Performance boiler': 2.07,
                         'Wood fuel-Standard boiler': 2.07,
                         'Heating-District heating': 1.75
                         }
AUXILIARY_CONSUMPTION = 3

CARBON_CONTENT = {'Electricity': 0.079,
                  'Natural gas': 0.227,
                  'Oil fuel': 0.324,
                  'Wood fuel': 0.03,
                  'Heating': 0.227
                  }
CARBON_CONTENT = {'Electricity-Direct electric': 0.079,
                  'Electricity-Wood stove': 0.079,
                  'Electricity-Heat pump air': 0.079,
                  'Electricity-Heat pump': 0.079,
                  'Electricity-Heat pump water': 0.079,
                  'Natural gas-Performance boiler': 0.227,
                  'Natural gas-Standard boiler': 0.227,
                  'Natural gas-Collective boiler': 0.227,
                  'Oil fuel-Performance boiler': 0.324,
                  'Oil fuel-Standard boiler': 0.324,
                  'Oil fuel-Collective boiler': 0.324,
                  'Wood fuel-Performance boiler': 0.03,
                  'Wood fuel-Standard boiler': 0.03,
                  'Heating-District heating': 0.03
                  }
C_LIGHT = 0.9
P_LIGHT = 1.4 # W/m2
HOURS_LIGHT = 1500 # h
CONSUMPTION_LIGHT = (C_LIGHT * P_LIGHT * HOURS_LIGHT) / 1e3 # kWh/m2.a

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


def size_heating_system(u_wall, u_floor, u_roof, u_windows, ratio_surface,
                        th_bridging='Medium', vent_types='Ventilation naturelle', infiltration='Medium',
                        air_rate=None, unobserved=None, temp_indoor=None):
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
    temp_indoor: optional, default temp_indoor

    Returns
    -------
    Conventional heating need (kWh/m2.a)
    """

    if temp_indoor is None:
        temp_indoor = TEMP_INDOOR

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

    size = heat_transfer_coefficient * (temp_indoor - TEMP_OUTDOOR_PEAK)

    return size


def conventional_heating_need(
    u_wall, u_floor, u_roof, u_windows, ratio_surface,
    th_bridging='Medium', vent_types='Ventilation naturelle', infiltration='Medium',
    air_rate=None, unobserved=None, climate=None, smooth=False, freq='year',
    hourly_profile=None, temp_indoor=None, gain_utilization_factor=GAIN_UTILIZATION_FACTOR,
    zcl_thermal_parameters=None,
    # --- 4.2 Physics Additions ---
    roof_albedo=None,             
    h_ext_roof=None,              
    wind_speed=None
):
    """Seasonal method for space heating need. Includes roof solar absorption channel."""
    if zcl_thermal_parameters is None:
        zcl_thermal_parameters = {
            'activated': False, 'temperature_difference': 0, 'days_heating_factor': 1
        }
    else:
        zcl_thermal_parameters = dict(zcl_thermal_parameters)

    cfg = zcl_thermal_parameters.get('config', {})
    # ==================================================================
    # Feature Toggle for Roof Albedo
    # Read physics.roof_albedo_activated from the top-level JSON. If not configured, default to False for backward compatibility.
    # ==================================================================
    clim_cfg_early = cfg.get('climate_data', {})
    is_albedo_activated = clim_cfg_early.get('roof_albedo_activated', False)

    if not is_albedo_activated:
        roof_albedo = None
        h_ext_roof = None
        wind_speed = None
    # ==================================================================
    
    if 'freq' in cfg: freq = cfg['freq']
    if 'smooth' in cfg: smooth = cfg['smooth']
    if cfg.get('climate') is not None: climate = cfg['climate']

    temp_ext = TEMP_EXT_3CL + zcl_thermal_parameters.get('temperature_difference', 0)
    days_heating_season = DAYS_HEATING_SEASON_3CL * zcl_thermal_parameters.get('days_heating_factor', 1)
    solar_radiation = SOLAR_RADIATION_3CL

    if temp_indoor is None:
        temp_indoor = TEMP_INDOOR
    # =========================================================
    # To be removed: Print once if roof_albedo is received
    if roof_albedo is not None and not getattr(conventional_heating_need, '_has_printed_physics', False):
        print(f"\n[Info Physics] Received roof_albedo, dynamic wind speed (v) and roof convective heat transfer (h) calculations activated!")
        conventional_heating_need._has_printed_physics = True
    # =========================================================

    wind_speed_from_file = None

    # Priority: 1. climate parameter in config, 2. default climate data
    if climate is not None:
        key = "smooth_day" if smooth else freq
        clim_cfg = cfg.get('climate_data', {})
        path = clim_cfg.get(key, CLIMATE_DATA.get(key))

        data = get_pandas(path, func=lambda x: pd.read_csv(x, index_col=[0], parse_dates=True))
        current_zcl = zcl_thermal_parameters.get('zcl', 'H1a')
        delta_temp = zcl_thermal_parameters.get('temperature_difference', 0)
        
        if freq == 'year':
            temp_ext = float(data.loc[data.index.year == climate, 'TEMP_EXT']) + delta_temp
            days_heating_season = float(data.loc[data.index.year == climate, 'DAYS_HEATING_SEASON'])
            solar_radiation = float(data.loc[data.index.year == climate, 'SOLAR_RADIATION'])
        else:
            temp_ext = data.loc[data.index.year == climate, 'TEMP_EXT'].rename(None) + delta_temp
            days_heating_season = data.loc[data.index.year == climate, 'DAYS_HEATING_SEASON'].rename(None)
            days_heating_season = days_heating_season.replace({True: 1, False: float('nan')})
            solar_radiation = data.loc[data.index.year == climate, 'SOLAR_RADIATION'].rename(None)

        if wind_speed is None and is_albedo_activated:
            ws_path = clim_cfg.get('wind_speed_file')
            is_zcl_activated = zcl_thermal_parameters.get('activated', False)
            # Wind speed is only relevant if subregion is activated, and we have a path to the wind speed data file
            if is_zcl_activated and ws_path is not None:
                try:
                    ws_data = get_pandas(ws_path, func=lambda x: pd.read_csv(x, index_col=[0, 1]))
                    # ==================================================================
                    # [Debug print] Check if wind speed file been called and used
                    # ==================================================================
                    if not getattr(conventional_heating_need, '_has_printed_wind', False):
                        print(f"[WIND SUCCESS] Loaded dynamically for Zone {current_zcl}! ")
                        conventional_heating_need._has_printed_wind = True
                    # ==================================================================
                    is_urban = cfg.get('urban_rural', {}).get('activated', False)
                    # 判断当前是否激活了城乡差异
                    is_urban_activated = cfg.get('urban_rural', {}).get('activated', False)
                    
                    if not is_urban_activated:
                        wind_speed_from_file = float(ws_data.loc[(current_zcl, 'Winter'), 'All'])
                    else:
                        # ==== 激活了城乡差异 ====
                        
                        # 尝试1：看顶层是否直接通过字典传了具体的 area (例如 'Urban' 或 'Rural')
                        current_area = zcl_thermal_parameters.get('area')
                        if current_area in ['Urban', 'Rural']:
                            wind_speed_from_file = float(ws_data.loc[(current_zcl, 'Winter'), current_area])
                            
                        # 尝试2：如果字典里没传，看看建筑存量矩阵 (u_wall) 的索引里有没有城乡标识列
                        else:
                            # 寻找名为 'area', 'urban', 'rural' 相关的 MultiIndex 层级
                            possible_names = [n for n in u_wall.index.names if n and ('urban' in n.lower() or n.lower() == 'area')]
                            
                            if possible_names:
                                area_level = possible_names[0]
                                
                                # 读取 CSV 里对应的三个标量风速
                                v_all = float(ws_data.loc[(current_zcl, 'Winter'), 'All'])
                                v_urban = float(ws_data.loc[(current_zcl, 'Winter'), 'Urban'])
                                v_rural = float(ws_data.loc[(current_zcl, 'Winter'), 'Rural'])
                                
                                # 此时风速不再是一个标量，而是变成了一个和建筑矩阵完全对齐的 Pandas Series
                                wind_speed_from_file = pd.Series(v_all, index=u_wall.index)
                                
                                # 根据 Index 的值进行向量化掩码赋值 (兼容大小写)
                                idx_urban = u_wall.index.get_level_values(area_level).astype(str).str.lower().isin(['urban', 'urbain'])
                                idx_rural = u_wall.index.get_level_values(area_level).astype(str).str.lower().isin(['rural'])
                                
                                wind_speed_from_file.loc[idx_urban] = v_urban
                                wind_speed_from_file.loc[idx_rural] = v_rural
                                
                            # 兜底：既没传参，矩阵里也没对应层级，回退到 All 以防崩溃
                            else:
                                wind_speed_from_file = float(ws_data.loc[(current_zcl, 'Winter'), 'All'])

                except Exception as e:
                    print(f"[Warning] can not read wind speed for {current_zcl} in winter ({e}), using default value 3.0")
                    wind_speed_from_file = 3.0
            else:
                wind_speed_from_file = 3.0

    # ---- U-values and Surfaces ----
    if unobserved == 'Minimal':
        th_bridging = 'Minimal'; vent_types = 'VMC SF hydrogérable'; infiltration = 'Minimal'
    elif unobserved == 'High':
        th_bridging = 'High'; vent_types = 'Ventilation naturelle'; infiltration = 'High'

    surface_components = ratio_surface.copy()
    df_u = pd.concat([u_wall, u_floor, u_roof, u_windows], axis=1, keys=['Wall', 'Floor', 'Roof', 'Windows'])
    surface_components.loc[:, 'Floor'] *= FACTOR_SOIL
    surface_components = reindex_mi(surface_components, df_u.index)

    H_tr_components = (surface_components * df_u).sum(axis=1)
    H_tr = H_tr_components + surface_components.sum(axis=1) * THERMAL_BRIDGING[th_bridging]

    if air_rate is None:
        air_rate = VENTILATION_TYPES[vent_types] + AIR_TIGHTNESS_INFILTRATION[infiltration]
    H_ve = HEAT_CAPACITY_AIR * air_rate * ROOM_HEIGHT
    H_total = H_tr + H_ve

    solar_coefficient = (FACTOR_SHADING * (1 - FACTOR_FRAME) * FACTOR_NON_PERPENDICULAR *
                         SOLAR_ENERGY_TRANSMITTANCE * surface_components.loc[:, 'Windows'])

    coefficient = 24 / 1000 * FACTOR_NON_UNIFORM * days_heating_season
    coefficient_climatic = coefficient * (temp_indoor - temp_ext)
    internal_heat_sources = 24 / 1000 * INTERNAL_HEAT_SOURCES * days_heating_season

    if freq == 'year':
        heat_transfer = H_total * coefficient_climatic
        solar_load = solar_coefficient * solar_radiation
        heat_gains = solar_load + internal_heat_sources

        if roof_albedo is not None:
            v = wind_speed if wind_speed is not None else wind_speed_from_file
            v = v if v is not None else 3.0
            h = h_ext_roof if h_ext_roof is not None else (9.0 + 4.0 * v)
            
            alpha = 1.0 - roof_albedo
            H_roof = surface_components.loc[:, 'Roof'] * df_u.loc[:, 'Roof']
            roof_factor = H_roof * (alpha / h)
            annual_roof_gain = roof_factor * solar_radiation * FACTOR_NON_UNIFORM
            heat_gains = heat_gains + annual_roof_gain

        if gain_utilization_factor is True:
            time_constant = INTERNAL_HEAT_CAPACITY / (H_tr_components + H_ve)
            a_h = A_0 + time_constant / TAU_0
            heat_balance_ratio = heat_gains / heat_transfer
            gain_utilization_factor_eff = (1 - heat_balance_ratio ** a_h) / (1 - heat_balance_ratio ** (a_h + 1))
        else:
            gain_utilization_factor_eff = 1

        factor_tabula_3cl = 1.0 if zcl_thermal_parameters.get('activated') else FACTOR_TABULA_3CL
        heat_need = (heat_transfer - heat_gains * gain_utilization_factor_eff) * factor_tabula_3cl
        # ========================================
        # [Debug] heat_need can not be negative, if negative, set to 0
        if hasattr(heat_need, 'clip'):
            heat_need = heat_need.clip(lower=0)
        else:
            heat_need = max(0, heat_need)
        # ========================================
        return heat_need

    else:
        heat_transfer = H_total.rename(None).to_frame().dot(coefficient_climatic.to_frame().T)
        solar_load = solar_coefficient.rename(None).to_frame().dot(solar_radiation.to_frame().T)
        heat_gains = solar_load + internal_heat_sources

        if roof_albedo is not None:
            v = wind_speed if wind_speed is not None else wind_speed_from_file
            v = v if v is not None else 3.0
            h = h_ext_roof if h_ext_roof is not None else (9.0 + 4.0 * v)
            
            alpha = 1.0 - roof_albedo.rename(None)
            H_roof = surface_components.loc[:, 'Roof'] * df_u.loc[:, 'Roof']
            roof_factor = H_roof * (alpha / h)
            roof_solar_load = roof_factor.rename(None).to_frame().dot(solar_radiation.to_frame().T)
            heat_gains = heat_gains + (roof_solar_load * FACTOR_NON_UNIFORM)

        if gain_utilization_factor is True:
            time_constant = INTERNAL_HEAT_CAPACITY / (H_tr_components + H_ve)
            a_h = A_0 + time_constant / TAU_0
            heat_balance_ratio = (heat_gains.groupby(heat_gains.columns.month, axis=1).sum() /
                                  heat_transfer.groupby(heat_transfer.columns.month, axis=1).sum())
            guf_month = (1 - (heat_balance_ratio.T ** a_h).T) / (1 - (heat_balance_ratio.T ** (a_h + 1)).T)

            temp = []
            for i in guf_month.columns.astype(str):
                if len(i) == 1:
                    i = '0{}'.format(i)
                temp.append(i)
            guf_month.columns = [np.datetime64('{}-{}'.format(climate, i), 'D') for i in temp]
            gain_utilization_factor_eff = guf_month.reindex(heat_gains.columns, axis=1, method='pad')
        else:
            gain_utilization_factor_eff = 1

        factor_tabula_3cl = 1.0 if zcl_thermal_parameters.get('activated') else FACTOR_TABULA_3CL

        heat_need = ((heat_transfer - heat_gains * gain_utilization_factor_eff) * factor_tabula_3cl).fillna(0)
        heat_need = heat_need.stack(heat_need.columns.names)

        if freq == 'hour':
            if hourly_profile is None:
                hourly_profile = HOURLY_PROFILE_POWER

            heat_need = heat_need.to_frame().dot(hourly_profile.to_frame().T)
            heat_need = heat_need.unstack(['time'])
            heat_need.columns = heat_need.columns.get_level_values(None) + heat_need.columns.get_level_values('time')
        else:
            heat_need = heat_need.unstack(['time'])

        # ========================================
        # [Debug] heat_need can not be negative, if negative, set to 0
        heat_need = heat_need.clip(lower=0)
        # ========================================

        return heat_need.sort_index(axis=1)


def conventional_heating_final(
    u_wall, u_floor, u_roof, u_windows, ratio_surface, efficiency,
    th_bridging='Medium', vent_types='Ventilation naturelle', infiltration='Medium',
    air_rate=None, unobserved=None, climate=None, freq='year', smooth=False,
    temp_indoor=None, gain_utilization_factor=GAIN_UTILIZATION_FACTOR,
    efficiency_hour=False, hourly_profile=None, temp_sink=None,
    zcl_thermal_parameters=None,
    # --- 4.2 Physics Additions ---
    roof_albedo=None,             
    h_ext_roof=None,              
    wind_speed=None
):
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
    # =========================================================
    if zcl_thermal_parameters is None:
        zcl_thermal_parameters = {}
    cfg = zcl_thermal_parameters.get('config', {})
    if 'freq' in cfg: freq = cfg['freq']
    if 'smooth' in cfg: smooth = cfg['smooth']
    if cfg.get('climate') is not None: climate = cfg['climate']
    # =========================================================

    heat_need = conventional_heating_need(
        u_wall, u_floor, u_roof, u_windows, ratio_surface,
        th_bridging=th_bridging, vent_types=vent_types,
        infiltration=infiltration, air_rate=air_rate, unobserved=unobserved,
        climate=climate, freq=freq, smooth=smooth,
        temp_indoor=temp_indoor, gain_utilization_factor=gain_utilization_factor,
        hourly_profile=hourly_profile, zcl_thermal_parameters=zcl_thermal_parameters,
        roof_albedo=roof_albedo, h_ext_roof=h_ext_roof, wind_speed=wind_speed
    )

    if (freq == 'hour' or freq == 'day') and (efficiency_hour is True):
        key = "smooth_day" if smooth else freq
        clim_cfg = cfg.get('climate_data', {})
        path = clim_cfg.get(key, CLIMATE_DATA.get(key))
        
        data = get_pandas(path, func=lambda x: pd.read_csv(x, index_col=[0], parse_dates=True))
        delta_temp = zcl_thermal_parameters.get('temperature_difference', 0)
        temp_ext = data.loc[data.index.year == climate, 'TEMP_EXT'].rename(None) + delta_temp
        
        if temp_sink is None:
            temp_sink = TEMP_SINK
        delta_temperature = temp_sink - temp_ext
        
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
            final_energy = (heat_need.T / efficiency).T
        elif isinstance(efficiency, pd.DataFrame):
            final_energy = heat_need / efficiency
            
        if freq != 'year':
            return final_energy.sum(axis=1)
            
        return final_energy


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
                              air_rate=None, unobserved=None, method='3uses',zcl_thermal_parameters=None,
                              # New parameters for 4.2 Physics Additions
                              roof_albedo=None, h_ext_roof=None, wind_speed=None
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
        Should include Housing type and Heating system.
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
                                               infiltration=infiltration, air_rate=air_rate, unobserved=unobserved,
                                               zcl_thermal_parameters=zcl_thermal_parameters,
                                                # New parameters for 4.2 Physics Additions
                                                roof_albedo=roof_albedo, 
                                                h_ext_roof=h_ext_roof, 
                                                wind_speed=wind_speed
                                               )
    dhw_final = conventional_dhw_final(index)
    ac_final = 0
    energy_final = heating_final + dhw_final + ac_final
    if 'Energy' not in index.names:
        energy_carrier = index.get_level_values('Heating system').str.split('-').str[0].rename('Energy')
    else:
        energy_carrier = index.get_level_values('Energy')

    energy_primary = final2primary(energy_final, energy_carrier)
    other_consumptions = None
    if method == '5uses':
        other_consumptions = (CONSUMPTION_LIGHT + AUXILIARY_CONSUMPTION) * CONVERSION

    performance = find_certificate(energy_primary, method=method, other_consumptions=other_consumptions)

    # performance = pd.concat((performance_3uses, performance_5uses), axis=1, keys=['3uses', '5uses'])
    return performance, energy_primary


def find_certificate(primary_consumption, other_consumptions=None, method='3uses'):
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
    if method == '3uses':
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

    elif method == '5uses':
        carbon_content = pd.Series(CARBON_CONTENT).rename_axis('Heating system')
        emission = primary_consumption * reindex_mi(carbon_content, primary_consumption.index)
        primary_consumption += other_consumptions
        emission += other_consumptions * reindex_mi(carbon_content, primary_consumption.index)

        if isinstance(primary_consumption, pd.Series):
            certificate_energy = pd.Series(dtype=str, index=primary_consumption.index)
            for key, item in CERTIFICATE_5USES_BOUNDARIES_ENERGY.items():
                cond = (primary_consumption > item[0]) & (primary_consumption <= item[1])
                certificate_energy[cond] = key

            certificate_emission = pd.Series(dtype=str, index=primary_consumption.index)
            for key, item in CERTIFICATE_5USES_BOUNDARIES_EMISSION.items():
                cond = (emission > item[0]) & (emission <= item[1])
                certificate_emission[cond] = key

            # maximum between energy and emission
            temp = pd.concat([certificate_energy, certificate_emission], axis=1, keys=['Energy', 'Emission'])
            certificate = temp.max(axis=1)

            return certificate

        elif isinstance(primary_consumption, pd.DataFrame):
            raise NotImplementedError

        else:
            raise NotImplementedError


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


def heat_intensity(budget, method='v4'):
    if method == 'v3':
        return -0.191 * budget.apply(log) + 0.1105
    elif method == 'v4':
        return 0.3564 * budget**(-0.244)


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


def stat_certificate(df):
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
    return primary_heat_consumption, stat_certificate(primary_heat_consumption)



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
