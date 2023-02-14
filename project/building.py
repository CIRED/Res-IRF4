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

import os
from typing import Union, Any

import numpy as np
import pandas as pd
from pandas import Series, DataFrame, MultiIndex, Index, IndexSlice, concat, to_numeric, unique, read_csv
from numpy import exp, log, append, array
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import logging
from copy import deepcopy
from itertools import product


from project.utils import make_plot, reindex_mi, make_plots, calculate_annuities
from project.input.resources import resources_data
import project.thermal as thermal


ACCURACY = 10**-5
EPC2INT = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
VTA = 0.1
NB_MEASURES = {1: [(False, False, False, True), (False, False, True, False), (False, True, False, False),
                   (True, False, False, False)],
               2: [(False, False, True, True), (False, True, False, True), (True, False, False, True),
                   (True, False, True, False),
                   (True, True, False, False), (False, True, True, False)],
               3: [(False, True, True, True), (True, False, True, True), (True, True, False, True),
                   (True, True, True, False)],
               4: [(True, True, True, True)]}
INSULATION = {'Wall': (True, False, False, False), 'Floor': (False, True, False, False),
              'Roof': (False, False, True, False), 'Windows': (False, False, False, True)}
CONSUMPTION_LEVELS = ['Housing type', 'Wall', 'Floor', 'Roof', 'Windows', 'Heating system']

class ThermalBuildings:
    """ThermalBuildings classes.

    Parameters:
    ----------
    stock: Series
        Building stock.
    surface: Series
        Surface by dwelling type.
    ratio_surface: dict
        Losses area of each envelop component
    efficiency: Series
        Heating system efficiency.
    income: Series
        Average income value by income class.
    consumption_ini: Series
    path: str, optional
    year: int, default: 2018
    debug_mode: bool, default: False

    Attributes:
    ----------

    """

    def __init__(self, stock, surface, ratio_surface, efficiency, income, path=None, year=2018,
                 debug_mode=False):

        self._debug_mode = debug_mode

        if isinstance(stock, MultiIndex):
            stock = Series(index=stock, dtype=float)

        self._efficiency = efficiency
        self._ratio_surface = ratio_surface
        self.path = path
        if path is not None:
            self.path_calibration = os.path.join(path, 'calibration')
            if not os.path.isdir(self.path_calibration):
                os.mkdir(self.path_calibration)

        self.coefficient_global, self.coefficient_heater = None, None

        self._surface_yrs = surface
        self._surface = surface.loc[:, year]

        self._income = income
        self._income_owner = self._income.copy()
        self._income_owner.index.rename('Income owner', inplace=True)
        self._income_tenant = self._income.copy()
        self._income_tenant.index.rename('Income tenant', inplace=True)

        self._residual_rate = 0.05
        self._stock_residual = self._residual_rate * stock
        self.stock_mobile = stock - self._stock_residual

        self.first_year = year
        self._year = year

        self.energy_poverty_save = None
        self.heating_intensity_save = None

        self.taxes_list = []

        self.consumption_before_retrofit = None

        # store values to not recalculate standard energy consumption
        self._consumption_store = {
            'consumption': Series(dtype='float'),
            'consumption_3uses': Series(dtype='float'),
            'certificate': Series(dtype='float'),
            'consumption_renovation': Series(dtype='float'),
            'consumption_3uses_renovation': Series(dtype='float'),
            'certificate_renovation': Series(dtype='float'),
        }

        self.stock = stock

    @property
    def year(self):
        return self._year

    @year.setter
    def year(self, year):
        self._year = year
        self._surface = self._surface_yrs.loc[:, year]
        self.energy_poverty_save = None
        self.heating_intensity_save = None
        self.consumption_before_retrofit = None

    @property
    def stock(self):
        return self._stock

    @stock.setter
    def stock(self, stock):
        """Update stock property.


        Automatically calculate consumption standard and certificate.

        Parameters
        ----------
        stock: Series

        Returns
        -------

        """

        self._stock = stock
        stock_mobile = stock - self._stock_residual.reindex(stock.index, fill_value=0)
        self.stock_mobile = stock_mobile[stock_mobile > ACCURACY]
        self.surface = reindex_mi(self._surface, stock.index)

        self.energy = self.to_energy(stock)
        consumption_sd, _, certificate = self.consumption_heating_store(stock.index)
        self.certificate = reindex_mi(certificate, stock.index)

    def simplified_stock(self, energy_level=False):
        """Return simplified stock.

        Parameters
        ----------
        energy_level

        Returns
        -------
        Series
            Simplified stock.
        """

        stock = self.stock.fillna(0)
        certificate = self.certificate.rename('Performance')
        energy = self.energy.rename('Energy')
        stock = concat((stock, certificate, energy), axis=1).set_index(['Performance', 'Energy'], append=True).squeeze()
        if energy_level:
            stock = stock.groupby(
                ['Occupancy status', 'Income owner', 'Income tenant', 'Housing type', 'Heating system',
                 'Energy', 'Performance']).sum()

        else:
            stock = stock.groupby(
                ['Occupancy status', 'Income owner', 'Income tenant', 'Housing type', 'Heating system',
                 'Performance']).sum()

        return stock

    @staticmethod
    def to_energy(df):
        return Series(df.index.get_level_values('Heating system'), index=df.index).str.split('-').str[0].rename('Energy')

    def add_certificate(self, df):
        """Add energy performance certificate to df index.

        Parameters
        ----------
        df

        Returns
        -------

        """
        certificate = self.certificate.rename('Performance')
        lvl = [i for i in certificate.index.names if i in df.index.names]
        certificate = certificate.groupby(lvl).first()

        certificate = reindex_mi(certificate, df.index)
        df = concat((df, certificate), axis=1).set_index('Performance', append=True).squeeze()

        return df

    def add_energy(self, df):
        energy = self.energy.rename('Energy')
        lvl = [i for i in energy.index.names if i in df.index.names]
        energy = energy.groupby(lvl).first()
        energy = reindex_mi(energy, df.index)
        df = concat((df, energy), axis=1).set_index('Energy', append=True).squeeze()
        return df

    def need_heating(self, climate=2006, smooth=False, freq='year', hourly_profile=None, unit='kWh/y'):
        """Calculate heating need of the current building stock.

        Returns
        -------
        pd.DataFrame
        """
        idx = self.stock.index
        wall = Series(idx.get_level_values('Wall'), index=idx)
        floor = Series(idx.get_level_values('Floor'), index=idx)
        roof = Series(idx.get_level_values('Roof'), index=idx)
        windows = Series(idx.get_level_values('Windows'), index=idx)

        heating_need = thermal.conventional_heating_need(wall, floor, roof, windows, self._ratio_surface.copy(),
                                                         th_bridging='Medium', vent_types='Ventilation naturelle',
                                                         infiltration='Medium', climate=climate,
                                                         smooth=smooth, freq=freq, hourly_profile=hourly_profile)

        if unit == 'kWh/y':
            if isinstance(heating_need, (pd.Series, float, int)):
                heating_need = heating_need * self.stock * self.surface
            elif isinstance(heating_need, pd.DataFrame):
                heating_need = (heating_need.T * self.stock * self.surface).T

            return heating_need

    def consumption_heating(self, index=None, freq='year', climate=None, smooth=False, temp_indoor=None, unit='kWh/m2.y',
                            full_output=False, efficiency_hour=False, level_heater='Heating system'):
        """Calculation consumption standard of the current building stock [kWh/m2.a].

        Parameters
        ----------
        index
        freq
        climate
        smooth
        temp_indoor
        unit
        full_output: bool, default False
        efficiency_hour
        level_heater: {'Heating system', 'Heating system final'}

        Returns
        -------

        """

        # TODO: add when index is not None grouped by levels

        if index is None:
            levels = ['Housing type', 'Heating system', 'Wall', 'Floor', 'Roof', 'Windows']
            index = self.stock.groupby(levels).sum().index

        levels_consumption = ['Wall', 'Floor', 'Roof', 'Windows', level_heater, 'Housing type']
        _index = index.to_frame().loc[:, levels_consumption].set_index(levels_consumption).index
        _index = _index[~_index.duplicated()]

        wall = Series(_index.get_level_values('Wall'), index=_index)
        floor = Series(_index.get_level_values('Floor'), index=_index)
        roof = Series(_index.get_level_values('Roof'), index=_index)
        windows = Series(_index.get_level_values('Windows'), index=_index)
        heating_system = Series(_index.get_level_values(level_heater), index=_index).astype('object')
        efficiency = to_numeric(heating_system.replace(self._efficiency))
        consumption = thermal.conventional_heating_final(wall, floor, roof, windows, self._ratio_surface.copy(),
                                                         efficiency, climate=climate, freq=freq, smooth=smooth,
                                                         temp_indoor=temp_indoor, efficiency_hour=efficiency_hour)

        consumption = reindex_mi(consumption, index)

        if full_output is True:
            certificate, consumption_3uses = thermal.conventional_energy_3uses(wall, floor, roof, windows,
                                                                               self._ratio_surface.copy(),
                                                                               efficiency, _index)
            certificate = reindex_mi(certificate, index)
            consumption_3uses = reindex_mi(consumption_3uses, index)

            return consumption, certificate, consumption_3uses
        else:
            return consumption

    def consumption_heating_store(self, index, level_heater='Heating system', unit='kWh/m2.y', full_output=True):
        """Pre-calculate space energy consumption based only on relevant levels.



        Climate should be None. Consumption is stored for later use, so climate cannot change calculation.

        Parameters
        ----------
        index: MultiIndex, Index
            Used to estimate consumption standard.
        level_heater: {'Heating system', 'Heating system final'}, default 'Heating system'
        unit
        full_output: bool, default True

        Returns
        -------
        """
        levels_consumption = ['Wall', 'Floor', 'Roof', 'Windows', level_heater, 'Housing type']
        _index = index.to_frame().loc[:, levels_consumption].set_index(levels_consumption).index
        _index = _index[~_index.duplicated()]

        _index.rename({level_heater: 'Heating system'}, inplace=True)
        # remove index already calculated
        if not self._consumption_store['consumption'].empty:
            temp = self._consumption_store['consumption'].index.intersection(_index)
            idx = _index.drop(temp)
        else:
            idx = _index

        if not idx.empty:

            consumption, certificate, consumption_3uses = self.consumption_heating(index=idx, freq='year', climate=None,
                                                                                   full_output=True)

            self._consumption_store['consumption'] = concat((self._consumption_store['consumption'], consumption))
            self._consumption_store['consumption'].index = MultiIndex.from_tuples(
                self._consumption_store['consumption'].index).set_names(consumption.index.names)
            self._consumption_store['consumption_3uses'] = concat((self._consumption_store['consumption_3uses'], consumption_3uses))
            self._consumption_store['consumption_3uses'].index = MultiIndex.from_tuples(
                self._consumption_store['consumption_3uses'].index).set_names(consumption.index.names)
            self._consumption_store['certificate'] = concat((self._consumption_store['certificate'], certificate))
            self._consumption_store['certificate'].index = MultiIndex.from_tuples(
                self._consumption_store['certificate'].index).set_names(consumption.index.names)

        levels_consumption = [i for i in index.names if i in levels_consumption]

        consumption_sd = self._consumption_store['consumption'].loc[_index]
        consumption_sd.index.rename({'Heating system': level_heater}, inplace=True)
        consumption_sd = consumption_sd.reorder_levels(levels_consumption)

        if full_output:
            consumption_3uses = self._consumption_store['consumption_3uses'].loc[_index]
            consumption_3uses.index.rename({'Heating system': level_heater}, inplace=True)
            consumption_3uses = consumption_3uses.reorder_levels(levels_consumption)
            certificate = self._consumption_store['certificate'].loc[_index]
            certificate.index.rename({'Heating system': level_heater}, inplace=True)
            certificate = certificate.reorder_levels(levels_consumption)

            return consumption_sd, consumption_3uses, certificate
        else:
            return consumption_sd

    def to_heating_intensity(self, index, prices, consumption=None, level_heater='Heating system'):
        """Calculate heating intensity of index based on energy prices.

        Parameters
        ----------
        index: MultiIndex or Index
        prices: Series
        consumption: Series, default None
        level_heater: {'Heating system', 'Heating system final'}

        Returns
        -------
        Series
            Heating intensity
        """
        if consumption is None:
            consumption = reindex_mi(self.consumption_heating_store(index, full_output=False), index) * reindex_mi(self._surface, index)
        energy_bill = AgentBuildings.energy_bill(prices, consumption, level_heater=level_heater)

        if isinstance(energy_bill, Series):
            budget_share = energy_bill / reindex_mi(self._income_tenant, index)
            heating_intensity = thermal.heat_intensity(budget_share)
        elif isinstance(energy_bill, DataFrame):
            budget_share = (energy_bill.T / reindex_mi(self._income_tenant, energy_bill.index)).T
            heating_intensity = thermal.heat_intensity(budget_share)
        else:
            raise NotImplemented

        return heating_intensity

    def consumption_actual(self, prices, consumption=None, store=False):
        """Space heating consumption based on standard space heating consumption and heating intensity (kWh/building.a).


        Space heating consumption is in kWh/building.y
        Equation is based on Allibe (2012).

        Parameters
        ----------
        prices: Series
        consumption: Series or None, optional
            kWh/building.a
        store: bool, default False
            Store calculation in object attributes.

        Returns
        -------
        Series
        """

        if consumption is None:
            index = self.stock.index
            consumption = self.consumption_heating_store(index, full_output=False)
            consumption = reindex_mi(consumption, index) * reindex_mi(self._surface, index)
            # consumption = self.consumption_heat_sd.copy() * self.surface
        else:
            consumption = consumption.copy()
            index = consumption.index

        energy_bill = AgentBuildings.energy_bill(prices, consumption)
        if isinstance(energy_bill, Series):
            budget_share = energy_bill / reindex_mi(self._income_tenant, index)
            heating_intensity = thermal.heat_intensity(budget_share)
            consumption *= heating_intensity
            if store:
                self.heating_intensity_save = heating_intensity
                self.energy_poverty_save = (self.stock[self.stock.index.get_level_values(
                    'Income owner') == ('D1' or 'D2' or 'D3')])[budget_share >= 0.08].sum()
        elif isinstance(energy_bill, DataFrame):
            budget_share = (energy_bill.T / reindex_mi(self._income_tenant, index)).T
            heating_intensity = thermal.heat_intensity(budget_share)
            consumption = heating_intensity * consumption

        return consumption

    def consumption_total(self, prices=None, freq='year', climate=None, smooth=False, temp_indoor=None, unit='TWh/y',
                          standard=False, efficiency_hour=False, existing=False, energy=False):
        """Aggregated final energy consumption (TWh final energy).

        Parameters
        ----------
        prices: Series
            Energy prices.
        freq
        climate
        smooth
        temp_indoor
        unit
        standard: bool, default False
            If yes, consumption standard (conventional). Otherwise, consumption actual.
        efficiency_hour: bool, default False
        existing: bool, default False
            If yes, calculate consumption only for existing buildings.
        energy: bool, default False
            If yes, calculate consumption by energy carriers. Otherwise, aggregated consumption.


        Returns
        -------
        float or Series
        """

        if standard is True:
            if freq == 'year':
                consumption = self.consumption_heating(freq=freq, climate=None)
                consumption = reindex_mi(consumption, self.stock.index) * self.surface * self.stock
                if existing is True:
                    consumption = consumption[consumption.index.get_level_values('Existing')]
                if energy is False:
                    return consumption.sum() / 10 ** 9
                else:
                    energy = self.energy.reindex(consumption.index)
                    return consumption.groupby(energy).sum() / 10**9

        if standard is False:
            if freq == 'year':
                consumption = self.consumption_heating(freq=freq, climate=climate, smooth=smooth,
                                                       temp_indoor=temp_indoor)
                consumption = reindex_mi(consumption, self.stock.index) * self.surface
                if existing is True:
                    consumption = consumption[consumption.index.get_level_values('Existing')]
                consumption = self.consumption_actual(prices, consumption=consumption) * self.stock
                consumption = self.apply_calibration(consumption) / 10**9

                if energy is False:
                    return consumption.sum()
                else:
                    return consumption

            if freq == 'hour':
                temp = self.consumption_heating(freq=freq, climate=climate, smooth=smooth, efficiency_hour=efficiency_hour)
                temp = reindex_mi(temp, self.stock.index)
                t = (temp.T * self.stock * self.surface).T
                # adding heating intensity
                t = (t.T * self.heating_intensity_save).T
                t = self.apply_calibration(t)
                return t

    def apply_calibration(self, consumption, level_heater='Heating system'):
        if self.coefficient_global is None:
            raise AttributeError

        consumption_heater = consumption.groupby('Heating system').sum()
        consumption_heater *= self.coefficient_global

        if isinstance(consumption, Series):
            _consumption_heater = self.coefficient_heater * consumption_heater
            consumption_secondary = (1 - self.coefficient_heater) * consumption_heater
            _consumption_energy = _consumption_heater.groupby(_consumption_heater.index.str.split('-').str[0].rename('Energy')).sum()
            _consumption_energy['Wood fuel'] += consumption_secondary.sum()

            return _consumption_energy

        elif isinstance(consumption, DataFrame):
            _consumption_heater = (consumption_heater.T * self.coefficient_heater).T
            consumption_secondary = (consumption_heater.T * (1 - self.coefficient_heater)).T.sum()
            _consumption_energy = _consumption_heater.groupby(_consumption_heater.index.str.split('-').str[0].rename('Energy')).sum()
            _consumption_energy.loc['Wood fuel', :] += consumption_secondary

            return _consumption_energy
            
    def calibration_consumption(self, prices, consumption_ini, climate=None, temp_indoor=None, store=True):
        """Calculate energy indicators.

        Parameters
        ----------
        prices: Series
        consumption_ini: Series
        climate
        temp_indoor
        store: bool, default True
            If True, store results in object attribute.

        Returns
        -------

        """

        if self.coefficient_global is None:
            consumption = self.consumption_heating(climate=climate, temp_indoor=temp_indoor)
            consumption = reindex_mi(consumption, self.stock.index) * self.surface

            _consumption_actual = self.consumption_actual(prices, consumption=consumption, store=store) * self.stock

            consumption_energy = _consumption_actual.groupby(self.energy).sum()

            # 1. consumption total
            coefficient_global = consumption_ini.sum() * 10**9 / _consumption_actual.sum()

            consumption_energy *= coefficient_global

            # 2. coefficient among energies
            coefficient = consumption_ini * 10**9 / consumption_energy
            coefficient['Wood fuel'] = 1

            # 3. apply coefficient
            _consumption_energy = coefficient * consumption_energy
            _consumption_energy['Wood fuel'] += ((1 - coefficient) * consumption_energy).sum()
            pd.concat((_consumption_energy / 10**9, consumption_ini), axis=1)
            self.coefficient_global = coefficient_global
            idx_heater = _consumption_actual.index.get_level_values('Heating system').unique()
            if 'Electricity-Heat pump air' not in idx_heater:
                idx_heater = idx_heater.insert(0, 'Electricity-Heat pump air')
            energy = idx_heater.str.split('-').str[0].rename('Energy')
            self.coefficient_heater = coefficient.reindex(energy)
            self.coefficient_heater.index = idx_heater
            idx_heat_pump = ['Electricity-Heat pump air', 'Electricity-Heat pump water']
            self.coefficient_heater.loc[idx_heat_pump] = 1

            self.apply_calibration(_consumption_actual)
            validation = dict()
            # stock initial
            temp = concat((self.stock, self.energy), axis=1).groupby(
                ['Housing type', 'Energy']).sum().iloc[:, 0] / 10**3
            temp.index = temp.index.map(lambda x: 'Stock {} {} (Thousands)'.format(x[0], x[1]))
            validation.update(temp)
            temp = self.stock.groupby('Housing type').sum() / 10**3
            temp.index = temp.index.map(lambda x: 'Stock {} (Thousands)'.format(x))
            validation.update(temp)
            validation.update({'Stock (Thousands)': self.stock.sum() / 10**3})

            # surface initial
            temp = concat((self.stock * self.surface, self.energy), axis=1).groupby(
                ['Housing type', 'Energy']).sum().iloc[:, 0] / 10**6
            temp.index = temp.index.map(lambda x: 'Surface {} {} (Million m2)'.format(x[0], x[1]))
            validation.update(temp)
            temp = (self.stock * self.surface).groupby('Housing type').sum() / 10**6
            temp.index = temp.index.map(lambda x: 'Surface {} (Million m2)'.format(x))
            validation.update(temp)
            validation.update({'Surface (Million m2)': (self.stock * self.surface).sum() / 10**6})

            # heating consumption initial
            temp = concat((_consumption_actual, self.energy), axis=1).groupby(
                ['Housing type', 'Energy']).sum().iloc[:, 0] / 10**9
            temp.index = temp.index.map(lambda x: 'Consumption {} {} (TWh)'.format(x[0], x[1]))
            validation.update(temp)
            temp = _consumption_actual.groupby('Housing type').sum() / 10**9
            temp.index = temp.index.map(lambda x: 'Consumption {} (TWh)'.format(x))
            validation.update(temp)
            validation.update({'Consumption (TWh)': _consumption_actual.sum() / 10**9})

            validation.update({'Coefficient calibration global (%)': self.coefficient_global})

            temp = self.coefficient_heater.copy()
            temp.index = temp.index.map(lambda x: 'Coefficient calibration {} (%)'.format(x))
            validation.update(temp)

            temp = consumption_energy / 10**9
            temp.index = temp.index.map(lambda x: 'Consumption {} (TWh)'.format(x))
            validation.update(temp)

            validation = Series(validation)
            if resources_data['data_calibration'] is not None:
                validation = concat((validation, resources_data['data_calibration']), keys=['Calcul', 'Data'], axis=1)
                validation['Error'] = (validation['Calcul'] - validation['Data']) / validation['Data']

            if self.path is not None:
                validation.round(2).to_csv(os.path.join(self.path_calibration, 'validation_stock.csv'))

    @staticmethod
    def energy_bill(prices, consumption, level_heater='Heating system'):
        """Calculate energy bill by dwelling for each stock segment (€/dwelling.a).

        Parameters
        ----------
        prices: Series
            Energy prices for year (€/kWh)
        consumption: Series
            Energy consumption by dwelling (kWh/dwelling.a)
        level_heater
            Heating system level to calculate the bill. Enable to calculate energy bill before or after change of
            heating system.

        Returns
        -------
        pd.Series
            Energy bill by dwelling for each stock segment (€/dwelling)
        """

        index = consumption.index

        heating_system = Series(index.get_level_values(level_heater), index=index)
        energy = heating_system.str.split('-').str[0].rename('Energy')

        prices = prices.rename('Energy').reindex(energy)
        prices.index = index

        if isinstance(consumption, pd.Series):
            # * reindex_mi(self._surface, index)
            return reindex_mi(consumption, index) * prices
        else:
            # * reindex_mi(self._surface, index)
            return (reindex_mi(consumption, index).T * prices).T

    def optimal_temperature(self, prices):
        """Find indoor temperature based on energy prices, housing performance and income level.

        Parameters
        ----------
        prices: Series
            Energy prices.

        Returns
        -------

        """

        def func(temp, consumption, index):
            consumption_temp = self.consumption_heating(temp_indoor=temp)
            consumption_temp = reindex_mi(consumption_temp, index) * self.surface
            return consumption - consumption_temp

        consumption_actual = self.consumption_actual(prices)
        consumption_sd = self.consumption_heating(temp_indoor=None)
        consumption_sd = reindex_mi(consumption_sd, self.stock.index) * self.surface

        temp_optimal = {}
        for i, v in consumption_actual.iteritems():
            temp_optimal.update({i: fsolve(func, 19, args=(consumption_actual.loc[i], i))[0]})
        temp_optimal = pd.Series(temp_optimal)

        temp = concat((consumption_actual, consumption_sd, temp_optimal), axis=1,
                      keys=['actual', 'conventional', 'temp optimal'])

        return temp_optimal

    def store_consumption(self, prices):
        """Store energy consumption.


        Useful to calculate energy saving and rebound effect.

        Parameters
        ----------
        prices
        """
        output = dict()
        temp = self.consumption_total(freq='year', standard=True, existing=True, energy=True)
        output.update({'Consumption standard (TWh)': temp.sum()})
        temp.index = temp.index.map(lambda x: 'Consumption standard {} (TWh)'.format(x))
        output.update(temp)

        temp = self.consumption_total(prices=prices, freq='year', standard=False, climate=None, smooth=False,
                                      existing=True, energy=True)
        output.update({'Consumption (TWh)': temp.sum()})
        temp.index = temp.index.map(lambda x: 'Consumption {} (TWh)'.format(x))
        output.update(temp)
        return output


class AgentBuildings(ThermalBuildings):
    """Class AgentBuildings represents thermal dynamic building stock.

    Parameters:
    ----------
    stock: Series
        Building stock.
    surface: Series
        Surface by dwelling type.
    ratio_surface: dict
        Losses area of each envelop component
    efficiency: Series
        Heating system efficiency.
    income: Series
        Average income value by income class.
    consumption_ini: Series
    preferences: dict
    performance_insulation: dict
    path: str, optional
    year: int, default: 2018
    debug_mode: bool, default: False
    demolition_rate: float, default 0.0
    endogenous: bool, default True
    number_exogenous: float or int, default 300000
    insulation_representative: {'market_share', 'max}
    logger: default
    debug_mode: bool, default False
        Detailed output.
    calib_scale: bool, default True
    full_output: None
    quintiles: bool or None, default None
    financing_cost: bool, default True

    Attributes
    ----------
    """

    def __init__(self, stock, surface, ratio_surface, efficiency, income, preferences,
                 performance_insulation, path=None, year=2018, demolition_rate=0.0,
                 endogenous=True, exogenous=None, insulation_representative='market_share',
                 logger=None, debug_mode=False, calib_scale=True, full_output=None,
                 quintiles=None, financing_cost=True, threshold=None
                 ):
        super().__init__(stock, surface, ratio_surface, efficiency, income, path=path, year=year,
                         debug_mode=debug_mode)

        self._replaced_by = None
        self._only_heater = None


        if logger is None:
            logger = logging.getLogger()
        self.logger = logger

        self.quintiles = quintiles

        if full_output is None:
            full_output = True
        self.full_output = full_output

        self.policies = []

        self.financing_cost = financing_cost

        self._demolition_rate = demolition_rate
        self._demolition_total = (stock * self._demolition_rate).sum()
        self._target_demolition = ['E', 'F', 'G']

        choice_insulation = {'Wall': [False, True], 'Floor': [False, True], 'Roof': [False, True],
                             'Windows': [False, True]}
        names = list(choice_insulation.keys())
        choice_insulation = list(product(*[i for i in choice_insulation.values()]))
        choice_insulation.remove((False, False, False, False))
        choice_insulation = MultiIndex.from_tuples(choice_insulation, names=names)
        self._choice_insulation = choice_insulation
        self._performance_insulation = {i: min(val, self.stock.index.get_level_values(i).min()) for i, val in
                                        performance_insulation.items()}
        self.surface_insulation = self._ratio_surface.copy()

        self._endogenous, self.param_exogenous = endogenous, exogenous

        self.preferences_heater = deepcopy(preferences['heater'])
        self.preferences_insulation = deepcopy(preferences['insulation'])
        self._insulation_representative = insulation_representative
        self._calib_scale = calib_scale
        self.constant_insulation_extensive, self.constant_insulation_intensive, self.constant_heater = None, None, None
        self.scale = 1.0

        self._heater_store = {}
        self._renovation_store = {}
        self._condition_store = None

        self._annuities_store = {'investment_owner': Series(dtype=float),
                                 'rent_tenant': Series(dtype=float),
                                 'stock_annuities': Series(dtype=float),
                                 }

        self.consumption_saving_heater = None
        self.consumption_saving_insulation = None
        self.cost_rebound = None
        self.rebound = None

        self._threshold = threshold
        if threshold is None:
            self._threshold = False
        self.threshold_indicator = None

    @property
    def year(self):
        return self._year

    @year.setter
    def year(self, year):
        self._year = year
        self._surface = self._surface_yrs.loc[:, year]
        self.consumption_before_retrofit = None
        self._condition_store = None

        self._renovation_store['market_share'], self._renovation_store['renovation_rate'] = None, None

        self._replaced_by = None
        self._only_heater = None

        index = self.stock.droplevel('Income tenant').index
        index = index[~index.duplicated()]

        ini = {
            'market_share': None,
            'renovation_rate': None,
            'global_renovation_high_income': 0,
            'global_renovation_low_income': 0,
            'bonus_best': 0,
            'bonus_worst': 0,
            'renovation_with_heater': 0,
            'vta': 0,
            'replacement': DataFrame(0, index=index, columns=self._choice_insulation),
            'cost': DataFrame(0, index=index, columns=self._choice_insulation),
            'cost_financing': DataFrame(0, index=index, columns=self._choice_insulation),
            'subsidies': DataFrame(0, index=index, columns=self._choice_insulation),
            'annuities': DataFrame(0, index=index, columns=self._choice_insulation),
            'consumption_saved': DataFrame(0, index=index, columns=self._choice_insulation),
            'debt': Series(0, index=resources_data['index']['Income owner']),
            'saving': Series(0, index=resources_data['index']['Income owner']),
            'nb_measures': Series(0, index=[1, 2, 3, 4, 5]),
            'subsidies_count': {},
            'subsidies_average': {},
            'subsidies_details': {},
            'certificate_jump_all': {},

        }

        for k, item in ini.items():
            self._renovation_store[k] = item

        ini = {
            'subsidies_details': {},
            'subsidies_count': {},
            'subsidies_average': {}
        }

        for k, item in ini.items():
            self._heater_store[k] = item

    def add_flows(self, flows):
        """Update stock attribute by adding flow series.

        Parameters
        ----------
        flows: Series, list
        """
        flow_total = None
        if isinstance(flows, Series):
            flow_total = flows
        if isinstance(flows, list):
            for flow in flows:
                if flow_total is None:
                    flow_total = flow.copy()
                else:
                    union = flow.index.union(flow_total.index)
                    flow_total = flow.reindex(union, fill_value=0) + flow_total.reindex(union, fill_value=0)

        union = flow_total.index.union(self.stock.index)
        stock = flow_total.reindex(union, fill_value=0) + self.stock.reindex(union, fill_value=0)

        assert (stock >= -ACCURACY).all(), 'Stock Error: Building stock cannot be negative'
        # stock[stock < 0] = 0

        stock = stock[stock > ACCURACY]
        self.stock = stock

    @staticmethod
    def add_level(df, ref, level):
        """Add level to df respecting ref distribution.

        Parameters
        ----------
        df
        ref
        level

        Returns
        -------

        """
        share = (ref.unstack(level).T / ref.unstack(level).sum(axis=1)).T
        temp = concat([df] * share.shape[1], keys=share.columns, names=share.columns.names, axis=1)
        share = reindex_mi(share, temp.columns, axis=1)
        share = reindex_mi(share, temp.index)
        df = (share * temp).stack(level).dropna()
        return df

    def frame_to_flow(self, replaced_by):
        """Transform insulation transition Dataframe to flow.

        Parameters
        ----------
        replaced_by: DataFrame

        Returns
        -------
        replaced_by: Series
            Flow Series.
        """

        replaced_by_sum = replaced_by.sum().sum()

        if 'Income tenant' not in replaced_by.index.names:
            share = (self.stock_mobile.unstack('Income tenant').T / self.stock_mobile.unstack('Income tenant').sum(
                axis=1)).T
            temp = concat([replaced_by] * share.shape[1], keys=share.columns, names=share.columns.names, axis=1)
            share = reindex_mi(share, temp.columns, axis=1)
            share = reindex_mi(share, temp.index)
            replaced_by = (share * temp).stack('Income tenant').dropna()

            assert round(replaced_by.sum().sum(), 0) == round(replaced_by_sum, 0), 'Sum problem'

        replaced_by = replaced_by.droplevel('Heating system').rename_axis(
            index={'Heating system final': 'Heating system'})

        replaced_by.index.set_names(
            {'Wall': 'Wall before', 'Roof': 'Roof before', 'Floor': 'Floor before', 'Windows': 'Windows before'},
            inplace=True)
        replaced_by.columns.set_names(
            {'Wall': 'Wall after', 'Roof': 'Roof after', 'Floor': 'Floor after', 'Windows': 'Windows after'},
            inplace=True)
        replaced_by = replaced_by.stack(replaced_by.columns.names).rename('Data')

        replaced_by = replaced_by[replaced_by > 0]

        replaced_by = replaced_by.reset_index()

        for component in ['Wall', 'Floor', 'Roof', 'Windows']:
            replaced_by[component] = replaced_by['{} before'.format(component)]
            replaced_by.loc[replaced_by['{} after'.format(component)], component] = self._performance_insulation[component]

        replaced_by.drop(
            ['Wall before', 'Wall after', 'Roof before', 'Roof after', 'Floor before', 'Floor after', 'Windows before',
             'Windows after'], axis=1, inplace=True)

        replaced_by = replaced_by.set_index(self.stock.index.names).loc[:, 'Data']
        replaced_by = replaced_by.groupby(replaced_by.index.names).sum()

        assert replaced_by.sum().round(0) == replaced_by_sum.round(0), 'Sum problem'

        return replaced_by

    @staticmethod
    def find_best_option(criteria, dict_df=None, func='max'):
        """Find best option (columns), and returns

        Parameters
        ----------
        criteria: DataFrame
            Find for each index the column based on criteria values.
        dict_df: dict
            Dataframe to return.
        func

        Returns
        -------
        DataFrame
        """

        if dict_df is None:
            dict_df = {}
        dict_ds = {key: Series(dtype=float) for key in dict_df.keys()}
        dict_ds.update({'criteria': Series(dtype=float)})

        columns = None
        if func == 'max':
            columns = criteria.idxmax(axis=1)
        elif func == 'min':
            columns = criteria.idxmin(axis=1)

        for c in columns.unique():
            idx = columns.index[columns == c]
            for key in dict_df.keys():
                dict_ds[key] = concat((dict_ds[key], dict_df[key].loc[idx, c]), axis=0)
            dict_ds['criteria'] = concat((dict_ds['criteria'], criteria.loc[idx, c]), axis=0)
        for key in dict_df.keys():
            dict_ds[key].index = MultiIndex.from_tuples(dict_ds[key].index).set_names(
                dict_df[key].index.names)
        dict_ds['criteria'].index = MultiIndex.from_tuples(dict_ds['criteria'].index).set_names(criteria.index.names)
        dict_ds['columns'] = columns

        return concat(dict_ds, axis=1)

    def prepare_consumption(self, choice_insulation=None, performance_insulation=None, index=None,
                            level_heater='Heating system', full_output=True, store=True, climate=None):
        """Standard energy consumption and energy performance certificate for each insulation for al households.


        Standard energy consumption only depends on building characteristics.

        Returns
        -------
        DataFrame
            Final consumption standard.
        DataFrame
            Primary consumption standard 3 uses.
        DataFrame
            Cerfificate.
        """

        if climate is not None:
            store = False

        if index is None:
            index = self.stock.index

        if not isinstance(choice_insulation, MultiIndex):
            choice_insulation = self._choice_insulation

        if not isinstance(performance_insulation, MultiIndex):
            performance_insulation = self._performance_insulation

        # only selecting useful levels
        _index = index.copy()
        _index = _index.droplevel(
            [i for i in _index.names if i not in ['Housing type', 'Wall', 'Floor', 'Roof', 'Windows'] + [level_heater]])
        _index = _index[~_index.duplicated()]
        _index = _index.rename({level_heater: 'Heating system'})

        _index = _index.reorder_levels(CONSUMPTION_LEVELS)

        # remove idx already calculated
        if not self._consumption_store['consumption_renovation'].empty and store is True:
            temp = self._consumption_store['consumption_renovation'].index.intersection(_index)
            idx = _index.drop(temp)
        else:
            idx = _index

        if not idx.empty:
            s = concat([Series(index=idx, dtype=float)] * len(choice_insulation), axis=1).set_axis(choice_insulation, axis=1)
            # choice_insulation = choice_insulation.drop(no_insulation) # only for
            s.index.rename({'Wall': 'Wall before', 'Floor': 'Floor before', 'Roof': 'Roof before', 'Windows': 'Windows before'}, inplace=True)
            temp = s.fillna(0).stack(s.columns.names)
            temp = temp.reset_index().drop(0, axis=1)
            for i in ['Wall', 'Floor', 'Roof', 'Windows']:
                # keep the info to unstack later
                temp.loc[:, '{} bool'.format(i)] = temp.loc[:, i]
                temp.loc[temp[i], i] = performance_insulation[i]
                temp.loc[temp[i] == False, i] = temp.loc[temp[i] == False, '{} before'.format(i)]
            temp = temp.astype(
                {'Housing type': 'string', 'Wall': 'float', 'Floor': 'float', 'Roof': 'float', 'Windows': 'float',
                 'Heating system': 'string'})
            index = MultiIndex.from_frame(temp)
            # consumption based on insulated components
            if climate is None:
                consumption, consumption_3uses, certificate = self.consumption_heating_store(index, level_heater='Heating system')
            else:
                # TODO: not work - not good index
                consumption = self.consumption_heating(index=index, climate=climate, level_heater='Heating system',
                                                       full_output=False)

            rslt = dict()
            t = {'consumption': consumption}
            if full_output is True:
                t.update({'consumption_3uses': consumption_3uses, 'certificate': certificate})
            for key, temp in t.items():
                temp = reindex_mi(temp, index).droplevel(['Wall', 'Floor', 'Roof', 'Windows']).unstack(
                    ['{} bool'.format(i) for i in ['Wall', 'Floor', 'Roof', 'Windows']])
                temp.index.rename({'Wall before': 'Wall', 'Floor before': 'Floor', 'Roof before': 'Roof', 'Windows before': 'Windows'},
                                  inplace=True)
                temp.columns.rename({'Wall bool': 'Wall', 'Floor bool': 'Floor', 'Roof bool': 'Roof', 'Windows bool': 'Windows'},
                                    inplace=True)
                rslt[key] = temp

            if store is True:
                if self._consumption_store['consumption_renovation'].empty:
                    self._consumption_store['consumption_renovation'] = rslt['consumption']
                    self._consumption_store['consumption_3uses_renovation'] = rslt['consumption_3uses']
                    self._consumption_store['certificate_renovation'] = rslt['certificate']
                else:
                    self._consumption_store['consumption_renovation'] = concat((self._consumption_store['consumption_renovation'], rslt['consumption']))
                    self._consumption_store['consumption_3uses_renovation'] = concat((self._consumption_store['consumption_3uses_renovation'], rslt['consumption_3uses']))
                    self._consumption_store['certificate_renovation'] = concat((self._consumption_store['certificate_renovation'], rslt['certificate']))
            else:
                consumption = rslt['consumption'].loc[_index]
                consumption.index.rename({'Heating system': level_heater}, inplace=True)
                return consumption

        consumption = self._consumption_store['consumption_renovation'].loc[_index]
        consumption.index.rename({'Heating system': level_heater}, inplace=True)
        if full_output is True:
            consumption_3uses = self._consumption_store['consumption_3uses_renovation'].loc[_index]
            consumption_3uses.index.rename({'Heating system': level_heater}, inplace=True)

            certificate = self._consumption_store['certificate_renovation'].loc[_index]
            certificate.index.rename({'Heating system': level_heater}, inplace=True)

            return consumption, consumption_3uses, certificate
        else:
            return consumption

    def calculate_financing(self, cost, subsidies, financing_cost):
        """Apply financing cost.

        Parameters
        ----------
        cost
        subsidies
        financing_cost

        Returns
        -------

        """

        if financing_cost is not None and self.financing_cost:
            to_pay = cost - subsidies
            share_debt = financing_cost['share_debt'][0] + to_pay * financing_cost['share_debt'][1]
            cost_debt = financing_cost['interest_rate'] * share_debt * to_pay * financing_cost['duration']
            cost_saving = (financing_cost['saving_rate'] * reindex_mi(financing_cost['factor_saving_rate'],
                                                                      to_pay.index) * (
                                   (1 - share_debt) * to_pay).T).T * financing_cost['duration']
            cost_financing = cost_debt + cost_saving
            cost_total = cost + cost_financing

            amount_debt = share_debt * to_pay
            amount_saving = (1 - share_debt) * to_pay

            discount = financing_cost['interest_rate'] * share_debt + ((1 - share_debt).T * reindex_mi(financing_cost[
                'factor_saving_rate'] * financing_cost['saving_rate'], share_debt.index)).T
            # discount_factor = (1 - (1 + discount) ** -30) / 30

            return cost_total, cost_financing, amount_debt, amount_saving, discount

        else:
            return cost, None, None, None, None

    def apply_subsidies_heater(self, index, policies_heater, cost_heater):
        """Calculate subsidies for each dwelling and each heating system.

        Parameters
        ----------
        policies_heater: list
        cost_heater: Series
        frame: DataFrame
            Index matches segments and columns heating system.

        Returns
        -------

        """

        subsidies_total = DataFrame(0, index=index, columns=cost_heater.index)
        subsidies_details = {}

        vta = VTA
        p = [p for p in policies_heater if 'reduced_vta' == p.policy]
        if p:
            vta = p[0].value
            sub = cost_heater * (VTA - vta)
            subsidies_details.update({'reduced_vta': concat([sub] * index.shape[0], keys=index, axis=1).T})

        vta_heater = cost_heater * vta
        cost_heater += vta_heater

        sub = None
        for policy in policies_heater:
            if policy.name not in self.policies and policy.policy in ['subsidy_target', 'subsidy_non_cumulative', 'subsidy_ad_valorem', 'subsidies_cap']:
                self.policies += [policy.name]
            if policy.policy == 'subsidy_target':
                sub = policy.value.reindex(cost_heater.index, axis=1).fillna(0)
                sub = reindex_mi(sub, index)
            elif policy.policy == 'subsidy_ad_valorem':

                if isinstance(policy.value, (float, int)):
                    sub = policy.value * cost_heater
                    sub = concat([sub] * index.shape[0], keys=index, axis=1).T

                if isinstance(policy.value, DataFrame):
                    sub = policy.value * cost_heater
                    sub = reindex_mi(sub, index).fillna(0)

                if isinstance(policy.value, Series):
                    if policy.by == 'index':
                        sub = policy.value.to_frame().dot(cost_heater.to_frame().T)
                        sub = reindex_mi(sub, index).fillna(0)
                    elif policy.by == 'columns':
                        sub = (policy.value * cost_heater).fillna(0).reindex(cost_heater.index)
                        sub = concat([sub] * index.shape[0], keys=index, names=index.names, axis=1).T
                    else:
                        raise NotImplemented
                if policy.cap:
                    sub[sub > policy.cap] = sub
            else:
                continue

            subsidies_details[policy.name] = sub
            subsidies_total += subsidies_details[policy.name]

        regulation = [p for p in policies_heater if p.policy == 'regulation']
        if 'inertia_heater' in [p.name for p in regulation]:
            pass
            # apply_regulation('Privately rented', 'Owner-occupied', 'Occupancy status')

        return cost_heater, vta_heater, subsidies_details, subsidies_total

    def endogenous_market_share_heater(self, index, prices, subsidies_total, cost_heater, ms_heater=None):

        def calibration_constant_heater(_utility, _ms_heater):
            """Constant to match the observed market-share.

            Market-share is defined by initial and final heating system.

            Parameters
            ----------
            _utility: DataFrame
            _ms_heater: DataFrame

            Returns
            -------
            DataFrame
            """
            _utility = _utility.loc[:, _ms_heater.columns]

            # removing unnecessary level
            utility_ref = _utility.droplevel(['Occupancy status']).copy()
            utility_ref = utility_ref[~utility_ref.index.duplicated(keep='first')]

            possible = reindex_mi(_ms_heater, utility_ref.index)
            utility_ref[~(possible > 0)] = float('nan')

            stock = self.stock.groupby(utility_ref.index.names).sum() * 1/20

            # initializing constant to 0
            constant = _ms_heater.copy()
            constant[constant > 0] = 0
            market_share_ini, market_share_agg = None, None
            for i in range(50):
                constant.loc[constant['Electricity-Heat pump water'].notna(), 'Electricity-Heat pump water'] = 0
                constant.loc[constant['Electricity-Heat pump water'].isna() & constant['Electricity-Heat pump air'].notna(), 'Electricity-Heat pump air'] = 0

                _utility_constant = reindex_mi(constant.reindex(utility_ref.columns, axis=1), _utility.index)
                _utility = utility_ref + _utility_constant
                _market_share = (exp(_utility).T / exp(_utility).sum(axis=1)).T
                agg = (_market_share.T * stock).T.groupby(['Housing type', 'Heating system']).sum()
                market_share_agg = (agg.T / agg.sum(axis=1)).T
                if i == 0:
                    market_share_ini = market_share_agg.copy()
                constant = constant + log(_ms_heater / market_share_agg)

                _ms_heater = _ms_heater.reindex(market_share_agg.index)

                if (market_share_agg.round(decimals=3) == _ms_heater.round(decimals=3).fillna(0)).all().all():
                    self.logger.debug('Constant heater optim worked')
                    break

            constant.loc[constant['Electricity-Heat pump water'].notna(), 'Electricity-Heat pump water'] = 0
            constant.loc[constant['Electricity-Heat pump water'].isna() & constant['Electricity-Heat pump air'].notna(), 'Electricity-Heat pump air'] = 0

            wtp = - constant / self.preferences_heater['cost']

            details = concat((constant.stack(), market_share_ini.stack(), market_share_agg.stack(), _ms_heater.stack(), agg.stack() / 10**3, wtp.stack()),
                             axis=1, keys=['constant', 'calcul ini', 'calcul', 'observed', 'thousand', 'wtp']).round(decimals=3)
            if self.path is not None:
                details.to_csv(os.path.join(self.path_calibration, 'calibration_constant_heater.csv'))

            return constant

        choice_heater = cost_heater.columns
        choice_heater_idx = Index(choice_heater, name='Heating system final')
        energy = Series(choice_heater).str.split('-').str[0].set_axis(choice_heater_idx)

        temp = pd.Series(0, index=index, dtype='float').to_frame().dot(pd.Series(0, index=choice_heater_idx, dtype='float').to_frame().T)
        index_final = temp.stack().index

        consumption, _, certificate = self.consumption_heating_store(index_final, level_heater='Heating system final')
        consumption = reindex_mi(consumption.unstack('Heating system final'), index)
        prices_re = prices.reindex(energy).set_axis(consumption.columns)
        energy_bill_sd = ((consumption * prices_re).T * reindex_mi(self._surface, index)).T

        consumption_before = self.consumption_heating_store(index, level_heater='Heating system')[0]
        consumption_before = reindex_mi(consumption_before, index) * reindex_mi(self._surface, index)
        energy_bill_before = AgentBuildings.energy_bill(prices, consumption_before)

        bill_saved = - energy_bill_sd.sub(energy_bill_before, axis=0)
        utility_bill_saving = (bill_saved.T * reindex_mi(self.preferences_heater['bill_saved'], bill_saved.index)).T / 1000
        utility_bill_saving = utility_bill_saving.loc[:, choice_heater]

        certificate = reindex_mi(certificate.unstack('Heating system final'), index)
        certificate_before = self.consumption_heating_store(index)[2]
        certificate_before = reindex_mi(certificate_before, index)

        self._heater_store['certificate_jump'] = - certificate.replace(EPC2INT).sub(
            certificate_before.replace(EPC2INT), axis=0)

        utility_subsidies = subsidies_total * self.preferences_heater['subsidy'] / 1000

        cost_heater = cost_heater.reindex(index).reindex(choice_heater, axis=1)
        pref_investment = reindex_mi(self.preferences_heater['cost'], index)
        utility_cost = (pref_investment * cost_heater.T).T / 1000

        utility_inertia = DataFrame(0, index=utility_bill_saving.index, columns=utility_bill_saving.columns)
        for hs in choice_heater:
            utility_inertia.loc[
                utility_inertia.index.get_level_values('Heating system') == hs, hs] = self.preferences_heater['inertia']

        utility = utility_inertia + utility_cost + utility_bill_saving + utility_subsidies

        # removing heat-pump for low-efficient buildings
        utility = self.add_certificate(utility)
        utility.columns.names = ['Heating system final']
        idx = utility[utility.index.get_level_values('Performance').isin(['F', 'G'])].index
        hp_idx = ['Electricity-Heat pump water', 'Electricity-Heat pump air']
        utility.loc[idx, hp_idx] = float('nan')
        utility = utility.droplevel('Performance')

        if (self.constant_heater is None) and (ms_heater is not None):
            self.logger.info('Calibration market-share heating system')
            ms_heater.dropna(how='all', inplace=True)
            self.constant_heater = calibration_constant_heater(utility, ms_heater)
        utility_constant = reindex_mi(self.constant_heater.reindex(utility.columns, axis=1), utility.index)

        utility += utility_constant
        market_share = (exp(utility).T / exp(utility).sum(axis=1)).T

        return market_share

    def exogenous_market_share_heater(self, index, choice_heater_idx):
        """Define exogenous market-share.

        Market-share is defined by _market_share_exogenous attribute.
        Replacement

        Parameters
        ----------
        index: MultiIndex or Index
        choice_heater_idx: Index

        Returns
        -------
        DataFrame
            Market-share by segment and possible heater choice.
        Series
            Probability replacement.
        """
        market_share_exogenous = {'Wood fuel-Standard boiler': 'Wood fuel-Performance boiler',
                                        'Wood fuel-Performance boiler': 'Wood fuel-Performance boiler',
                                        'Oil fuel-Standard boiler': 'Electricity-Heat pump water',
                                        'Oil fuel-Performance boiler': 'Electricity-Heat pump water',
                                        'Natural gas-Standard boiler': 'Natural gas-Performance boiler',
                                        'Natural gas-Performance boiler': 'Natural gas-Performance boiler',
                                        'Electricity-Performance boiler': 'Electricity-Heat pump air',
                                        'Electricity-Heat pump air': 'Electricity-Heat pump air',
                                        'Electricity-Heat pump water': 'Electricity-Heat pump water',
                                        }

        market_share = Series(index=index, dtype=float).to_frame().dot(
            Series(index=choice_heater_idx, dtype=float).to_frame().T)

        for initial, final in market_share_exogenous.items():
            market_share.loc[market_share.index.get_level_values('Heating system') == initial, final] = 1

        temp = pd.Series(0, index=index, dtype='float').to_frame().dot(pd.Series(0, index=choice_heater_idx, dtype='float').to_frame().T)
        index_final = temp.stack().index
        _, _, certificate = self.consumption_heating_store(index_final, level_heater='Heating system final')
        certificate = reindex_mi(certificate.unstack('Heating system final'), index)
        certificate_before = self.consumption_heating_store(index)[2]
        certificate_before = reindex_mi(certificate_before, index)

        self._heater_store['certificate_jump'] = - certificate.replace(EPC2INT).sub(
            certificate_before.replace(EPC2INT), axis=0)

        return market_share

    def store_information_heater(self, cost_heater, subsidies_total, subsidies_details, replacement, vta_heater,
                                 cost_financing, amount_debt, amount_saving, discount):
        """Store information yearly heater replacement.

        Parameters
        ----------
        cost_heater: Series
            Cost of each heating system (€).
        subsidies_total: DataFrame
            Total amount of eligible subsidies by dwelling and heating system (€).
        subsidies_details: dict
            Amount of eligible subsidies by dwelling and heating system (€).
        replacement: Series
            Dwelling updated with a new heating system.
        vta_heater: Series
            VTA tax of each heating system (€).
        """
        # information stored during
        self._heater_store.update(
            {
                'cost_households': cost_heater,
                'subsidies_households': subsidies_total,
                'replacement': replacement,
                'cost': replacement * cost_heater,
                'cost_financing': replacement * cost_financing,
                'vta': (replacement * vta_heater).sum().sum(),
                'subsidies': replacement * subsidies_total,
                'debt': (replacement * amount_debt).sum(axis=1).groupby('Income owner').sum(),
                'saving': (replacement * amount_saving).sum(axis=1).groupby('Income owner').sum(),
                'discount': discount

            }
        )

        for key, item in subsidies_details.items():
            self._heater_store['subsidies_details'][key] = replacement * item

        for key, sub in self._heater_store['subsidies_details'].items():
            mask = sub.copy()
            mask[mask > 0] = 1
            self._heater_store['subsidies_count'].update({key: (replacement.fillna(0) * mask).sum(axis=1).groupby('Housing type').sum()})
            self._heater_store['subsidies_average'].update({key: sub.sum().sum() / replacement.fillna(0).sum().sum()})

    def heater_replacement(self, stock, prices, cost_heater, lifetime_heater, policies_heater, ms_heater=None,
                           step=1, financing_cost=None):
        """Function returns new building stock after heater replacement.

        Parameters
        ----------
        stock: Series
        prices: Series
        cost_heater: Series
        ms_heater: DataFrame, optional
        policies_heater: list
        probability_replacement: float or Series, default 1/17

        Returns
        -------
        Series
        """

        index = stock.index

        list_heater = list(stock.index.get_level_values('Heating system').unique().union(cost_heater.index))

        probability = 1 / lifetime_heater
        probability *= step
        if isinstance(lifetime_heater, (float, int)):
            probability = Series(len(list_heater) * [1 / lifetime_heater], Index(list_heater, name='Heating system'))

        premature_heater = [p for p in policies_heater if p.policy == 'premature_heater']
        for premature in premature_heater:
            temp = [i for i in probability.index if '{}'.format(i.split('-')[0]) in premature.target]
            probability.loc[temp] = premature.value

        cost_heater, vta_heater, subsidies_details, subsidies_total = self.apply_subsidies_heater(index, policies_heater,
                                                                                                  cost_heater.copy())

        restriction_heater = [p for p in policies_heater if p.policy == 'restriction_heater']
        for restriction in restriction_heater:
            temp = [i for i in restriction.value.index if self.year >= restriction.value[i]]
            temp = [i for i in cost_heater.index if '{}'.format(i.split('-')[0]) in temp]
            if temp:
                idx = subsidies_total.index
                if restriction.target is not None:
                    idx = subsidies_total.index.get_level_values('Housing type') == restriction.target
                subsidies_total.loc[idx, temp] = float('nan')

        cost_total, cost_financing, amount_debt, amount_saving, discount = self.calculate_financing(
            cost_heater,
            subsidies_total,
            financing_cost)

        if self._endogenous:
            market_share = self.endogenous_market_share_heater(index, prices, subsidies_total, cost_total,
                                                               ms_heater=ms_heater)
        else:
            market_share = self.exogenous_market_share_heater(index, cost_heater.index)

        assert (market_share.sum(axis=1).round(0) == 1).all(), 'Market-share issue'

        replacement = (market_share.T * stock * reindex_mi(probability, stock.index)).T

        stock_replacement = replacement.stack('Heating system final')
        to_replace = replacement.sum(axis=1)

        stock = stock - to_replace

        # adding heating system final equal to heating system because no switch
        stock = concat((stock, Series(stock.index.get_level_values('Heating system'), index=stock.index,
                                      name='Heating system final')), axis=1).set_index('Heating system final', append=True).squeeze()
        stock = concat((stock.reorder_levels(stock_replacement.index.names), stock_replacement),
                       axis=0, keys=[False, True], names=['Heater replacement'])
        assert round(stock.sum() - self.stock_mobile.xs(True, level='Existing', drop_level=False).sum(), 0) == 0, 'Sum problem'

        if self.full_output:
            self.store_information_heater(cost_heater, subsidies_total, subsidies_details, replacement, vta_heater,
                                          cost_financing, amount_debt, amount_saving, discount)
        else:
            self._heater_store['cost_households'] = cost_heater

        return stock

    def prepare_cost_insulation(self, cost_insulation):
        """Constitute insulation choice set cost. Cost is equal to the sum of each individual cost component.

        Parameters
        ----------
        cost_insulation: Series

        Returns
        -------
        Series
            Multiindex series. Levels are Wall, Floor, Roof and Windows and values are boolean.
        """
        cost = DataFrame(0, index=cost_insulation.index, columns=self._choice_insulation)
        idx = IndexSlice
        cost.loc[:, idx[True, :, :, :]] = (cost.loc[:, idx[True, :, :, :]].T + cost_insulation['Wall']).T
        cost.loc[:, idx[:, True, :, :]] = (cost.loc[:, idx[:, True, :, :]].T + cost_insulation['Floor']).T
        cost.loc[:, idx[:, :, True, :]] = (cost.loc[:, idx[:, :, True, :]].T + cost_insulation['Roof']).T
        cost.loc[:, idx[:, :, :, True]] = (cost.loc[:, idx[:, :, :, True]].T + cost_insulation['Windows']).T
        return cost

    def prepare_subsidy_insulation(self, subsidies_insulation, policy=None):
        """Constitute insulation choice set subsidies. Subsidies are equal to the sum of each individual subsidy.

        Parameters
        ----------
        policy
        subsidies_insulation: DataFrame

        Returns
        -------
        DataFrame
            Multiindex columns. Levels are Wall, Floor, Roof and Windows and values are boolean.
        """

        idx = IndexSlice
        subsidies = {}
        for i in self.surface_insulation.index:
            subsidy = DataFrame(0, index=subsidies_insulation.index, columns=self._choice_insulation)
            subsidy.loc[:, idx[True, :, :, :]] = subsidy.loc[:, idx[True, :, :, :]].add(
                subsidies_insulation['Wall'] * self.surface_insulation.loc[i, 'Wall'], axis=0)
            subsidy.loc[:, idx[:, True, :, :]] = subsidy.loc[:, idx[:, True, :, :]].add(
                subsidies_insulation['Floor'] * self.surface_insulation.loc[i, 'Floor'], axis=0)
            subsidy.loc[:, idx[:, :, True, :]] = subsidy.loc[:, idx[:, :, True, :]].add(
                subsidies_insulation['Roof'] * self.surface_insulation.loc[i, 'Roof'], axis=0)
            subsidy.loc[:, idx[:, :, :, True]] = subsidy.loc[:, idx[:, :, :, True]].add(
                subsidies_insulation['Windows'] * self.surface_insulation.loc[i, 'Windows'], axis=0)
            subsidies[i] = subsidy.copy()
        subsidies = concat(list(subsidies.values()), axis=0, keys=self.surface_insulation.index,
                           names=self.surface_insulation.index.names)

        if policy == 'subsidy_ad_valorem':
            # NotImplemented: ad_valorem with different subsididies rate
            value = [v for v in subsidies_insulation.stack().unique() if v != 0][0]
            subsidies[subsidies > 0] = value

        return subsidies

    def apply_subsidies_insulation(self, index, policies_insulation, cost_insulation, surface, certificate,
                                   certificate_before, certificate_before_heater, energy_saved_3uses, consumption_saved):
        """Calculate subsidies amount for each possible insulation choice.


        Parameters
        ----------
        index
        policies_insulation: list
        cost_insulation: DataFrame
            Cost for each segment and each possible insulation choice (€).
        surface: Series
            Surface / dwelling for each segment (m2/dwelling).
        certificate : DataFrame
            Certificate by segment after insulation replacement for each possible insulation choice.
        certificate_before : Series
            Certificate by segment before insulation (but after heater replacement)
        certificate_before_heater: Series
            Certificate by segment before insulation, and before heater replacement. Useful to calculate the total
            number of upgrade this year.
        energy_saved_3uses: DataFrame
            Primary conventional energy consumption saved by the insulation work (% - kWh PE/m2).
        consumption_saved: kWh

        Returns
        -------
        cost_insulation: DataFrame
        vta_insulation: DataFrame
        tax : float
        subsidies_details: dict
        subsidies_total: DataFrame
        condition: dict
        """

        def defined_condition(_index, _certificate, _certificate_before, _certificate_before_heater,
                              _energy_saved_3uses, _cost_insulation, _consumption_saved, _list_conditions):
            """Define condition to get subsidies or loan.


            Depends on income (index) and energy performance of renovationd defined by certificate jump or
            energy_saved_3uses.

            Parameters
            ----------
            _index: MultiIndex
            _certificate: DataFrame
            _certificate_before: Series
            _certificate_before_heater: Series
            _energy_saved_3uses: DataFrame
            _cost_insulation: DataFrame
            _consumption_saved: DataFrame
            _list_conditions: list

            Returns
            -------
            condition: dict
                Contains boolean DataFrame that established condition to get subsidies.
            """

            def define_zil_target(__certificate, __certificate_before, __energy_saved_3uses):
                """Define target (households and insulation work) that can get zil.


                Zero_interest_loan_old is the target in terms of EPC jump.
                zero_interest_loan_new is the requirement to be eligible to a 'global renovation' program,
                the renovation must reduce of 35% the conventional primary energy need
                and the resulting building must not be of G or F epc level.

                Parameters
                ----------
                __certificate
                __certificate_before
                __energy_saved_3uses

                Returns
                -------
                target_subsidies: pd.DataFrame
                    Each cell, a gesture and a segment, is a boolean which is True if it is targeted by the policy

                """
                energy_saved_min = 0.35

                target_subsidies = {}
                target_0 = __certificate.isin(['D', 'C', 'B', 'A']).astype(int).mul(
                    __certificate_before.isin(['G', 'F', 'E']).astype(int), axis=0).astype(bool)
                target_1 = __certificate.isin(['B', 'A']).astype(int).mul(__certificate_before.isin(['D', 'C']).astype(int),
                                                                   axis=0).astype(bool)
                target_subsidies['zero_interest_loan_old'] = target_0 | target_1

                target_0 = __certificate.isin(['E', 'D', 'C', 'B', 'A']).astype(bool)
                target_1 = __energy_saved_3uses[__energy_saved_3uses >= energy_saved_min].fillna(0).astype(bool)
                target_subsidies['zero_interest_loan_new'] = target_0 & target_1

                return target_subsidies

            _condition = dict()

            out_worst = (~_certificate.isin(['G', 'F'])).T.multiply(_certificate_before.isin(['G', 'F'])).T
            out_worst = reindex_mi(out_worst, _index).fillna(False).astype('float')
            _condition.update({'bonus_worst': out_worst})

            in_best = (_certificate.isin(['A', 'B'])).T.multiply(~_certificate_before.isin(['A', 'B'])).T
            in_best = reindex_mi(in_best, _index).fillna(False).astype('float')
            _condition.update({'bonus_best': in_best})

            if 'best_option' in _list_conditions:
                # selecting best cost efficiency opportunities to reach A or B
                cost_saving = _cost_insulation / _energy_saved_3uses.replace(0, float('nan'))
                cost_saving = reindex_mi(cost_saving, _index)
                _temp = reindex_mi(_certificate, _index)
                best = _temp.isin(['A', 'B']).replace(False, float('nan')) * cost_saving
                # if B is not possible then C or D
                idx = best[best.isna().all(axis=1)].index
                s_best = _temp.loc[idx, :].isin(['C']).replace(False, float('nan')) * cost_saving.loc[idx, :]
                best = best.dropna(how='all')
                best = concat((best, s_best)).astype(float)
                if s_best.isna().all(axis=1).any():
                    print('D best peformance')
                    idx = s_best[s_best.isna().all(axis=1)].index
                    best = best.dropna(how='all')
                    t_best = _temp.loc[idx, :].isin(['D']).replace(False, float('nan')) * cost_saving.loc[idx, :]
                    best = concat((best, t_best)).astype(float)

                _condition.update({'best_option': (best.T == best.min(axis=1)).T})

            if 'best_efficiency' in _list_conditions or 'efficiency_100' in _list_conditions:
                _cost_annualized = calculate_annuities(_cost_insulation)
                cost_saving = reindex_mi(_cost_annualized, _consumption_saved.index) / _consumption_saved
                best_efficiency = (cost_saving.T == cost_saving.min(axis=1)).T
                _condition.update({'best_efficiency': best_efficiency.copy()})
                best_efficiency_fg = (_condition['best_efficiency'].T & _certificate_before.isin(['G', 'F'])).T
                _condition.update({'best_efficiency_fg': best_efficiency_fg})
                _condition.update({'efficiency_100': cost_saving < 0.1})

                fg = cost_saving.copy()
                fg[fg > 0] = False
                fg.loc[reindex_mi(_certificate_before, fg.index).isin(['G', 'F']), :] = True
                _condition.update({'fg': fg})

            minimum_gest_condition, global_condition = 1, 2
            energy_condition = 0.35

            _certificate_jump = - _certificate.replace(EPC2INT).sub(_certificate_before.replace(EPC2INT),
                                                                          axis=0)
            _certificate_jump = reindex_mi(_certificate_jump, _index)
            _certificate_before_heater = reindex_mi(_certificate_before_heater, _index)
            _certificate = reindex_mi(_certificate, _index)
            _certificate_jump_all = - _certificate.replace(EPC2INT).sub(
                _certificate_before_heater.replace(EPC2INT),
                axis=0)
            _condition.update({'certificate_jump_all': _certificate_jump_all})
            _condition.update({'global_renovation': _certificate_jump_all >= global_condition})

            if 'certificate_jump_min' in _list_conditions:
                _condition.update({'certificate_jump_min': _certificate_jump_all >= minimum_gest_condition})

            if 'global_renovation_fg' in _list_conditions:
                _condition.update({'global_renovation_fg': (_condition['global_renovation'].T & _certificate_before.isin(['G', 'F'])).T})

            if 'global_renovation_fge' in _list_conditions:
                _condition.update({'global_renovation_fge': (_condition['global_renovation'].T & _certificate_before.isin(['G', 'F', 'E'])).T})

            low_income_condition = ['D1', 'D2', 'D3', 'D4']
            if self.quintiles:
                low_income_condition = ['C1', 'C2']
            low_income_condition = _index.get_level_values('Income owner').isin(low_income_condition)
            low_income_condition = pd.Series(low_income_condition, index=_index)

            if 'global_renovation_low_income' in _list_conditions:
                _condition.update({'global_renovation_low_income': (low_income_condition & _condition['global_renovation'].T).T})

            if 'global_renovation_high_income' in _list_conditions:
                high_income_condition = ['D5', 'D6', 'D7', 'D8', 'D9', 'D10']
                if self.quintiles:
                    high_income_condition = ['C3', 'C4', 'C5']
                high_income_condition = _index.get_level_values('Income owner').isin(high_income_condition)
                high_income_condition = pd.Series(high_income_condition, index=_index)
                _condition.update({'global_renovation_high_income': (high_income_condition & _condition['global_renovation'].T).T})

            if 'mpr_serenite' in _list_conditions:
                energy_condition = _energy_saved_3uses >= energy_condition
                _condition.update({'mpr_serenite': (reindex_mi(energy_condition, _index).T & low_income_condition).T})

            if 'zero_interest_loan' in _list_conditions:
                _condition.update({'zero_interest_loan': define_zil_target(_certificate, _certificate_before, _energy_saved_3uses)})

            return _condition

        def apply_regulation(idx_target, idx_replace, level):
            """Apply regulation by replacing utility specific constant.


            Example removing the split incentive between landlord and tenant.

            Parameters
            ----------
            idx_target: str
                Index to replace. Example: 'Privately rented'.
            idx_replace: str
                Index replaced by. Example: 'Owner-occupied'.
            level: str
                Level to look for index. Example: 'Occupancy status'

            """
            _temp = self.constant_insulation_extensive.copy()
            _temp = _temp.drop(_temp[_temp.index.get_level_values(level) == idx_target].index)
            t = _temp[_temp.index.get_level_values(level) == idx_replace]
            t.rename(index={idx_replace: idx_target}, inplace=True)
            _temp = pd.concat((_temp, t)).loc[self.constant_insulation_extensive.index]
            self.constant_insulation_extensive = _temp.copy()

        subsidies_total = DataFrame(0, index=index, columns=cost_insulation.columns)
        subsidies_details = {}

        tax = VTA
        p = [p for p in policies_insulation if 'reduced_vta' == p.policy]
        if p:
            tax = p[0].value
            subsidies_details.update({p[0].name: reindex_mi(cost_insulation * (VTA - tax), index)})
            # subsidies_total += subsidies_details['reduced_vta']
        vta_insulation = cost_insulation * tax
        cost_insulation += vta_insulation

        list_conditions = ['global_renovation_low_income', 'global_renovation_high_income']
        list_conditions += [i.target for i in policies_insulation if i.target is not None]

        condition = defined_condition(index, certificate, certificate_before,
                                      certificate_before_heater,
                                      energy_saved_3uses, cost_insulation,
                                      consumption_saved, list_conditions
                                      )

        for policy in policies_insulation:
            if policy.name not in self.policies and policy.policy in ['subsidy_target', 'subsidy_non_cumulative', 'subsidy_ad_valorem', 'subsidies_cap']:
                self.policies += [policy.name]

            if policy.policy == 'subsidy_target':
                temp = (reindex_mi(self.prepare_subsidy_insulation(policy.value),
                                   index).T * surface).T
                subsidies_total += temp
                if policy.name in subsidies_details.keys():
                    subsidies_details[policy.name] = subsidies_details[policy.name] + temp
                else:
                    subsidies_details[policy.name] = temp.copy()

            elif policy.policy == 'subsidy_ad_valorem':

                cost = policy.cost_targeted(reindex_mi(cost_insulation, index), target_subsidies=condition.get(policy.target))

                if isinstance(policy.value, (Series, float, int)):
                    temp = reindex_mi(policy.value, cost.index)
                    subsidies_details[policy.name] = (temp * cost.T).T
                    subsidies_total += subsidies_details[policy.name]
                else:
                    temp = self.prepare_subsidy_insulation(policy.value, policy=policy.policy)
                    temp = reindex_mi(temp, cost.index)
                    subsidies_details[policy.name] = temp * cost
                    subsidies_total += subsidies_details[policy.name]

        subsidies_bonus = [p for p in policies_insulation if p.policy == 'bonus']
        for policy in subsidies_bonus:
            temp = (reindex_mi(policy.value, condition[policy.target].index) * condition[policy.target].T).T
            subsidies_total += temp
            if policy.name in subsidies_details.keys():
                subsidies_details[policy.name] = subsidies_details[policy.name] + temp
            else:
                subsidies_details[policy.name] = temp.copy()

        subsidies_non_cumulative = [p for p in policies_insulation if p.policy == 'subsidy_non_cumulative']
        for policy in subsidies_non_cumulative:
            sub = (reindex_mi(policy.value, condition[policy.target].index) * condition[policy.target].T).T
            sub = sub.astype(float)
            for name in policy.non_cumulative:
                # TODO: could be a bug here when policy.non_cumulative are all there and when non cumulative alone
                if name in subsidies_details.keys():
                    subsidies_total -= subsidies_details[name]
                    comp = reindex_mi(subsidies_details[name], sub.index)
                    temp = comp.where(comp > sub, 0)
                    subsidies_details[name] = temp.copy()
                    temp = sub.where(sub > comp, 0)
                    subsidies_details[policy.name] = temp.copy()
                    subsidies_total += subsidies_details[name] + subsidies_details[policy.name]

        subsidies_cap = [p for p in policies_insulation if p.policy == 'subsidies_cap']
        subsidies_uncaped = subsidies_total.copy()

        zil = [p for p in policies_insulation if p.policy == 'subsidy_ad_valorem' and p.name == 'zero_interest_loan']
        if 'zero_interest_loan' in subsidies_details.keys() and zil is []:
            subsidies_uncaped -= subsidies_details['zero_interest_loan']

        if subsidies_cap:
            subsidies_cap = subsidies_cap[0]
            subsidies_cap = reindex_mi(subsidies_cap.value, subsidies_uncaped.index)
            cap = (reindex_mi(cost_insulation, index).T * subsidies_cap).T
            over_cap = subsidies_uncaped > cap
            subsidies_details['over_cap'] = (subsidies_uncaped - cap)[over_cap].fillna(0)
            remaining = subsidies_details['over_cap'].copy()
            if 'mpr_serenite' in subsidies_details.keys():
                temp = subsidies_details['over_cap'].where(
                    subsidies_details['over_cap'] <= subsidies_details['mpr_serenite'],
                    subsidies_details['mpr_serenite'])
                subsidies_details['mpr_serenite'] -= temp
                remaining = subsidies_details['over_cap'] - temp
                assert (subsidies_details['mpr_serenite'].values >= 0).all(), 'MPR Serenite got negative values'
            if 'mpr' in subsidies_details.keys() and not (remaining > 0).any().any():
                subsidies_details['mpr'] -= remaining
                assert (subsidies_details['mpr'].values >= 0).all(), 'MPR got negative values'

            subsidies_total -= subsidies_details['over_cap']

        regulation = [p for p in policies_insulation if p.policy == 'regulation']
        if 'landlord' in [p.name for p in regulation]:
            apply_regulation('Privately rented', 'Owner-occupied', 'Occupancy status')
        if 'multi_family' in [p.name for p in regulation]:
            apply_regulation('Multi-family', 'Single-family', 'Housing type')

        return cost_insulation, vta_insulation, tax, subsidies_details, subsidies_total, condition

    def endogenous_retrofit(self, stock, prices, subsidies_total, cost_insulation,
                            calib_renovation=None, calib_intensive=None, min_performance=None,
                            subsidies_details=None):
        """Calculate endogenous retrofit based on discrete choice model.


        Utility variables are investment cost, energy bill saving, and subsidies.
        Preferences are object attributes defined initially.

        # bill saved calculated based on the new heating system
        # certificate before work and so subsidies before the new heating system

        Parameters
        ----------
        financing_cost
        prices: Series
        subsidies_total: DataFrame
        cost_insulation: DataFrame
        stock: Series, default None
        calib_intensive: dict, optional
        calib_renovation: dict, optional
        min_performance: str, optional
        subsidies_details: dict, optional

        Returns
        -------
        Series
            Retrofit rate
        DataFrame
            Market-share insulation
        """

        def market_share_func(u):
            return (exp(u).T / exp(u).sum(axis=1)).T

        def to_market_share(_bill_saved, _subsidies_total, _cost_total):
            """Calculate market-share between insulation options.


            Parameters
            ----------
            _bill_saved: DataFrame
            _subsidies_total: DataFrame
            _cost_total: DataFrame

            Returns
            -------
            ms_intensive: DataFrame
            util_intensive: DataFrame
            """

            _bill_saved[_bill_saved == 0] = float('nan')
            _subsidies_total[_bill_saved == 0] = float('nan')
            _cost_total[_bill_saved == 0] = float('nan')

            pref_sub = reindex_mi(self.preferences_insulation['subsidy'], _subsidies_total.index).rename(None)
            utility_subsidies = (_subsidies_total.T * pref_sub).T / 1000

            pref_investment = reindex_mi(self.preferences_insulation['cost'], _cost_total.index).rename(None)
            utility_investment = (_cost_total.T * pref_investment).T / 1000

            utility_bill_saving = (_bill_saved.T * reindex_mi(self.preferences_insulation['bill_saved'], _bill_saved.index)).T / 1000

            util_intensive = utility_bill_saving + utility_investment + utility_subsidies

            if self.constant_insulation_intensive is not None:
                util_intensive += self.constant_insulation_intensive

            ms_intensive = market_share_func(util_intensive)
            return ms_intensive, util_intensive

        def to_utility_extensive(_cost_total, _bill_saved, _subsidies_total, _market_share, _utility_intensive=None):
            # extensive margin
            _bill_saved_insulation, _subsidies_insulation, _investment_insulation = None, None, None
            if self._insulation_representative == 'market_share':
                _investment_insulation = (_cost_total.reindex(_market_share.index) * _market_share).sum(axis=1)
                _bill_saved_insulation = (_bill_saved.reindex(_market_share.index) * _market_share).sum(axis=1)
                _subsidies_insulation = (_subsidies_total.reindex(_market_share.index) * _market_share).sum(axis=1)

            elif self._insulation_representative == 'max':
                _utility_intensive = _utility_intensive.dropna(how='all')
                dict_df = {'investment': _cost_total, 'bill_saved': _bill_saved, 'subsidies': _subsidies_total}
                dict_int = self.find_best_option(_utility_intensive, dict_df, func='max')

                def rename_tuple(tuple, names):
                    idx = tuple.index
                    tuple = DataFrame([[a, b, c, d] for a, b, c, d in tuple.values])
                    tuple.columns = names
                    for i in names:
                        tuple.loc[tuple[i] == True, i] = i
                        tuple.loc[tuple[i] == False, i] = ''
                    return Series(list(zip(*(tuple[i] for i in names))), index=idx)

                dict_int['representative'] = rename_tuple(dict_int['columns'], _utility_intensive.columns.names)
                _bill_saved_insulation = dict_int['bill_saved']
                _subsidies_insulation = dict_int['subsidies']
                _investment_insulation = dict_int['investment']

            # bill saved == 0 should have been removed in market_share calculation
            idx = _bill_saved_insulation[_bill_saved_insulation <= 0].index
            _bill_saved_insulation.drop(idx, inplace=True)
            _subsidies_insulation.drop(idx, inplace=True)
            _investment_insulation.drop(idx, inplace=True)
            return _investment_insulation, _bill_saved_insulation, _subsidies_insulation

        def retrofit_func(u):
            return 1 / (1 + exp(- u))

        def to_retrofit_rate(_bill_saved, _subsidies_total, _cost_total):
            """Calculate retrofit rate based on binomial logit model.

            Parameters
            ----------
            _bill_saved
            _subsidies_total
            _cost_total

            Returns
            -------
            retrofit_proba: pd.Series
                Retrofit rate for each household.
            utility: pd.Series
                Utility to renovate for each household.
            """

            utility_bill_saving = reindex_mi(self.preferences_insulation['bill_saved'], _bill_saved.index) * _bill_saved / 1000

            pref_sub = reindex_mi(self.preferences_insulation['subsidy'], _subsidies_total.index).rename(None)
            utility_subsidies = (pref_sub * _subsidies_total) / 1000

            pref_investment = reindex_mi(self.preferences_insulation['cost'], _cost_total.index).rename(None)
            utility_investment = (pref_investment * _cost_total) / 1000

            utility_renovate = utility_investment + utility_bill_saving + utility_subsidies

            if self.constant_insulation_extensive is not None:
                _utility = self.add_certificate(utility_renovate.copy())
                utility_constant = reindex_mi(self.constant_insulation_extensive, _utility.index)
                _utility += utility_constant
                utility_renovate = _utility.droplevel('Performance')

            retrofit_proba = retrofit_func(utility_renovate)

            return retrofit_proba, utility_renovate

        def apply_endogenous_retrofit(_bill_saved, _subsidies_total, _cost_total):

            _market_share, _utility_intensive = to_market_share(_bill_saved, _subsidies_total, _cost_total)

            _cost, _bill, _subsidies = to_utility_extensive(_cost_total, _bill_saved, _subsidies_total, _market_share,
                                                            _utility_intensive)

            _renovation_rate, _utility = to_retrofit_rate(_bill, _subsidies, _cost)

            return _market_share, _renovation_rate

        def apply_rational_choice(_consumption_saved, _subsidies_total, _cost_total):
            _consumption_saved.replace(0, float('nan'), inplace=True)
            # subsidies do not change market-share
            _cost_efficiency = _cost_total / _consumption_saved
            dict_df = {'consumption': _consumption_saved, 'subsidies': _subsidies_total, 'cost': _cost_total}
            best_cost_efficiency = AgentBuildings.find_best_option(_cost_efficiency, dict_df=dict_df, func='min')
            _market_share = DataFrame(0, index=_consumption_saved.index, columns=_consumption_saved.columns)
            for i in _market_share.index:
                _market_share.loc[i, best_cost_efficiency.loc[i, 'columns']] = 1

            # assert (_market_share.sum(axis=1) == 1).all(), 'Market-share problem'

            _consumption_saved, _subsidies_total, _cost_total = best_cost_efficiency['consumption'], \
            best_cost_efficiency['subsidies'], best_cost_efficiency['cost']

            indicator = (_cost_total - _subsidies_total) / _consumption_saved
            if self.threshold_indicator is None:
                self.threshold_indicator = indicator.min()

            _renovation_rate = indicator < self.threshold_indicator
            _renovation_rate = _renovation_rate.astype(float)

            return _market_share, _renovation_rate

        def to_freeriders(_scale, _utility, _stock, delta_sub, pref_sub):
            """Calculate freeriders due to implementation of subsidy.


            Parameters
            ----------
            _scale: float
            _utility: Series
            _stock: Series
            delta_sub: Series
            pref_sub: Series

            Returns
            -------
            float
            """

            retrofit = retrofit_func(_utility * _scale)
            flow = (retrofit * _stock).sum()

            utility_plus = (_utility + pref_sub * delta_sub).dropna() * _scale
            retrofit_plus = retrofit_func(utility_plus)
            flow_plus = (retrofit_plus * _stock).sum()

            return min(flow, flow_plus) / max(flow, flow_plus)

        def calibration_intensive(util, _stock, market_share_ini, _renovation_rate, iteration=1000):
            """Calibrate alternative-specific constant to match observed market-share.


            Parameters
            ----------
            _stock: Series
            util: DataFrame
            market_share_ini: Series
                Observed market-share.
            _renovation_rate: Series
                Renovation rate.
            iteration: optional, int, default 100

            Returns
            -------
            Series
            """

            if 'Performance' in _renovation_rate.index.names:
                _stock = self.add_certificate(_stock)

            f_retrofit = _stock * reindex_mi(_renovation_rate, _stock.index)
            utility_ref = reindex_mi(util, f_retrofit.index).dropna(how='all')

            const = market_share_ini.reindex(utility_ref.columns, axis=0).copy()
            const[const > 0] = 0

            # insulation of the roof is the most frequent insulation work
            insulation_ref = (False, False, True, False)
            ms_segment, ms_agg, ms_ini = None, None, None
            for i in range(iteration):
                const.loc[insulation_ref] = 0
                _utility = (utility_ref + const).copy()
                ms_segment = market_share_func(_utility)
                f_replace = (ms_segment.T * f_retrofit).T
                ms_agg = (f_replace.sum() / f_replace.sum().sum()).reindex(market_share_ini.index)
                if i == 0:
                    ms_ini = ms_agg.copy()
                const = const + log(market_share_ini / ms_agg)

                if (ms_agg.round(decimals=2) == market_share_ini.round(decimals=2)).all():
                    self.logger.debug('Constant intensive optim worked')
                    break
            const.loc[insulation_ref] = 0
            _utility = (utility_ref + const).copy()
            ms_segment = market_share_func(_utility)
            f_replace = (ms_segment.T * f_retrofit).T
            ms_agg = (f_replace.sum() / f_replace.sum().sum()).reindex(market_share_ini.index)

            nb_renovation = (_stock * reindex_mi(_renovation_rate, _stock.index)).sum()
            wtp = const / self.preferences_insulation['cost']
            details = concat((const, ms_ini, ms_agg, market_share_ini, (market_share_ini * nb_renovation) / 10 ** 3, wtp), axis=1,
                             keys=['constant', 'calcul ini', 'calcul', 'observed', 'thousand', 'wtp']).round(decimals=3)
            if self.path is not None:
                details.to_csv(os.path.join(self.path_calibration, 'calibration_constant_insulation.csv'))

            return const

        def calibration_extensive(_utility, _stock, _calib_renovation):
            """Simultaneously calibrate constant and scale to match freeriders and retrofit rate.

            Parameters
            ----------
            _utility: Series
            _stock: Series
            _calib_renovation: dict

            Returns
            -------

            """

            def solve_feeriders(x, utility_ini, stock_ini, retrofit_rate_target, freeride, delta_sub, pref_sub):
                scale = x[-1]
                cst = x[:-1]

                # calibration constant
                cst = Series(cst, index=retrofit_rate_target.index)
                utility_ref = utility_ini.copy()
                stock_ref = stock_ini.copy()
                utility_cst = reindex_mi(cst, utility_ref.index)
                u = (utility_ref + utility_cst).copy()
                retrofit_rate_calc = retrofit_func(u * scale)
                agg = (retrofit_rate_calc * stock_ref).groupby(retrofit_rate_target.index.names).sum()
                retrofit_rate_agg = agg / stock_ref.groupby(retrofit_rate_target.index.names).sum()
                rslt = retrofit_rate_agg - retrofit_rate_target

                # calibration scale
                calcul = to_freeriders(scale, u, stock_ini, delta_sub, pref_sub)
                rslt = append(rslt, calcul - freeride)

                return rslt

            def solve_deviation(x, utility_ini, stock_ini, retrofit_rate_target, std_deviation):
                scale = x[-1]
                cst = x[:-1]

                # calibration constant
                cst = Series(cst, index=retrofit_rate_target.index)
                utility_ref = utility_ini.copy()
                stock_ref = stock_ini.copy()
                utility_cst = reindex_mi(cst, utility_ref.index)
                u = (utility_ref + utility_cst).copy()
                retrofit_rate_calc = retrofit_func(u * scale)
                agg = (retrofit_rate_calc * stock_ref).groupby(retrofit_rate_target.index.names).sum()
                retrofit_rate_agg = agg / stock_ref.groupby(retrofit_rate_target.index.names).sum()
                rslt = retrofit_rate_agg - retrofit_rate_target

                # calibration scale
                retrofit_mean = (retrofit_rate_calc * stock_ini).sum() / stock_ini.sum()
                std_deviation_calc = (((retrofit_rate_calc - retrofit_mean)**2 * stock_ini).sum() / (stock_ini.sum() - 1))**(1/2)
                rslt = append(rslt, std_deviation - std_deviation_calc)

                return rslt

            def solve_noscale(x, utility_ini, stock_ini, retrofit_rate_target):

                # calibration constant
                cst = Series(x, index=retrofit_rate_target.index)
                utility_ref = utility_ini.copy()
                stock_ref = stock_ini.copy()
                utility_cst = reindex_mi(cst, utility_ref.index)
                u = (utility_ref + utility_cst).copy()
                retrofit_rate_calc = retrofit_func(u)
                agg = (retrofit_rate_calc * stock_ref).groupby(retrofit_rate_target.index.names).sum()
                retrofit_rate_agg = agg / stock_ref.groupby(retrofit_rate_target.index.names).sum()
                rslt = retrofit_rate_agg - retrofit_rate_target

                return rslt

            retrofit_rate_ini = _calib_renovation['renovation_rate_ini']
            calibration_scale = _calib_renovation['scale']

            if 'Performance' in retrofit_rate_ini.index.names:
                _stock = self.add_certificate(_stock)
                stock_retrofit = _stock[_stock.index.get_level_values('Performance') > 'B']
                _utility = self.add_certificate(_utility)
            else:
                stock_retrofit = _stock

            _constant = retrofit_rate_ini.copy()
            _constant[retrofit_rate_ini > 0] = 0

            _scale = 1.0
            if calibration_scale is not None:
                x = append(_constant.to_numpy(), 1)

                if calibration_scale['name'] == 'freeriders':
                    root, info_dict, ier, mess = fsolve(solve_feeriders, x, args=(
                        _utility, stock_retrofit, retrofit_rate_ini, calibration_scale['target_freeriders'],
                        - calibration_scale['delta_subsidies_sum'] / 1000, calibration_scale['pref_subsidies']),
                                                        full_output=True)
                    _scale = root[-1]
                    _constant = Series(root[:-1], index=retrofit_rate_ini.index) * _scale

                elif calibration_scale['name'] == 'standard_deviation':
                    root, info_dict, ier, mess = fsolve(solve_deviation, x, args=(
                        _utility, stock_retrofit, retrofit_rate_ini, calibration_scale['deviation']), full_output=True)
                    _scale = root[-1]
                    _constant = Series(root[:-1], index=retrofit_rate_ini.index) * _scale

                else:
                    x = _constant.to_numpy()
                    root, info_dict, ier, mess = fsolve(solve_noscale, x,
                                                        args=(_utility, stock_retrofit, retrofit_rate_ini),
                                                        full_output=True)
                    _constant = Series(root, index=retrofit_rate_ini.index) * _scale

                self.logger.info(mess)
                self.logger.info('Scale: {}'.format(_scale))

            # without calibration
            retrofit_ini = retrofit_func(_utility)
            agg_ini = (retrofit_ini * _stock).groupby(retrofit_rate_ini.index.names).sum()
            retrofit_rate_agg_ini = agg_ini / _stock.groupby(retrofit_rate_ini.index.names).sum()

            utility_constant = reindex_mi(_constant, _utility.index)
            _utility = _utility * _scale + utility_constant
            retrofit_rate = retrofit_func(_utility)
            agg = (retrofit_rate * _stock).groupby(retrofit_rate_ini.index.names).sum()
            retrofit_rate_agg = agg / _stock.groupby(retrofit_rate_ini.index.names).sum()

            retrofit_rate_cst = retrofit_func(utility_constant)
            agg_cst = (retrofit_rate_cst * _stock).groupby(retrofit_rate_ini.index.names).sum()
            retrofit_rate_agg_cst = agg_cst / _stock.groupby(retrofit_rate_ini.index.names).sum()

            coefficient_cost = abs(self.preferences_insulation['cost'] * _scale)
            wtp = _constant / coefficient_cost
            ref = ('Single-family', 'Owner-occupied', False)
            diff = wtp - wtp[ref]

            details = concat((
                             _constant, retrofit_rate_agg_ini, retrofit_rate_agg, retrofit_rate_ini, agg / 10 ** 3, wtp,
                             diff, retrofit_rate_agg_cst), axis=1,
                             keys=['constant', 'ini', 'calcul', 'observed', 'thousand', 'wtp', 'market_barriers',
                                   'cste']).round(decimals=3)
            if self.path is not None:
                details.to_csv(os.path.join(self.path_calibration, 'calibration_constant_extensive.csv'))

            return _constant, _scale

        def calibration_coupled(_stock, _cost_total, _bill_saved, _subsidies_total, _calib_renovation,
                                _calib_intensive):
            self.logger.info('Calibration intensive and renovation rate')

            renovation_rate_ini = _calib_renovation['renovation_rate_ini']
            ms_insulation_ini = _calib_intensive['ms_insulation_ini']

            # initialization of first renovation rate and then market-share
            # better conversion when
            _market_share = concat([ms_insulation_ini] * _cost_total.shape[0], axis=1).T
            _market_share.index = _cost_total.index
            investment_insulation, bill_saved_insulation, subsidies_insulation = to_utility_extensive(_cost_total,
                                                                                                      _bill_saved,
                                                                                                      _subsidies_total,
                                                                                                      _market_share)

            compare = None
            for k in range(10):
                # calibration of renovation rate
                self.constant_insulation_extensive = None
                _, utility = to_retrofit_rate(bill_saved_insulation, subsidies_insulation, investment_insulation)
                constant, scale = calibration_extensive(utility, _stock, _calib_renovation)
                self.constant_insulation_extensive = constant
                self.apply_scale(scale)
                _renovation_rate, _ = to_retrofit_rate(bill_saved_insulation, subsidies_insulation,
                                                       investment_insulation)

                # recalibration of market_share with new scale and renovation rate
                self.constant_insulation_intensive = None
                _, utility_intensive = to_market_share(_bill_saved, _subsidies_total, _cost_total)
                self.constant_insulation_intensive = calibration_intensive(utility_intensive, _stock, ms_insulation_ini,
                                                                           _renovation_rate)
                # test market-share (test with other renovation rate so can differ)
                _market_share, utility_intensive = to_market_share(_bill_saved, _subsidies_total, _cost_total)

                # test renovation_rate (test with other utility extensive so can differ)
                investment_insulation, bill_saved_insulation, subsidies_insulation = to_utility_extensive(_cost_total,
                                                                                                          _bill_saved,
                                                                                                          _subsidies_total,
                                                                                                          _market_share,
                                                                                                          utility_intensive)

                _renovation_rate, _ = to_retrofit_rate(bill_saved_insulation, subsidies_insulation,
                                                       investment_insulation)

                flow = _renovation_rate * _stock
                rate = flow.groupby(renovation_rate_ini.index.names).sum() / _stock.groupby(
                    renovation_rate_ini.index.names).sum()
                compare_rate = concat((rate.rename('Calculated'), renovation_rate_ini.rename('Observed')),
                                      axis=1).round(3)
                flow_insulation = (flow * _market_share.T).T.sum()
                share = flow_insulation / flow_insulation.sum()
                compare_share = concat((share.rename('Calculated'), ms_insulation_ini.rename('Observed')),
                                       axis=1).round(3)

                compare = concat((compare_rate, compare_share), ignore_index=True)
                if (compare['Calculated'] == compare['Observed']).all():
                    self.logger.debug('Coupled optim worked')
                    break

            if self.path is not None:
                compare.to_csv(os.path.join(self.path_calibration, 'result_calibration.csv'))

        def assess_sensitivity(_stock, _cost_total, _bill_saved, _subsidies_total, path):

            # bill saved
            result_bill_saved = dict()
            result_bill_saved['Average cost (Thousand euro)'] = dict()
            result_bill_saved['Flow renovation (Thousand)'] = dict()

            values = [-0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5]
            for v in values:
                v += 1
                _market_share, _renovation_rate = apply_endogenous_retrofit(_bill_saved * v, _subsidies_total, _cost_total)
                f_renovation = _renovation_rate * _stock
                f_replace = (f_renovation * _market_share.T).T
                result_bill_saved['Average cost (Thousand euro)'].update({v: (f_replace * _cost_total).sum().sum() / f_renovation.sum() / 10**3})
                result_bill_saved['Flow renovation (Thousand)'].update({v: f_renovation.sum() / 10**3})
            result_bill_saved['Average cost (Thousand euro)'] = pd.Series(result_bill_saved['Average cost (Thousand euro)'])
            result_bill_saved['Flow renovation (Thousand)'] = pd.Series(result_bill_saved['Flow renovation (Thousand)'])

            # subsidies
            result_subsidies = dict()
            result_subsidies['Average cost (Thousand euro)'] = dict()
            result_subsidies['Flow renovation (Thousand)'] = dict()

            values = [-0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5]
            for v in values:
                v += 1
                _market_share, _renovation_rate = apply_endogenous_retrofit(_bill_saved, _subsidies_total * v, _cost_total)
                f_renovation = _renovation_rate * _stock
                f_replace = (f_renovation * _market_share.T).T
                result_subsidies['Average cost (Thousand euro)'].update({v: (f_replace * _cost_total).sum().sum() / f_renovation.sum() / 10**3})
                result_subsidies['Flow renovation (Thousand)'].update({v: f_renovation.sum() / 10**3})
            result_subsidies['Average cost (Thousand euro)'] = pd.Series(result_subsidies['Average cost (Thousand euro)'])
            result_subsidies['Flow renovation (Thousand)'] = pd.Series(result_subsidies['Flow renovation (Thousand)'])

            # subsidies
            result_cost = dict()
            result_cost['Average cost (Thousand euro)'] = dict()
            result_cost['Flow renovation (Thousand)'] = dict()

            values = [-0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5]
            for v in values:
                v += 1
                _market_share, _renovation_rate = apply_endogenous_retrofit(_bill_saved, _subsidies_total, _cost_total * v)
                f_renovation = _renovation_rate * _stock
                f_replace = (f_renovation * _market_share.T).T
                result_cost['Average cost (Thousand euro)'].update({v: (f_replace * _cost_total).sum().sum() / f_renovation.sum() / 10**3})
                result_cost['Flow renovation (Thousand)'].update({v: f_renovation.sum() / 10**3})
            result_cost['Average cost (Thousand euro)'] = pd.Series(result_cost['Average cost (Thousand euro)'])
            result_cost['Flow renovation (Thousand)'] = pd.Series(result_cost['Flow renovation (Thousand)'])

            # result
            dict_df = {'Bill saving': result_bill_saved['Average cost (Thousand euro)'],
                       'Cost': result_cost['Average cost (Thousand euro)'],
                       'Subsidies': result_subsidies['Average cost (Thousand euro)']
                       }
            make_plots(dict_df, 'Average cost (Thousand euro)', save=os.path.join(path, 'sensi_average_cost.png'))
            dict_df = {'Bill saving': result_bill_saved['Flow renovation (Thousand)'],
                       'Cost': result_cost['Flow renovation (Thousand)'],
                       'Subsidies': result_subsidies['Flow renovation (Thousand)']
                       }
            make_plots(dict_df, 'Flow renovation (Thousand)', save=os.path.join(path, 'sensi_flow_renovation.png'))

        def assess_policies(_stock, _subsidies_details, _cost_total, _bill_saved, _subsidies_total):
            """Test freeriders and intensive margin of subsidies

            Parameters
            ----------
            _stock
            _subsidies_details
            _bill_saved
            _subsidies_total
            _cost_total

            Returns
            -------

            """

            _market_share, _renovation_rate = apply_endogenous_retrofit(_bill_saved, _subsidies_total, _cost_total)

            f_retrofit = _renovation_rate * _stock
            f_replace = (f_retrofit * _market_share.T).T
            avg_cost_global = (f_replace * _cost_total).sum().sum() / f_retrofit.sum()

            _subsidies_details = {k: item for k, item in _subsidies_details.items() if k != 'over_cap'}
            rslt = dict()
            for key, _sub in _subsidies_details.items():
                _c = 0
                mask = _sub > 0
                if key == 'reduced_vta':
                    _c = _sub
                    _sub = 0

                c_total = _cost_total + _c
                assert (c_total >= _cost_total).all().all(), 'Cost issue'
                sub_total = _subsidies_total - _sub
                assert (sub_total <= _subsidies_total).all().all(), 'Subsidies issue'
                ms_sub, retrofit_sub = apply_endogenous_retrofit(_bill_saved, sub_total, c_total)
                f_retrofit_sub = retrofit_sub * stock
                f_replace_sub = (f_retrofit_sub * ms_sub.T).T
                avg_cost_global_sub = (f_replace_sub * _cost_total).sum().sum() / f_retrofit_sub.sum()
                f_replace_sub = f_replace_sub[mask]
                avg_cost_benef_sub = (f_replace_sub * _cost_total).sum().sum() / f_replace_sub.sum().sum()
                f_replace_benef = f_replace[mask]
                avg_cost_benef = (f_replace_benef * _cost_total).sum().sum() / f_replace_benef.sum().sum()

                if key == 'reduced_vta':
                    avg_sub = (f_replace_benef * _c).sum().sum() / f_replace_benef.sum().sum()
                else:
                    avg_sub = (f_replace_benef * _sub).sum().sum() / f_replace_benef.sum().sum()

                _free_riders = f_retrofit_sub.sum() / f_retrofit.sum()
                _intensive_margin = (avg_cost_global - avg_cost_global_sub) / avg_cost_global_sub
                _intensive_margin_benef = (avg_cost_benef - avg_cost_benef_sub) / avg_cost_benef_sub
                share_sub = avg_sub / avg_cost_benef_sub

                rslt.update({key: pd.Series([_free_riders, _intensive_margin, _intensive_margin_benef, avg_sub, avg_cost_benef_sub, share_sub],
                                            index=['Freeriders (%)', 'Intensive margin (%)', 'Intensive margin benef (%)', 'Average sub (euro)', 'Average cost benef (euro)', 'Share sub (%)'])})

            rslt = DataFrame(rslt)
            return rslt

        index = stock.index
        cost_total = reindex_mi(cost_insulation, index)

        consumption_before = self.consumption_heating_store(index, level_heater='Heating system final')[0]
        consumption_before = reindex_mi(consumption_before, index) * reindex_mi(self._surface, index)
        energy_bill_before = AgentBuildings.energy_bill(prices, consumption_before, level_heater='Heating system final')

        consumption_after = self.prepare_consumption(self._choice_insulation, index=index,
                                                     level_heater='Heating system final', full_output=False)
        consumption_after = reindex_mi(consumption_after, index).reindex(self._choice_insulation, axis=1)
        consumption_after = (consumption_after.T * reindex_mi(self._surface, index)).T
        consumption_saved = (consumption_before - consumption_after.T).T

        energy = pd.Series(index.get_level_values('Heating system final'), index=index).str.split('-').str[0].rename(
            'Energy')
        energy_prices = prices.reindex(energy).set_axis(index)
        energy_bill_sd = (consumption_after.T * energy_prices).T
        bill_saved = - energy_bill_sd.sub(energy_bill_before, axis=0).dropna()

        if self.constant_insulation_intensive is None and self._threshold is False:
            calibration_coupled(stock, cost_total, bill_saved, subsidies_total, calib_renovation, calib_intensive)

            result = assess_policies(stock, subsidies_details, cost_total, bill_saved, subsidies_total)
            if self.path is not None:
                result.to_csv(os.path.join(self.path_calibration, 'result_policies_assessment.csv'))
            assess_sensitivity(stock, cost_total, bill_saved, subsidies_total, self.path_calibration)

        if self._threshold is False:
            market_share, renovation_rate = apply_endogenous_retrofit(bill_saved, subsidies_total, cost_total)
        else:
            market_share, renovation_rate = apply_rational_choice(consumption_saved, subsidies_total, cost_total)

        if min_performance is not None:
            certificate = reindex_mi(self._consumption_store['certificate_renovation'], market_share.index)
            market_share = market_share[certificate <= min_performance]
            market_share = (market_share.T / market_share.sum(axis=1)).T

        return renovation_rate, market_share

    def exogenous_retrofit(self, stock, condition):
        """Format retrofit rate and market share for each segment.


        Global retrofit and retrofit rate to match exogenous numbers.
        Retrofit all heating system replacement dwelling.

        Parameters
        ----------
        stock: Series
        condition: dict

        Returns
        -------
        Series
            Retrofit rate by segment.
        DataFrame
            Market-share by segment and insulation choice.
        """

        if self.param_exogenous['target'] == 'heater_replacement':
            retrofit_rate = pd.Series(0, index=stock.index)
            retrofit_rate[retrofit_rate.index.get_level_values('Heater replacement')] = 1
        elif self.param_exogenous['target'] == 'worst':
            consumption = reindex_mi(self._consumption_store['consumption'], stock.index)
            temp = pd.concat((consumption, stock), keys=['Consumption', 'Stock'], axis=1)
            temp = temp.sort_values('Consumption', ascending=False)
            temp['Stock'] = temp['Stock'].cumsum()
            idx = temp[temp['Stock'] < self.param_exogenous['number']].index
            retrofit_rate = pd.Series(1, index=idx)
            # segment to complete
            to_complete = self.param_exogenous['number'] - stock[idx].sum()
            idx = temp[temp['Stock'] >= self.param_exogenous['number']].index
            if not idx.empty:
                idx = idx[0]
                to_complete = pd.Series(to_complete / stock[idx], index=pd.MultiIndex.from_tuples([idx], names=temp.index.names))
                retrofit_rate = concat((retrofit_rate, to_complete))
        else:
            raise NotImplemented

        """market_share = DataFrame(0, index=stock.index, columns=choice_insulation)
        market_share.loc[:, (True, True, True, True)] = 1"""
        market_share = condition['best_option'].astype(int)
        market_share = reindex_mi(market_share, retrofit_rate.index).dropna()

        assert market_share.loc[market_share.sum(axis=1) != 1].empty, 'Market share problem'

        return retrofit_rate, market_share

    def store_information_insulation(self, condition, cost_insulation_raw, tax, cost_insulation, cost_financing,
                                     vta_insulation, subsidies_details, subsidies_total, consumption_saved,
                                     amount_debt, amount_saving, discount):
        """Store insulation information.


        Information are post-treated to weight by the number of replacement.

        Parameters
        ----------
        condition: dict
        cost_insulation_raw: Series
            Cost of insulation for each envelope component of losses surface (€/m2).
        tax: float
            VTA to apply (%).
        cost_insulation: DataFrame
            Cost total for each dwelling and each insulation gesture (€). Financing cost included.
        cost_financing: DataFrame
            Financing cost  for each dwelling and each insulation gesture (€).
        vta_insulation: DataFrame
            VTA applied to each insulation gesture cost (€).
        subsidies_details: dict
            Amount of subsidies for each dwelling and each insulation gesture (€).
        subsidies_total: DataFrame
            Total mount of subsidies for each dwelling and each insulation gesture (€).
        consumption_saved: DataFrame
        amount_debt: Series
        amount_saving: Series
        discount:Series
        """

        self._condition_store = condition

        self._renovation_store.update({
            'cost_households': cost_insulation,
            'cost_financing_households': cost_financing,
            'vta_households': vta_insulation,
            'subsidies_households': subsidies_total,
            'subsidies_details_households': subsidies_details,
            'cost_component': cost_insulation_raw * self.surface_insulation * (1 + tax),
            'consumption_saved_households': consumption_saved,
            'debt_households': amount_debt,
            'saving_households': amount_saving,
            'discount': discount

        })

    def insulation_replacement(self, stock, prices, cost_insulation_raw, policies_insulation=None, financing_cost=None,
                               calib_renovation=None, calib_intensive=None, min_performance=None):
        """Calculate insulation retrofit in the dwelling stock.

        1. Intensive margin
        2. Extensive margin
        Calibrate function first year.

        Consumption saving only depends on insulation work.
        However, certificate upgrade also consider the heat

        To reduce calculation time attributes are grouped.
        Cost, subsidies and constant depends on Housing type, Occupancy status, Housing type and Insulation performance.

        Parameters
        ----------
        financing_cost
        stock: Series
        prices: Series
        cost_insulation_raw: Series
            €/m2 of losses area by component.
        policies_insulation: list
        calib_renovation: dict
        calib_intensive: dict
        min_performance


        Returns
        -------
        Series
            Retrofit rate
        DataFrame
            Market-share insulation
        """

        index = stock.index

        # select index that can undertake insulation replacement
        consumption_before_heater, _, certificate_before_heater = self.consumption_heating_store(index,
                                                                                                 level_heater='Heating system')

        # before include the change of heating system
        consumption_before, consumption_3uses_before, certificate_before = self.consumption_heating_store(index,
                                                                                                          level_heater='Heating system final')

        # select only index that can actually be retrofitted
        certificate_before = certificate_before[certificate_before > 'B']
        consumption_3uses_before = consumption_3uses_before.loc[certificate_before.index]
        consumption_before = consumption_before.loc[certificate_before.index]
        temp = reindex_mi(certificate_before, index)
        index = temp[temp > 'B'].index
        stock = stock[index]

        # seems to not be very useful
        condition = np.array([1] * stock.shape[0], dtype=bool)
        for k, v in self._performance_insulation.items():
            condition *= stock.index.get_level_values(k) <= v
        stock = stock[~condition]
        index = stock.index

        surface = reindex_mi(self._surface, index)

        # calculation of energy_saved_3uses after heating system final
        consumption_after, consumption_3uses, certificate = self.prepare_consumption(self._choice_insulation,
                                                                                     index=index,
                                                                                     level_heater='Heating system final')
        energy_saved_3uses = ((consumption_3uses_before - consumption_3uses.T) / consumption_3uses_before).T
        energy_saved_3uses.dropna(inplace=True)

        consumption_saved = (consumption_before - consumption_after.T).T
        consumption_saved = (reindex_mi(consumption_saved, index).T * reindex_mi(self._surface, index)).T

        cost_insulation = self.prepare_cost_insulation(cost_insulation_raw * self.surface_insulation)
        cost_insulation = cost_insulation.T.multiply(self._surface, level='Housing type').T

        cost_insulation, vta_insulation, tax, subsidies_details, subsidies_total, condition = self.apply_subsidies_insulation(
            index, policies_insulation, cost_insulation, surface, certificate, certificate_before,
            certificate_before_heater, energy_saved_3uses, consumption_saved)

        cost_total, cost_financing, amount_debt, amount_saving, discount = self.calculate_financing(
            reindex_mi(cost_insulation, index),
            subsidies_total,
            financing_cost)

        assert np.allclose(amount_debt + amount_saving + subsidies_total, reindex_mi(cost_insulation, subsidies_total.index)), 'Sum problem'

        if self._condition_store is None:
            if self.full_output:
                self.store_information_insulation(condition, cost_insulation_raw, tax, cost_insulation, cost_financing,
                                                  vta_insulation, subsidies_details, subsidies_total, consumption_saved,
                                                  amount_debt, amount_saving, discount)
            else:
                self._renovation_store['subsidies_details_households'] = subsidies_details

        if self._endogenous:

            if calib_renovation is not None:
                if calib_renovation['scale']['name'] == 'freeriders':
                    delta_subsidies = None
                    if (self.year in [self.first_year + 1]) and (self.scale is None):
                        delta_subsidies = subsidies_details[calib_renovation['scale']['target_policies']].copy()
                    calib_renovation['scale']['delta_subsidies'] = delta_subsidies

            retrofit_rate, market_share = self.endogenous_retrofit(stock, prices, subsidies_total, cost_total,
                                                                   calib_intensive=calib_intensive,
                                                                   calib_renovation=calib_renovation,
                                                                   min_performance=min_performance,
                                                                   subsidies_details=subsidies_details)

        else:
            retrofit_rate, market_share = self.exogenous_retrofit(stock, condition)

        if self._renovation_store['renovation_rate'] is None:
            self._renovation_store['renovation_rate'], self._renovation_store['market_share'] = retrofit_rate, market_share

        return retrofit_rate, market_share

    def store_energy_saving(self, flow_retrofit, flow_heater, prices):
        def saving(_flow, _prices):
            _flow = _flow[_flow > 0]
            _index = _flow.index

            _consumption_before = self.consumption_heating_store(_index, level_heater='Heating system', full_output=False)
            _consumption_before = reindex_mi(_consumption_before, _index) * reindex_mi(self._surface, _index)
            _consumption_before = self.consumption_actual(prices, consumption=_consumption_before)
            _consumption_before = (_consumption_before * _flow.T).T

            if isinstance(_flow, DataFrame):
                # insulation
                _consumption = self.prepare_consumption(index=_index, level_heater='Heating system',
                                                        full_output=False, store=False)
                _consumption = reindex_mi(_consumption, _index)
                _consumption = (_consumption.T * reindex_mi(self._surface, _index)).T

                heating_intensity = self.to_heating_intensity(_index, _prices, consumption=_consumption,
                                                              level_heater='Heating system final')
                _consumption *= _flow
                _consumption_after = _consumption * heating_intensity
                _consumption_saving_insulation = _consumption_before - _consumption_after

                # heater
                _consumption_saving_heater = None
                if 'Heating system final' in _index.names:
                    _consumption = self.prepare_consumption(index=_index, level_heater='Heating system final',
                                                            full_output=False)
                    _consumption = reindex_mi(_consumption, _index)
                    _consumption = (_consumption.T * reindex_mi(self._surface, _index)).T
                    heating_intensity = self.to_heating_intensity(_index, _prices, consumption=_consumption,
                                                                  level_heater='Heating system final')
                    _consumption *= _flow
                    _consumption = _consumption * heating_intensity
                    _consumption_saving_heater = _consumption_after - _consumption

                return _consumption_saving_insulation, _consumption_saving_heater

            if isinstance(_flow, Series):
                _consumption_saving_heater = None
                if 'Heating system final' in _index.names:
                    _consumption = self.consumption_heating_store(_index, level_heater='Heating system final',
                                                             full_output=False)
                    _consumption = reindex_mi(_consumption, _index)
                    _consumption = (_consumption.T * reindex_mi(self._surface, _index)).T
                    heating_intensity = self.to_heating_intensity(_index, _prices, consumption=_consumption,
                                                                  level_heater='Heating system final')
                    _consumption *= _flow
                    _consumption = _consumption * heating_intensity
                    _consumption_saving_heater = _consumption_before - _consumption
                    return _consumption_saving_heater

        consumption_saving_insulation, consumption_saving_heater = saving(flow_retrofit, prices)
        consumption_saving_insulation = consumption_saving_insulation.sum().sum()
        consumption_saving_heater = consumption_saving_heater.sum().sum()

        consumption_saving_only_heater = saving(flow_heater.stack(), prices)
        consumption_saving_only_heater = consumption_saving_only_heater.sum().sum()
        consumption_saving_heater += consumption_saving_only_heater

        self.consumption_saving_insulation = consumption_saving_insulation * self.coefficient_global
        self.consumption_saving_heater = consumption_saving_heater * self.coefficient_global

    def store_rebound(self, flow_retrofit, flow_heater, prices, prices_before=None, climate=2006):
        """Calculate consumption based on new building characteristics but current heating intensity.


        Useful to estimate the rebound effect.

        Parameters
        ----------
        flow_retrofit: Series or DataFrame
        flow_heater: Series or DataFrame
        prices: Series
        prices_before: Series
        climate: int, optional

        Returns
        -------
        Series
        """

        def calculate_rebound(_flow, _prices, _prices_before=None, _climate=None):
            # heating intensity is calculated based on index before renovation
            _flow = _flow[_flow > 0]
            _index = _flow.index

            if _prices_before is None:
                _prices_before = _prices

            consumption = self.consumption_heating(index=_index, freq='year', climate=_climate, full_output=False)
            consumption = reindex_mi(consumption, _index) * reindex_mi(self._surface, _index)
            intensity_before = self.to_heating_intensity(_index, _prices_before, consumption=consumption)

            # consumption standard is calculated based on retrofitting (heating system and insulation)
            # TODO: change prepare_consumption docstring
            if isinstance(_flow, DataFrame):
                # insulation and heater
                _consumption = self.prepare_consumption(index=_index, level_heater='Heating system final',
                                                        full_output=False, climate=_climate)
                _consumption = reindex_mi(_consumption, _index).reindex(_consumption.columns, axis=1)
                _consumption = (_consumption.T * reindex_mi(self._surface, _index)).T
                intensity_after = self.to_heating_intensity(_index, _prices, consumption=_consumption,
                                                            level_heater='Heating system final')
                _consumption *= _flow
                _consumption_before = (_consumption.T * intensity_before).T
                _consumption_before = _consumption_before.sum(axis=1)
                _consumption_after = _consumption * intensity_after
                _consumption_after = _consumption_after.sum(axis=1)

                return _consumption_before, _consumption_after

            elif isinstance(_flow, Series):
                level_heater = 'Heating system'
                if 'Heating system final' in _index.names:
                    level_heater = 'Heating system final'

                _consumption = self.consumption_heating(index=_index, freq='year', climate=_climate, full_output=False,
                                                        level_heater=level_heater)
                _consumption = reindex_mi(_consumption, _index) * reindex_mi(self._surface, _index)

                intensity_after = self.to_heating_intensity(_index, _prices, consumption=_consumption,
                                                            level_heater=level_heater)

                _consumption *= _flow
                _consumption_before = _consumption * intensity_before
                _consumption_after = _consumption * intensity_after

                return _consumption_before, _consumption_after

        def clean_flow(_flow, _index):
            if isinstance(_flow, DataFrame):
                _flow = _flow.sum(axis=1)

            if 'Heating system final' in _flow.index.names:
                _flow = _flow.droplevel('Heating system')
                _flow.index = _flow.index.rename('Heating system', level='Heating system final')
            _flow = _flow.groupby(_index.names).sum()

            _union = _index.union(_flow.index)
            _flow = _flow.reindex(_union, fill_value=0)
            return _flow

        flow_heater = flow_heater.stack()

        index = self.stock.index
        c_no_rebound_retrofit, c_rebound_retrofit = calculate_rebound(flow_retrofit, prices,
                                                                      _prices_before=prices_before, _climate=climate)
        c_no_rebound_retrofit = clean_flow(c_no_rebound_retrofit, index)
        c_rebound_retrofit = clean_flow(c_rebound_retrofit, index)

        c_no_rebound_heater, c_rebound_heater = calculate_rebound(flow_heater, prices, _prices_before=prices_before,
                                                                  _climate=climate)
        c_no_rebound_heater = clean_flow(c_no_rebound_heater, index)
        c_rebound_heater = clean_flow(c_no_rebound_heater, index)

        flow_retrofit = clean_flow(flow_retrofit.droplevel('Heating system final'), index)
        flow_heater = clean_flow(flow_heater.droplevel('Heating system final'), index)
        stock_remaining = self.stock - flow_retrofit - flow_heater
        c_no_rebound_remaining, c_rebound_remaining = calculate_rebound(stock_remaining, prices,
                                                                        _prices_before=prices_before, _climate=climate)

        union = c_no_rebound_retrofit.index.union(c_no_rebound_heater.index)
        union = union.union(c_no_rebound_remaining.index)

        c_no_rebound_retrofit = c_no_rebound_retrofit.reindex(union, fill_value=0)
        c_rebound_retrofit = c_rebound_retrofit.reindex(union, fill_value=0)
        c_no_rebound_heater = c_no_rebound_heater.reindex(union, fill_value=0)
        c_rebound_heater = c_rebound_heater.reindex(union, fill_value=0)
        c_no_rebound_remaining = c_no_rebound_remaining.reindex(union, fill_value=0)
        c_rebound_remaining = c_rebound_remaining.reindex(union, fill_value=0)

        c_no_rebound = c_no_rebound_retrofit + c_no_rebound_heater + c_no_rebound_remaining
        c_rebound = c_rebound_retrofit + c_rebound_heater + c_rebound_remaining

        c_no_rebound_energy = self.apply_calibration(c_no_rebound)
        c_rebound_energy = self.apply_calibration(c_rebound)

        self.rebound = c_rebound_energy - c_no_rebound_energy
        self.cost_rebound = (prices * self.rebound).sum()

    def flow_retrofit(self, prices, cost_heater, lifetime_heater, cost_insulation, policies_heater=None,
                      policies_insulation=None,
                      ms_heater=None, financing_cost=None, calib_renovation=None, calib_intensive=None, climate=None,
                      step=1):
        """Compute heater replacement and insulation retrofit.


        1. Heater replacement based on current stock segment.
        2. Knowing heater replacement (and new heating system) calculating retrofit rate by segment and market
        share by segment.
        3. Then, managing inflow and outflow.

        Parameters
        ----------
        prices: Series
        cost_heater: Series
        lifetime_heater: Series
        ms_heater: DataFrame
        cost_insulation
        policies_heater: list
            Policies for heating system.
        policies_insulation: list
            Policies for insulation.
        financing_cost: optional, dict
        calib_renovation: dict, optional
        calib_intensive: dict, optional
        climate

        Returns
        -------
        Series
        """

        # select only stock mobile and existing before the first year
        stock_mobile = self.stock_mobile.groupby([i for i in self.stock_mobile.index.names if i != 'Income tenant']).sum()
        stock_mobile = stock_mobile.xs(True, level='Existing', drop_level=False)

        # accounts for heater replacement - depends on energy prices, cost and policies heater
        self.logger.info('Calculation heater replacement')
        stock = self.heater_replacement(stock_mobile, prices, cost_heater, lifetime_heater, policies_heater,
                                        ms_heater=ms_heater, step=step, financing_cost=financing_cost)

        self.logger.debug('Agents: {:,.0f}'.format(stock.shape[0]))
        self.logger.info('Calculation insulation replacement')

        retrofit_rate, market_share = self.insulation_replacement(stock, prices, cost_insulation,
                                                                  calib_renovation=calib_renovation,
                                                                  calib_intensive=calib_intensive,
                                                                  policies_insulation=policies_insulation,
                                                                  financing_cost=financing_cost)
        self.logger.info('Formatting and storing replacement')
        # step
        s = sum([(1 - retrofit_rate)**k for k in range(step)])
        flow_insulation = retrofit_rate.reindex(stock.index).fillna(0) * stock * s.reindex(stock.index).fillna(0)

        # heater replacement without insulation upgrade
        flow_only_heater = stock - flow_insulation
        flow_only_heater = flow_only_heater.xs(True, level='Heater replacement', drop_level=False).unstack(
            'Heating system final')
        flow_only_heater_sum = flow_only_heater.sum().sum()

        # insulation upgrade
        flow_insulation = flow_insulation[flow_insulation > 0]
        replacement_sum = flow_insulation.sum().sum()
        replaced_by = (flow_insulation * market_share.T).T

        assert round(replaced_by.sum().sum(), 0) == round(replacement_sum, 0), 'Sum problem'

        # energy performance certificate jump due to heater replacement without insulation upgrade
        only_heater = (stock - flow_insulation.reindex(stock.index, fill_value=0)).xs(True, level='Heater replacement')

        # storing information (flow, investment, subsidies)
        if self.full_output:
            self.logger.debug('Store information retrofit')
            self._replaced_by = replaced_by.copy()
            self._only_heater = only_heater.copy()
            # self.store_information_retrofit(replaced_by, only_heater)

        # removing heater replacement level
        replaced_by = replaced_by.groupby(
            [c for c in replaced_by.index.names if c != 'Heater replacement']).sum()
        flow_only_heater = flow_only_heater.groupby(
            [c for c in flow_only_heater.index.names if c != 'Heater replacement']).sum()

        # adding income tenant information
        self.logger.debug('Adding income tenant')
        replaced_by = self.add_level(replaced_by, self.stock_mobile, 'Income tenant')
        assert round(replaced_by.sum().sum(), 0) == round(replacement_sum, 0), 'Sum problem'

        flow_only_heater = self.add_level(flow_only_heater, self.stock_mobile, 'Income tenant')
        assert round(flow_only_heater.sum().sum(), 0) == round(flow_only_heater_sum, 0), 'Sum problem'

        if self._debug_mode:
            self.logger.debug('Calculate rebound effect')
            self.store_rebound(replaced_by, flow_only_heater, prices, climate=climate)
            self.logger.debug('Calculate energy saving')
            self.store_energy_saving(replaced_by, flow_only_heater, prices)

        self.logger.debug('Formatting switch heater flow')
        flow_only_heater = flow_only_heater.stack('Heating system final')
        to_replace_heater = - flow_only_heater.droplevel('Heating system final')

        replaced_by_heater = flow_only_heater.droplevel('Heating system')
        replaced_by_heater.index = replaced_by_heater.index.rename('Heating system', level='Heating system final')
        replaced_by_heater = replaced_by_heater.reorder_levels(to_replace_heater.index.names)

        flow_only_heater = pd.concat((to_replace_heater, replaced_by_heater), axis=0)
        flow_only_heater = flow_only_heater.groupby(flow_only_heater.index.names).sum()
        assert round(flow_only_heater.sum(), 0) == 0, 'Sum problem'

        self.logger.debug('Formatting renovation flow')
        to_replace = replaced_by.droplevel('Heating system final').sum(axis=1).copy()
        to_replace = to_replace.groupby(to_replace.index.names).sum()
        assert round(to_replace.sum(), 0) == round(replacement_sum, 0), 'Sum problem'

        self.logger.debug('Start frame to flow')
        replaced_by = self.frame_to_flow(replaced_by)
        self.logger.debug('End frame to flow')

        self.logger.debug('Concatenate all flows')
        to_replace = to_replace.reorder_levels(replaced_by.index.names)
        flow_only_heater = flow_only_heater.reorder_levels(replaced_by.index.names)
        flow_retrofit = concat((-to_replace, replaced_by, flow_only_heater), axis=0)
        flow_retrofit = flow_retrofit.groupby(flow_retrofit.index.names).sum()
        assert round(flow_retrofit.sum(), 0) == 0, 'Sum problem'

        return flow_retrofit

    def flow_obligation(self, policies_insulation, prices, cost_insulation, financing_cost=True):
        """Account for flow obligation if defined in policies_insulation.

        Parameters
        ----------
        policies_insulation: list
            Check if obligation.
        prices: Series
        cost_insulation: Series
        financing_cost: bool, optional

        Returns
        -------
        flow_obligation: Series
        """

        stock = self.stock_mobile.copy()

        list_obligation = [p for p in policies_insulation if p.policy == 'obligation']
        if list_obligation == []:
            return None
        self.logger.info('Calculation flow obligation')
        # only work if there is one obligation
        flows_obligation = []
        for obligation in list_obligation:
            banned_performance = obligation.value.loc[self.year]
            if not isinstance(banned_performance, str):
                return None

            performance_target = [i for i in resources_data['index']['Performance'] if i >= banned_performance]

            stock_certificate = self.add_certificate(stock)
            idx = stock.index[stock_certificate.index.get_level_values('Performance').isin(performance_target)]
            if idx.empty:
                return None

            proba = 1
            if obligation.frequency is not None:
                proba = obligation.frequency
                proba = reindex_mi(proba, idx)

            to_replace = stock.loc[idx] * proba

            # formatting replace_by
            replaced_by = to_replace.copy()
            replaced_by = replaced_by.groupby([i for i in replaced_by.index.names if i != 'Income tenant']).sum()

            if 'Heater replacement' not in replaced_by:
                replaced_by = concat([replaced_by], keys=[False], names=['Heater replacement'])
            if 'Heating system final' not in replaced_by.index.names:
                temp = replaced_by.reset_index('Heating system')
                temp['Heating system final'] = temp['Heating system']
                replaced_by = temp.set_index(['Heating system', 'Heating system final'], append=True).squeeze()
            replaced_by.index = replaced_by.index.reorder_levels(self._renovation_store['market_share'].index.names)

            _, market_share = self.insulation_replacement(replaced_by, prices, cost_insulation,
                                                          policies_insulation=policies_insulation,
                                                          financing_cost=financing_cost,
                                                          min_performance=obligation.min_performance)

            if obligation.intensive == 'market_share':
                # market_share endogenously calculated by insulation_replacement
                pass
            elif obligation.intensive == 'global':
                market_share = DataFrame(0, index=replaced_by.index, columns=self._choice_insulation)
                market_share.loc[:, (True, True, True, True)] = 1
            else:
                raise NotImplemented

            assert ~market_share.isna().all(axis=1).any(), "Market-share issue"
            assert (market_share.sum(axis=1).round(5) == 1.0).all(), "Market-share sum issue"

            replaced_by = (replaced_by.rename(None) * market_share.T).T
            replaced_by = replaced_by.fillna(0)

            if self.full_output:
                self._replaced_by = self._replaced_by.add(replaced_by.copy(), fill_value=0)

            replaced_by = self.frame_to_flow(replaced_by)

            assert to_replace.sum().round(0) == replaced_by.sum().round(0), 'Sum problem'
            flow_obligation = concat((- to_replace, replaced_by), axis=0)
            flow_obligation = flow_obligation.groupby(flow_obligation.index.names).sum()
            flows_obligation.append(flow_obligation)
        return flows_obligation

    def store_information_retrofit(self):
        """Calculate and store main statistics based on yearly retrofit.

        Parameters
        ----------
        replaced_by: DataFrame
            Retrofit flow for each dwelling (index) and each insulation gesture (columns).
            Dwelling must be defined with 'Heating system final' and 'Heater replacement'.
        only_heater
        """

        replaced_by = self._replaced_by
        only_heater = self._only_heater

        if only_heater is not None:
            self._renovation_store['only_heater'] = only_heater.sum()
            certificate_jump = self._heater_store['certificate_jump'].stack()
            temp = {}
            for i in unique(certificate_jump):
                temp.update({i: ((certificate_jump == i) * only_heater).sum()})
            self._heater_store['certificate_jump'] = Series(temp).sort_index()

        names = self._condition_store['global_renovation_high_income'].index.names
        replaced_by.index = replaced_by.index.reorder_levels(names)

        self._renovation_store['global_renovation_high_income'] = (replaced_by * self._condition_store['global_renovation_high_income']).sum().sum()
        self._renovation_store['global_renovation_low_income'] = (replaced_by * self._condition_store['global_renovation_low_income']).sum().sum()
        self._renovation_store['bonus_best'] = (replaced_by * self._condition_store['bonus_best']).sum().sum()
        self._renovation_store['bonus_worst'] = (replaced_by * self._condition_store['bonus_worst']).sum().sum()
        if True in replaced_by.index.get_level_values('Heater replacement'):
            self._renovation_store['renovation_with_heater'] = replaced_by.xs(True, level='Heater replacement').sum().sum()

        self._renovation_store['vta'] = (replaced_by * self._renovation_store['vta_households']).sum().sum()

        # replaced_by
        levels = [i for i in replaced_by.index.names if i not in ['Heater replacement', 'Heating system final']]

        self._renovation_store['replacement'] = replaced_by.groupby(levels).sum()
        self._renovation_store['cost'] = (replaced_by * self._renovation_store['cost_households']).groupby(levels).sum()
        self._renovation_store['cost_financing'] = (
                    replaced_by * self._renovation_store['cost_financing_households']).groupby(levels).sum()
        self._renovation_store['subsidies'] = (replaced_by * self._renovation_store['subsidies_households']).groupby(
            levels).sum()

        to_pay = self._renovation_store['cost'] - self._renovation_store['subsidies']
        self._renovation_store['annuities'] = calculate_annuities(to_pay, lifetime=10,
                                                                  discount_rate=self._renovation_store['discount'])

        self._renovation_store['consumption_saved'] = (replaced_by * self._renovation_store['consumption_saved_households']).groupby(levels).sum()

        consumption = self.consumption_heating_store(self._renovation_store['consumption_saved_households'].index, full_output=False, level_heater='Heating system final')
        consumption = reindex_mi(consumption, self._renovation_store['consumption_saved_households'].index)
        consumption *= reindex_mi(self._surface, consumption.index)
        consumption_saved = (self._renovation_store['consumption_saved_households'].T / consumption).T
        assert (consumption_saved <= 1).all().all(), 'Percent issue'
        self._renovation_store['consumption_saved_mean'] = (consumption_saved * replaced_by).sum().sum() / replaced_by.sum().sum()
        self._renovation_store['consumption_saved_investor'] = (consumption_saved * replaced_by).sum(
            axis=1).groupby(['Housing type', 'Occupancy status']).sum() / replaced_by.sum(axis=1).groupby(
            ['Housing type', 'Occupancy status']).sum()

        for key, sub in self._renovation_store['subsidies_details_households'].items():
            self._renovation_store['subsidies_details'][key] = (replaced_by * reindex_mi(sub, replaced_by.index)).groupby(levels).sum()
            mask = sub.copy()
            mask[mask > 0] = 1
            self._renovation_store['subsidies_count'][key] = (replaced_by.fillna(0) * mask).sum(axis=1).groupby('Housing type').sum()
            if replaced_by.sum().sum() == 0:
                self._renovation_store['subsidies_average'][key] = 0
            else:
                self._renovation_store['subsidies_average'][key] = sub.sum().sum() / replaced_by.fillna(0).sum().sum()

        self._renovation_store['debt'] = (replaced_by * self._renovation_store['debt_households']).groupby(
            levels).sum().sum(axis=1).groupby('Income owner').sum()
        self._renovation_store['saving'] = (replaced_by * self._renovation_store['saving_households']).groupby(
            levels).sum().sum(axis=1).groupby('Income owner').sum()

        self._renovation_store['discount'] = self._renovation_store['discount'].groupby(levels).mean()

        certificate_jump_all = self._condition_store['certificate_jump_all']
        temp = {}
        for i in unique(certificate_jump_all.values.ravel('K')):
            temp.update({i: ((certificate_jump_all == i) * replaced_by).sum(axis=1)})
        self._renovation_store['certificate_jump_all'] = DataFrame(temp).groupby(levels).sum()

        temp = {i: 0 for i in range(1, 6)}
        for n, g in NB_MEASURES.items():
            temp[n] += replaced_by.loc[:, g].xs(False, level='Heater replacement').sum().sum()
            temp[n + 1] += replaced_by.loc[:, g].xs(True, level='Heater replacement').sum().sum()
        if only_heater is not None:
            temp[1] += only_heater.sum()
        self._renovation_store['nb_measures'] += Series(temp)

    def parse_output_run(self, prices, inputs, climate=None, step=1, taxes=None, detailed_output=True):
        """Parse output.

        Renovation : envelope
        Retrofit : envelope and/or heating system

        Parameters
        ----------
        prices: Series
        inputs: dict
            Exogenous data for post-treatment.
            'carbon_emission'
            'population'
            'surface'
            'embodied_energy_renovation'
            'carbon_footprint_renovation'
            'Carbon footprint construction (MtCO2)'
            'health_expenditure', 'mortality_cost', 'loss_well_being'
            'Embodied energy construction (TWh PE)'
        climate: int, optional

        Returns
        -------

        """

        def make_cost_curve(_consumption_saved, _cost_insulation, _stock, lifetime=50, discount_rate=0.032):
            cost_annualized = calculate_annuities(_cost_insulation, lifetime=lifetime,
                                                  discount_rate=discount_rate)

            insulation = pd.MultiIndex.from_frame(pd.DataFrame(INSULATION))

            _consumption_saved = _consumption_saved.loc[:, insulation]
            cost_annualized = cost_annualized.loc[:, insulation]
            cost_efficiency = cost_annualized / _consumption_saved

            # multiply cost_saved by stock cost_efficiency do not change
            _consumption_saved = (_stock.groupby(_consumption_saved.index.names).sum() * _consumption_saved.T).T

            x = _consumption_saved.stack(_consumption_saved.columns.names).squeeze().rename(
                'Consumption saved (TWh)')
            y = cost_efficiency.stack(cost_efficiency.columns.names).squeeze().rename(
                'Cost efficiency (euro/kWh)')
            c = (x * y).rename('Cost (euro/year)')
            df = pd.concat((x, y, c), axis=1)

            # sort by marginal cost
            df.sort_values(y.name, inplace=True)
            df.dropna(inplace=True)
            df[y.name] = df[y.name].round(3)
            df = df.groupby([y.name]).agg({x.name: 'sum', y.name: 'first'})
            df[x.name] = df[x.name].cumsum() / 10 ** 9
            df = df.set_index(x.name)[y.name]

            make_plot(df, 'Cost efficiency (euro/kWh)', ymax=0.5, legend=False,
                      format_y=lambda y, _: '{:.1f}'.format(y),
                      save=os.path.join(self.path, 'cost_curve_insulation.png'))

        if self.year == 2019:
            c_saved = self._renovation_store['consumption_saved_households'].droplevel(['Heating system final', 'Heater replacement'])
            c_saved = c_saved[~c_saved.index.duplicated()]
            carbon_value = inputs['carbon_value_kwh'].loc[self.year, :]
            # carbon_value_2030 = inputs['carbon_value_kwh'].loc[2030, :]

            carbon_value = carbon_value.reindex(self.to_energy(c_saved)).set_axis(c_saved.index)
            e_saved = (c_saved.T * carbon_value).T
            c_insulation = reindex_mi(self._renovation_store['cost_households'], c_saved.index)
            # TODO private and social discount rate
            make_cost_curve(c_saved, c_insulation, self.stock, discount_rate=self._renovation_store['discount'])

            _consumption_saved = (self.stock.groupby(c_saved.index.names).sum() * c_saved.T).T

            _, _, certificate = self.prepare_consumption()
            certificate = reindex_mi(certificate, self.stock.index)

        stock = self.simplified_stock()

        output = dict()
        output['Stock (Million)'] = stock.sum() / 10 ** 6
        output['Surface (Million m2)'] = (self.stock * self.surface).sum() / 10 ** 6
        output['Surface (m2/person)'] = (
                    output['Surface (Million m2)'] / (inputs['population'].loc[self.year] / 10 ** 6))

        output['Consumption standard (TWh)'] = self.consumption_total(prices=prices, freq='year', climate=climate,
                                                                      standard=True, energy=False)
        output['Consumption standard (kWh/m2)'] = (output['Consumption standard (TWh)'] * 10 ** 9) / (
                output['Surface (Million m2)'] * 10 ** 6)

        consumption_energy = self.consumption_total(prices=prices, freq='year', climate=None, standard=False, energy=True)
        output['Consumption (TWh)'] = consumption_energy.sum()
        output['Consumption (kWh/m2)'] = (output['Consumption (TWh)'] * 10 ** 9) / (
                output['Surface (Million m2)'] * 10 ** 6)
        output['Heating intensity (%)'] = (self.stock * self.heating_intensity_save).sum() / self.stock.sum()

        temp = consumption_energy.copy()
        temp.index = temp.index.map(lambda x: 'Consumption {} (TWh)'.format(x))
        output.update(temp.T)

        consumption_energy_climate = None
        if climate is not None:
            consumption_energy_climate = self.consumption_total(prices=prices, freq='year', climate=climate,
                                                                standard=False, energy=True)
            output['Consumption climate (TWh)'] = consumption_energy_climate.sum()
            temp = consumption_energy_climate.copy()
            temp.index = temp.index.map(lambda x: 'Consumption {} climate (TWh)'.format(x))
            output.update(temp.T)
            output['Factor climate (%)'] = output['Consumption climate (TWh)'] / output['Consumption (TWh)']

        consumption = self.consumption_actual(prices) * self.stock
        consumption_calib = consumption * self.coefficient_global
        temp = consumption_calib.groupby('Existing').sum()
        temp.rename(index={True: 'Existing', False: 'New'}, inplace=True)
        temp.index = temp.index.map(lambda x: 'Consumption {} (TWh)'.format(x))
        output.update(temp.T / 10 ** 9)

        temp = consumption_calib.groupby(self.certificate).sum()
        temp.index = temp.index.map(lambda x: 'Consumption {} (TWh)'.format(x))
        output.update(temp.T / 10 ** 9)

        emission = inputs['carbon_emission'].loc[self.year, :]
        temp = consumption_energy * emission
        output['Emission (MtCO2)'] = temp.sum() / 10 ** 3
        temp.index = temp.index.map(lambda x: 'Emission {} (MtCO2)'.format(x))
        output.update(temp.T / 10 ** 3)

        if consumption_energy_climate is not None:
            temp = consumption_energy_climate * emission
            output['Emission climate (MtCO2)'] = temp.sum() / 10 ** 3
            temp.index = temp.index.map(lambda x: 'Emission climate {} (MtCO2)'.format(x))
            output.update(temp.T / 10 ** 3)

        emission_reindex = emission.reindex(self.energy).set_axis(self.stock.index, axis=0)
        coefficient_heater = reindex_mi(self.coefficient_heater, consumption_calib.index)
        emission_stock = consumption_calib * coefficient_heater * emission_reindex
        emission_stock += consumption_calib * (1 - coefficient_heater) * emission.loc['Wood fuel']

        temp = emission_stock.groupby('Existing').sum()
        temp.rename(index={True: 'Existing', False: 'New'}, inplace=True)
        temp.index = temp.index.map(lambda x: 'Emission {} (MtCO2)'.format(x))
        output.update(temp.T / 10 ** 12)

        output['Rebound (TWh)'], output['Cost rebound (Billion euro)'] = None, None
        if self.rebound is not None:
            temp = self.rebound / 10**9
            output['Rebound (TWh)'] = temp.sum()
            temp.index = temp.index.map(lambda x: 'Rebound {} (TWh)'.format(x))
            output.update(temp.T)
            output['Cost rebound (Billion euro)'] = self.cost_rebound.sum() / 10**9

        output['Energy poverty (Million)'] = self.energy_poverty_save / 10 ** 6

        temp = self.stock.groupby(self.certificate).sum()
        temp.index = temp.index.map(lambda x: 'Stock {} (Million)'.format(x))
        output.update(temp.T / 10 ** 6)

        output['Stock efficient (Million)'] = 0
        if 'Stock A (Million)' in output.keys():
            output['Stock efficient (Million)'] += output['Stock A (Million)']
        if 'Stock B (Million)' in output.keys():
            output['Stock efficient (Million)'] += output['Stock B (Million)']

        output['Stock low-efficient (Million)'] = 0
        if 'Stock F (Million)' in output.keys():
            output['Stock low-efficient (Million)'] += output['Stock F (Million)']
        if 'Stock G (Million)' in output.keys():
            output['Stock low-efficient (Million)'] += output['Stock G (Million)']

        temp = self.stock.groupby('Heating system').sum()

        output['Stock Electricity (Million)'] = temp['Electricity-Performance boiler'] / 10**6
        output['Stock Heat pump (Million)'] = temp['Electricity-Heat pump water'] / 10**6
        if 'Electricity-Heat pump air' in output.keys():
            output['Stock Heat pump (Million)'] += temp['Electricity-Heat pump air'] / 10**6
        output['Stock Oil fuel (Million)'] = temp[['Oil fuel-Performance boiler', 'Oil fuel-Standard boiler']].sum()/ 10**6
        output['Stock Wood fuel (Million)'] = temp[['Wood fuel-Performance boiler', 'Wood fuel-Standard boiler']].sum()/ 10**6
        output['Stock Natural gas (Million)'] = temp[['Natural gas-Performance boiler', 'Natural gas-Standard boiler']].sum() / 10**6

        if self.year > self.first_year:

            # consumption saving
            if self.consumption_before_retrofit is not None:
                # do not consider coefficient
                consumption_before_retrofit = self.consumption_before_retrofit
                consumption_after_retrofit = self.store_consumption(prices)
                temp = {'{} saving (TWh/year)'.format(k.split(' (TWh)')[0]): consumption_before_retrofit[k] - consumption_after_retrofit[k]
                        for k in consumption_before_retrofit.keys()}
                output.update(temp)

            output.update({'Consumption standard saving insulation (TWh/year)': self._renovation_store['consumption_saved'].sum().sum() / 10**9})
            output.update({'Consumption standard saving insulation (%)': self._renovation_store['consumption_saved_mean']})
            temp = self._renovation_store['consumption_saved_investor']
            temp.index = temp.index.map(lambda x: 'Consumption standard saving {} - {} (%)'.format(x[0], x[1]))
            output.update(temp.T)

            # retrofit and renovation
            renovation = self._renovation_store['replacement'].sum().sum()
            output['Retrofit (Thousand households)'] = (renovation + self._renovation_store['only_heater']) / 10 ** 3 / step
            output['Renovation (Thousand households)'] = renovation / 10 ** 3 / step
            output['Renovation with heater replacement (Thousand households)'] = self._renovation_store['renovation_with_heater'] / 10 ** 3 / step
            output['Switch heater only (Thousand households)'] = self._renovation_store['only_heater'] / 10 ** 3 / step

            temp = self._renovation_store['nb_measures'].copy()
            output['Replacement total (Thousand)'] = (temp * temp.index).sum() / 10 ** 3 / step
            output['Replacement total (Thousand renovating)'] = output['Replacement total (Thousand)'] - output['Switch heater only (Thousand households)']
            temp.index = temp.index.map(lambda x: 'Retrofit measures {} (Thousand households)'.format(x))
            output.update(temp.T / 10 ** 3)

            temp = self._renovation_store['certificate_jump_all'].sum().squeeze().sort_index()
            temp = temp[temp.index.dropna()]
            o = {}
            for i in temp.index.union(self._heater_store['certificate_jump'].index):
                if i > 0:
                    t_renovation = 0
                    if i in temp.index:
                        t_renovation = temp.loc[i]
                        # o['Renovation {} EPC (Thousand households)'.format(i)] = t_renovation / 10 ** 3
                    t_heater = 0
                    if i in self._heater_store['certificate_jump'].index:
                        t_heater = self._heater_store['certificate_jump'].loc[i]
                    o['Retrofit {} EPC (Thousand households)'.format(i)] = (t_renovation + t_heater) / 10 ** 3 / step
            o = Series(o).sort_index(ascending=False)

            output['Retrofit at least 1 EPC (Thousand households)'] = sum([o['Retrofit {} EPC (Thousand households)'.format(i)] for i in temp.index.unique() if i >= 1]) / step
            output['Retrofit at least 2 EPC (Thousand households)'] = sum([o['Retrofit {} EPC (Thousand households)'.format(i)] for i in temp.index.unique() if i >= 2]) / step
            output.update(o.T / step)

            output['Global renovation high income (Thousand households)'] = self._renovation_store['global_renovation_high_income'] / 10 ** 3 / step
            output['Global renovation low income (Thousand households)'] = self._renovation_store['global_renovation_low_income'] / 10 ** 3 / step
            output['Global renovation (Thousand households)'] = output['Global renovation high income (Thousand households)'] + output['Global renovation low income (Thousand households)']
            output['Bonus best renovation (Thousand households)'] = self._renovation_store['bonus_best'] / 10 ** 3 / step
            output['Bonus worst renovation (Thousand households)'] = self._renovation_store['bonus_worst'] / 10 ** 3 / step
            if output['Renovation (Thousand households)'] != 0:
                output['Percentage of global renovation (% households)'] = output['Global renovation (Thousand households)'] / output[
                    'Renovation (Thousand households)']
                output['Percentage of bonus best renovation (% households)'] = output['Bonus best renovation (Thousand households)'] / output[
                    'Renovation (Thousand households)']
                output['Percentage of bonus worst renovation (% households)'] = output['Bonus worst renovation (Thousand households)'] / output[
                    'Renovation (Thousand households)']

            """temp = self._renovation_store['certificate_jump_all'].sum(axis=1)
            t = temp.groupby('Income owner').sum().loc[resources_data['index']['Income owner']]
            t.index = t.index.map(lambda x: 'Renovation {} (Thousand households)'.format(x))
            output.update(t.T / 10 ** 3 / step)"""

            # switch heater
            temp = self._heater_store['replacement'].sum()
            output['Switch heater (Thousand households)'] = temp.sum() / 10 ** 3 / step
            heat_pump = ['Electricity-Heat pump water', 'Electricity-Heat pump air']
            output['Switch Heat pump (Thousand households)'] = temp[heat_pump].sum() / 10 ** 3 / step
            temp.index = temp.index.map(lambda x: 'Switch {} (Thousand households)'.format(x))
            output.update((temp / 10 ** 3 / step).T)

            # insulation
            # who is renovating ?
            temp = self._renovation_store['replacement'].sum(axis=1)
            output['Renovation (Thousand households)'] = temp.sum() / 10 ** 3
            t = temp.groupby(['Housing type']).sum()
            t.index = t.index.map(lambda x: 'Renovation {} (Thousand households)'.format(x))
            output.update((t / 10 ** 3 / step).T)

            t = temp.groupby(['Housing type', 'Occupancy status']).sum()
            t.index = t.index.map(lambda x: 'Renovation {} - {} (Thousand households)'.format(x[0], x[1]))
            output.update((t / 10 ** 3 / step).T)
            """t = temp.groupby('Income owner').sum().loc[resources_data['index']['Income owner']]
            t.index = t.index.map(lambda x: 'Renovation {} (Thousand households)'.format(x))
            output.update(t.T / 10 ** 3 / step)"""

            # what is renovated ?
            o = {}
            for i in ['Wall', 'Floor', 'Roof', 'Windows']:
                temp = self._renovation_store['replacement'].xs(True, level=i, axis=1).sum(axis=1)
                o['Replacement {} (Thousand households)'.format(i)] = temp.sum() / 10 ** 3 / step

                cost = self._renovation_store['cost_component'].loc[:, i]
                t = reindex_mi(cost, temp.index) * temp
                surface = reindex_mi(inputs['surface'].loc[:, self.year], t.index)
                o['Investment {} (Billion euro)'.format(i)] = (t * surface).sum() / 10 ** 9 / step

                surface = reindex_mi(inputs['surface'].loc[:, self.year], temp.index)
                o['Embodied energy {} (TWh PE)'.format(i)] = (temp * surface *
                                                                   inputs['embodied_energy_renovation'][
                                                                       i]).sum() / 10 ** 9 / step
                o['Carbon footprint {} (MtCO2)'.format(i)] = (temp * surface *
                                                                   inputs['carbon_footprint_renovation'][
                                                                       i]).sum() / 10 ** 9 / step
            output['Replacement insulation (Thousand)'] = sum([o['Replacement {} (Thousand households)'.format(i)] for i in ['Wall', 'Floor', 'Roof', 'Windows']])
            output['Replacement insulation average (/household)'] = output['Replacement insulation (Thousand)'] / output['Renovation (Thousand households)']

            o = Series(o).sort_index(ascending=False)
            output.update(o.T)

            # economic output global
            temp = self._heater_store['cost'].sum()
            output['Investment heater (Billion euro)'] = temp.sum() / 10 ** 9 / step
            temp.index = temp.index.map(lambda x: 'Investment {} (Billion euro)'.format(x))
            output.update(temp.T / 10 ** 9 / step)
            investment_heater = self._heater_store['cost'].sum(axis=1)

            output['Financing heater (Billion euro)'] = self._heater_store['cost_financing'].sum().sum() / 10 ** 9 / step

            investment_insulation = self._renovation_store['cost'].sum(axis=1)
            output['Investment insulation (Billion euro)'] = investment_insulation.sum() / 10 ** 9 / step
            output['Financing insulation (Billion euro)'] = self._renovation_store['cost_financing'].sum().sum() / 10 ** 9 / step

            annuities = self._renovation_store['annuities']
            output['Annuities insulation (Billion euro/year)'] = annuities.sum().sum() / 10 ** 9 / step
            output['Efficiency insulation (euro/kWh standard)'] = output['Annuities insulation (Billion euro/year)'] / output['Consumption standard saving insulation (TWh/year)']

            index = investment_heater.index.union(investment_insulation.index)
            investment_total = investment_heater.reindex(index, fill_value=0) + investment_insulation.reindex(index,
                                                                                                              fill_value=0)
            output['Investment total (Billion euro)'] = investment_total.sum() / 10 ** 9 / step
            output['Financing total (Billion euro)'] = output['Financing insulation (Billion euro)'] + output['Financing heater (Billion euro)']

            """temp = investment_total.groupby('Income owner').sum().loc[resources_data['index']['Income owner']]
            temp.index = temp.index.map(lambda x: 'Investment total {} (Billion euro)'.format(x))
            output.update(temp.T / 10 ** 9 / step)"""

            subsidies_heater = self._heater_store['subsidies'].sum(axis=1)
            output['Subsidies heater (Billion euro)'] = subsidies_heater.sum() / 10 ** 9 / step

            subsidies_insulation = self._renovation_store['subsidies'].sum(axis=1)
            output['Subsidies insulation (Billion euro)'] = subsidies_insulation.sum() / 10 ** 9 / step

            index = subsidies_heater.index.union(subsidies_insulation.index)
            subsidies_total = subsidies_heater.reindex(index, fill_value=0) + subsidies_insulation.reindex(index,
                                                                                                           fill_value=0)
            output['Subsidies total (Billion euro)'] = subsidies_total.sum() / 10 ** 9 / step
            output['Lever insulation (%)'] = output['Investment insulation (Billion euro)'] / output['Subsidies insulation (Billion euro)']
            """temp = subsidies_total.groupby('Income owner').sum().loc[resources_data['index']['Income owner']]
            temp.index = temp.index.map(lambda x: 'Subsidies total {} (Billion euro)'.format(x))
            output.update(temp.T / 10 ** 9 / step)"""

            # financing - how households finance renovation ?
            temp = self._renovation_store['debt'].loc[resources_data['index']['Income owner']]
            output['Debt insulation (Billion euro)'] = temp.sum() / 10 ** 9 / step
            """temp.index = temp.index.map(lambda x: 'Debt insulation {} (Billion euro)'.format(x))
            output.update(temp.T / 10 ** 9 / step)"""

            saving = self._renovation_store['saving'].loc[resources_data['index']['Income owner']]
            output['Saving insulation (Billion euro)'] = saving.sum() / 10 ** 9 / step
            """temp.index = temp.index.map(lambda x: 'Saving insulation {} (Billion euro)'.format(x))
            output.update(temp.T / 10 ** 9 / step)"""

            # financing - how households finance new heating system
            temp = self._heater_store['debt'].loc[resources_data['index']['Income owner']]
            output['Debt heater (Billion euro)'] = temp.sum() / 10 ** 9 / step
            """temp.index = temp.index.map(lambda x: 'Debt heater {} (Billion euro )'.format(x))
            output.update(temp.T / 10 ** 9 / step)"""

            temp = self._heater_store['saving'].loc[resources_data['index']['Income owner']]
            output['Saving heater (Billion euro)'] = temp.sum() / 10 ** 9 / step
            """"temp.index = temp.index.map(lambda x: 'Saving heater {} (Billion euro / year)'.format(x))
            output.update(temp.T / 10 ** 9 / step)"""

            output['Debt total (Billion euro)'] = output['Debt heater (Billion euro)'] + output['Debt insulation (Billion euro)']
            output['Saving total (Billion euro)'] = output['Saving heater (Billion euro)'] + output['Saving insulation (Billion euro)']

            # average
            output['Investment total (Thousand euro/household)'] = output['Investment total (Billion euro)'] * 10 ** 6 / ( output['Retrofit (Thousand households)'] * 10**3)
            output['Investment insulation (Thousand euro/household)'] = 0
            if output['Renovation (Thousand households)'] != 0:
                output['Investment insulation (Thousand euro/household)'] = output['Investment insulation (Billion euro)'] * 10**6 / (output['Renovation (Thousand households)'] * 10**3)
                output['Subsidies insulation (Thousand euro/household)'] = output['Subsidies insulation (Billion euro)'] * 10**6 / (output['Renovation (Thousand households)'] * 10**3)
                output['Saving insulation (Thousand euro/household)'] = output['Saving insulation (Billion euro)'] * 10**6 / (output['Renovation (Thousand households)'] * 10**3)
                output['Debt insulation (Thousand euro/household)'] = output['Debt insulation (Billion euro)'] * 10**6 / (output['Renovation (Thousand households)'] * 10**3)

            # economic private impact - distributive indicator
            prices_reindex = prices.reindex(self.energy).set_axis(self.stock.index, axis=0)
            coefficient_heater = reindex_mi(self.coefficient_heater, consumption_calib.index)
            energy_expenditure = consumption_calib * coefficient_heater * prices_reindex
            energy_expenditure += consumption_calib * (1 - coefficient_heater) * prices.loc['Wood fuel']

            output['Energy expenditures (Billion euro)'] = energy_expenditure.sum() / 10 ** 9 / step
            energy_expenditure = energy_expenditure.groupby('Income tenant').sum()
            temp = energy_expenditure.loc[resources_data['index']['Income tenant']]
            temp.index = temp.index.map(lambda x: 'Energy expenditures {} (Billion euro)'.format(x))
            output.update(temp.T / 10 ** 9 / step)

            # do not consider the impact of heating system investment
            temp = Series({self.year: output['Annuities insulation (Billion euro/year)']})
            if self._annuities_store['stock_annuities'].empty:
                self._annuities_store['stock_annuities'] = temp
            else:
                self._annuities_store['stock_annuities'] = concat((self._annuities_store['stock_annuities'], temp))

            yrs = [y for y in self._annuities_store['stock_annuities'].index if y > self.year - 10]

            output['Stock annuities (Billion euro/year)'] = self._annuities_store['stock_annuities'].loc[yrs].sum()

            annuities = annuities.groupby(
                [c for c in annuities.index.names if c != 'Heater replacement']).sum()
            annuities = self.add_level(annuities, self.stock_mobile, 'Income tenant').sum(axis=1)
            annuities = annuities.groupby(['Occupancy status', 'Income owner', 'Income tenant']).sum()
            coefficient = pd.Series([1, 0.5, 0.5],
                                    index=pd.Index(['Owner-occupied', 'Privately rented', 'Social-housing'],
                                                   name='Occupancy status'))
            investment = (annuities * reindex_mi(coefficient, annuities.index)).groupby('Income owner').sum()
            temp = investment.loc[resources_data['index']['Income owner']].rename(self.year)
            if self._annuities_store['investment_owner'].empty:
                self._annuities_store['investment_owner'] = temp.to_frame()
            else:
                self._annuities_store['investment_owner'] = concat((self._annuities_store['investment_owner'], temp),
                                                                   axis=1)
            investment = self._annuities_store['investment_owner'].loc[:, yrs].sum(axis=1)

            rent = (annuities * reindex_mi(1 - coefficient, annuities.index)).groupby('Income tenant').sum()
            temp = rent.loc[resources_data['index']['Income tenant']].rename(self.year)
            if self._annuities_store['rent_tenant'].empty:
                self._annuities_store['rent_tenant'] = temp.to_frame()
            else:
                self._annuities_store['rent_tenant'] = concat((self._annuities_store['rent_tenant'], temp),
                                                              axis=1)
            rent = self._annuities_store['rent_tenant'].loc[:, yrs].sum(axis=1)

            expense = energy_expenditure + investment + rent

            investment.index = investment.index.map(lambda x: 'Investment owner {} (Billion euro / year)'.format(x))
            output.update(investment.T / 10 ** 9 / step)
            rent.index = rent.index.map(lambda x: 'Rent tenant {} (Billion euro / year)'.format(x))
            output.update(rent.T / 10 ** 9 / step)

            households = self.stock.groupby('Income tenant').sum()
            income = households * self._income
            temp = (expense / income).loc[resources_data['index']['Income tenant']]
            temp.index = temp.index.map(lambda x: 'Budget share {} (%)'.format(x))
            output.update(temp.T)

            # assert round(investment.sum() + rent.sum(), 0) == round(annuities.sum(), 0), 'Sum problem'

            # economic state impact
            output['VTA heater (Billion euro)'] = self._heater_store['vta'] / 10 ** 9 / step
            output['VTA insulation (Billion euro)'] = self._renovation_store['vta'] / 10 ** 9 / step
            output['VTA (Billion euro)'] = output['VTA heater (Billion euro)'] + output['VTA insulation (Billion euro)']

            output['Investment total WT (Billion euro)'] = output['Investment total (Billion euro)'] - output['VTA (Billion euro)']
            output['Investment total WT / households (Thousand euro)'] = output['Investment total WT (Billion euro)'] * 10**6 / (output['Retrofit (Thousand households)'] * 10**3)

            if taxes is not None:
                taxes_expenditures = dict()
                total_taxes = Series(0, index=prices.index)
                for tax in taxes:
                    if self.year in tax.value.index:
                        if tax.name not in self.taxes_list:
                            self.taxes_list += [tax.name]
                        amount = tax.value.loc[self.year, :] * consumption_energy
                        taxes_expenditures[tax.name] = amount
                        total_taxes += amount

                taxes_expenditures = DataFrame(taxes_expenditures).sum()
                taxes_expenditures.index = taxes_expenditures.index.map(
                    lambda x: '{} (Billion euro)'.format(x.capitalize().replace('_', ' ').replace('Cee', 'Cee tax')))
                output.update((taxes_expenditures / step).T)
                output['Taxes expenditure (Billion euro)'] = taxes_expenditures.sum() / step

            output['Income state (Billion euro)'] = output['VTA (Billion euro)'] + output[
                'Taxes expenditure (Billion euro)']
            output['Expenditure state (Billion euro)'] = output['Subsidies heater (Billion euro)'] + output[
                'Subsidies insulation (Billion euro)']
            output['Balance state (Billion euro)'] = output['Income state (Billion euro)'] - output[
                'Expenditure state (Billion euro)']

            # subsidies - description
            temp = subsidies_total.groupby('Income owner').sum() / investment_total.groupby('Income owner').sum()
            temp = temp.loc[resources_data['index']['Income owner']]
            temp.index = temp.index.map(lambda x: 'Share subsidies {} (%)'.format(x))
            output.update(temp.T)



            # TODO: NOT IMPLEMENTED YET

            if self.consumption_saving_insulation is not None:
                output['Consumption saving insulation (TWh)'] = self.consumption_saving_insulation / 10**9 / step
            else:
                output['Consumption saving insulation (TWh)'] = None

            if self.consumption_saving_insulation is not None:
                output['Consumption saving heater (TWh)'] = self.consumption_saving_heater / 10 ** 9 / step
            else:
                output['Consumption saving heater (TWh)'] = None




            if output['Consumption saving insulation (TWh)'] is not None:
                if output['Consumption saving insulation (TWh)'] != 0:
                    investment = calculate_annuities(output['Investment insulation (Billion euro)'])
                    output['Investment insulation (euro/year)'] = investment
                    output['Investment insulation / saving (euro/kWh)'] = investment / output['Consumption saving insulation (TWh)']

                    investment = calculate_annuities(output['Investment insulation (Billion euro)'], lifetime=50,
                                                     discount_rate=0.032)
                    output['Investment insulation low (euro/year)'] = investment
                    output['Investment insulation / saving low (euro/kWh)'] = investment / output['Consumption saving insulation (TWh)']

            if output['Consumption saving heater (TWh)'] is not None:
                if output['Consumption saving heater (TWh)'] != 0:
                    investment = calculate_annuities(output['Investment heater (Billion euro)'], lifetime=20)
                    output['Investment heater (euro/year)'] = investment
                    output['Investment heater / saving (euro/kWh)'] = investment / output['Consumption saving heater (TWh)']

                    investment = calculate_annuities(output['Investment insulation (Billion euro)'], lifetime=20,
                                                     discount_rate=0.032)
                    output['Investment heater low (euro/year)'] = investment
                    output['Investment heater / saving low (euro/kWh)'] = investment / output['Consumption saving heater (TWh)']

            # co-benefit
            if 'Embodied energy Wall (TWh PE)' in output.keys():
                output['Embodied energy renovation (TWh PE)'] = output['Embodied energy Wall (TWh PE)'] + output[
                    'Embodied energy Floor (TWh PE)'] + output['Embodied energy Roof (TWh PE)'] + output[
                                                                    'Embodied energy Windows (TWh PE)']

                output['Embodied energy construction (TWh PE)'] = inputs['Embodied energy construction (TWh PE)'].loc[
                    self.year]
                output['Embodied energy (TWh PE)'] = output['Embodied energy renovation (TWh PE)'] + output[
                    'Embodied energy construction (TWh PE)']

                output['Carbon footprint renovation (MtCO2)'] = output['Carbon footprint Wall (MtCO2)'] + output[
                    'Carbon footprint Floor (MtCO2)'] + output['Carbon footprint Roof (MtCO2)'] + output[
                                                                    'Carbon footprint Windows (MtCO2)']

                output['Carbon footprint construction (MtCO2)'] = inputs['Carbon footprint construction (MtCO2)'].loc[
                    self.year]
                output['Carbon footprint (MtCO2)'] = output['Carbon footprint renovation (MtCO2)'] + output[
                    'Carbon footprint construction (MtCO2)']
                output['Carbon value indirect (Billion euro)'] = output['Carbon footprint (MtCO2)'] * \
                                                                 inputs['carbon_value'].loc[self.year] / 10 ** 3 / step

            output['Carbon value (Billion euro)'] = (consumption_energy * inputs['carbon_value_kwh'].loc[self.year, :]).sum() / step

            output['Health cost (Billion euro)'], o = self.health_cost(inputs)
            output['Health cost (Billion euro)'] /= step
            # TODO need to be divide by step
            output.update(o)

            # subsidies - details
            # policies amount and number of beneficiaries
            subsidies, subsidies_count, sub_count = None, None, None
            for gest, subsidies_details in {'heater': self._heater_store['subsidies_details'],
                                            'insulation': self._renovation_store['subsidies_details']}.items():
                if gest == 'heater':
                    sub_count = DataFrame(self._heater_store['subsidies_count'], dtype=float)
                elif gest == 'insulation':
                    sub_count = DataFrame(self._renovation_store['subsidies_count'], dtype=float)

                subsidies_details = Series({k: i.sum().sum() for k, i in subsidies_details.items()}, dtype='float64')

                if detailed_output is True:
                    for i in subsidies_details.index:
                        temp = sub_count[i]
                        temp.index = temp.index.map(lambda x: '{} {} {} (Thousand households)'.format(i.capitalize().replace('_', ' '), gest, x))
                        output.update(temp.T / 10 ** 3 / step)
                        output['{} {} (Billion euro)'.format(i.capitalize().replace('_', ' '), gest)] = \
                            subsidies_details.loc[i] / 10 ** 9 / step
                if subsidies is None:
                    subsidies = subsidies_details.copy()
                    subsidies_count = sub_count.copy()
                else:
                    subsidies = concat((subsidies, subsidies_details), axis=0)
                    subsidies_count = concat((subsidies_count, sub_count))

                subsidies = subsidies.groupby(subsidies.index).sum()
                subsidies_count = subsidies_count.groupby(subsidies_count.index).sum()
                for i in subsidies.index:
                    temp = subsidies_count[i]
                    output['{} (Thousand households)'.format(i.capitalize().replace('_', ' '))] = temp.sum() / 10**3 / step
                    temp.index = temp.index.map(lambda x: '{} {} (Thousand households)'.format(i.capitalize().replace('_', ' '), x))
                    output.update(temp.T / 10 ** 3)
                    output['{} (Billion euro)'.format(i.capitalize().replace('_', ' '))] = subsidies.loc[i] / 10 ** 9 / step


            if detailed_output is True:

                t = self._heater_store['replacement'].groupby('Housing type').sum().loc['Multi-family']
                t.index = t.index.map(lambda x: 'Switch Multi-family {} (Thousand households)'.format(x))
                output.update((t / 10 ** 3 / step).T)

                t = self._heater_store['replacement'].groupby('Housing type').sum().loc['Single-family']
                t.index = t.index.map(lambda x: 'Switch Single-family {} (Thousand households)'.format(x))
                output.update((t / 10 ** 3 / step).T)

                temp = investment_total.groupby(['Housing type', 'Occupancy status']).sum()
                temp.index = temp.index.map(lambda x: 'Investment total {} - {} (Billion euro)'.format(x[0], x[1]))
                output.update(temp.T / 10 ** 9 / step)

                temp = subsidies_total.groupby(['Housing type', 'Occupancy status']).sum()
                temp.index = temp.index.map(lambda x: 'Subsidies total {} - {} (Million euro)'.format(x[0], x[1]))
                output.update(temp.T / 10 ** 6 / step)

        output = Series(output).rename(self.year)
        stock = stock.rename(self.year)
        return stock, output

    def apply_scale(self, scale):

        self.scale *= scale
        self.preferences_insulation['subsidy'] *= scale
        self.preferences_insulation['cost'] *= scale
        self.preferences_insulation['bill_saved'] *= scale

    def calibration_exogenous(self, coefficient_global=None, coefficient_heater=None, constant_heater=None,
                              constant_insulation_intensive=None, constant_insulation_extensive=None, scale=None,
                              energy_prices=None, taxes=None, threshold_indicator=None):
        """Function calibrating buildings object with exogenous data.


        Parameters
        ----------
        coefficient_global: float
        coefficient_heater: Series
        constant_heater: Series
        constant_insulation_intensive: Series
        constant_insulation_extensive: Series
        scale: float
        energy_prices: Series
            Energy prices for year y. Index are energy carriers {'Electricity', 'Natural gas', 'Oil fuel', 'Wood fuel'}.
        taxes: Series
            Energy taxes for year y.
        threshold_indicator:
        """

        # calibration energy consumption first year
        if (coefficient_global is None) and (energy_prices is not None) and (taxes is not None):
            self.calibration_consumption(energy_prices.loc[self.first_year, :], taxes)
        else:
            self.coefficient_global = coefficient_global
            self.coefficient_heater = coefficient_heater

        if constant_heater is not None:
            self.constant_heater = constant_heater

        if constant_insulation_intensive is not None:
            self.constant_insulation_intensive = constant_insulation_intensive.dropna()

        if constant_insulation_extensive is not None:
            self.constant_insulation_extensive = constant_insulation_extensive.dropna()

        self.scale = scale
        self.apply_scale(scale)
        self.threshold_indicator = threshold_indicator

    def remove_calibration(self):

        self.coefficient_global = None
        self.coefficient_heater = None

        self.preferences_insulation['subsidy'] /= self.scale
        self.preferences_insulation['cost'] /= self.scale
        self.preferences_insulation['bill_saved'] /= self.scale

        self.constant_heater = None
        self.constant_insulation_intensive = None
        self.constant_insulation_extensive = None
        self.scale = None

    def flow_demolition(self, step=1):
        """Demolition of E, F and G buildings based on their share in the mobile stock.

        Returns
        -------
        Series
        """
        self.logger.info('Demolition')
        stock_demolition = self.stock_mobile[self.certificate.isin(self._target_demolition)]
        demolition_total = self._demolition_total * step

        if stock_demolition.sum() < demolition_total:
            self._target_demolition = ['G', 'F', 'E', 'D']
            stock_demolition = self.stock_mobile[self.certificate.isin(self._target_demolition)]
            if stock_demolition.sum() < demolition_total:
                self._target_demolition = ['G', 'F', 'E', 'D', 'C']
                stock_demolition = self.stock_mobile[self.certificate.isin(self._target_demolition)]

        stock_demolition = stock_demolition / stock_demolition.sum()
        flow_demolition = (stock_demolition * demolition_total).dropna()
        return flow_demolition.reorder_levels(self.stock.index.names)

    def health_cost(self, param, stock=None):
        if stock is None:
            stock = self.simplified_stock()

        health_cost_type = {'health_expenditure': 'Health expenditure (Billion euro)',
                            'mortality_cost': 'Social cost of mortality (Billion euro)',
                            'loss_well_being': 'Loss of well-being (Billion euro)'}
        health_cost = dict()
        for key, item in health_cost_type.items():
            health_cost[item] = (stock * reindex_mi(param[key], stock.index)).sum() / 10 ** 9

        health_cost_total = Series(health_cost).sum()
        return health_cost_total, health_cost

    def mitigation_potential(self, prices, cost_insulation_raw, carbon_emission=None, carbon_value=None, health_cost=None,
                             index=None):
        """Function returns bill saved and cost for buildings stock retrofit.

        Not implemented yet but should be able to calculate private and social indicator.
        Make cost abatement cost graphs, payback period graphs.

        Parameters
        ----------
        index
        prices
        cost_insulation_raw
        carbon_emission
        carbon_value
        health_cost

        Returns
        -------

        """

        output = dict()

        if index is None:
            index = self.stock.index

        consumption_before = self.consumption_heating_store(index, full_output=False)
        consumption_after, _, certificate_after = self.prepare_consumption(self._choice_insulation, index=index)
        consumption_saved = (consumption_before - consumption_after.T).T

        self._efficiency.index.names = ['Heating system']
        efficiency = reindex_mi(self._efficiency, consumption_before.index)
        need_before = consumption_before * efficiency

        consumption_before = reindex_mi(consumption_before, index)
        need_before = reindex_mi(need_before, index)
        consumption_after = reindex_mi(consumption_after, index)
        consumption_saved = reindex_mi(consumption_saved, index)
        efficiency = reindex_mi(self._efficiency, consumption_saved.index)
        need_saved = (consumption_saved.T * efficiency).T

        consumption_actual_before = self.consumption_actual(prices.loc[self.year, :], consumption_before)
        consumption_actual_after = self.consumption_actual(prices.loc[self.year, :], consumption_after)
        consumption_actual_saved = (consumption_actual_before - consumption_actual_after.T).T

        consumption_before = reindex_mi(self._surface, index) * consumption_before
        consumption_after = (reindex_mi(self._surface, index) * consumption_after.T).T
        consumption_saved = (reindex_mi(self._surface, index) * consumption_saved.T).T

        need_before = reindex_mi(self._surface, index) * need_before
        need_saved = (reindex_mi(self._surface, index) * need_saved.T).T

        consumption_actual_before = (reindex_mi(self._surface, index) * consumption_actual_before.T).T
        consumption_actual_after = (reindex_mi(self._surface, index) * consumption_actual_after.T).T
        consumption_actual_saved = (reindex_mi(self._surface, index) * consumption_actual_saved.T).T

        output.update({'Stock (dwellings/segment)': self.stock,
                       'Surface (m2/segment)': self.stock * reindex_mi(self._surface, index),
                       'Consumption before (kWh/dwelling)': consumption_before,
                       'Consumption before (kWh/segment)': consumption_before * self.stock,
                       'Need before (kWh/segment)': need_before * self.stock,
                       'Consumption actual before (kWh/dwelling)': consumption_actual_before,
                       'Consumption actual before (kWh/segment)': consumption_actual_before * self.stock,
                       'Consumption actual after (kWh/dwelling)': consumption_actual_after,
                       'Consumption actual after (kWh/segment)': (consumption_actual_after.T * self.stock).T,
                       'Consumption saved (kWh/dwelling)': consumption_saved,
                       'Consumption saved (kWh/segment)': (consumption_saved.T * self.stock).T,
                       'Need saved (kWh/segment)': (need_saved.T * self.stock).T,
                       'Consumption actual saved (kWh/dwelling)': consumption_actual_saved,
                       'Consumption actual saved (kWh/segment)': (consumption_actual_saved.T * self.stock).T
                       })

        consumption_saved_agg = (self.stock * consumption_saved.T).T
        consumption_actual_saved_agg = (self.stock * consumption_actual_saved.T).T

        if carbon_emission is not None:
            c = self.add_energy(consumption_actual_before)
            emission_before = reindex_mi(carbon_emission.T.rename_axis('Energy', axis=0), c.index).loc[:,
                              self.year] * c

            c = self.add_energy(consumption_actual_after)
            emission_after = (reindex_mi(carbon_emission.T.rename_axis('Energy', axis=0), c.index).loc[:,
                              self.year] * c.T).T

            emission_saved = - emission_after.sub(emission_before, axis=0).dropna()

            output.update({'Emission before (gCO2/dwelling)': emission_before,
                           'Emission after (gCO2/dwelling)': emission_after,
                           'Emission saved (gCO2/dwelling)': emission_saved,
                           })

            if carbon_value is not None:
                c = self.add_energy(consumption_actual_before)
                emission_value_before = reindex_mi(carbon_value.T.rename_axis('Energy', axis=0), c.index).loc[:,
                                        self.year] * c

                c = self.add_energy(consumption_actual_after)
                emission_value_after = (reindex_mi(carbon_value.T.rename_axis('Energy', axis=0), c.index).loc[:,
                                        self.year] * c.T).T

                emission_value_saved = - emission_value_after.sub(emission_value_before, axis=0).dropna()

                output.update({'Emission value before (euro/dwelling)': emission_value_before,
                               'Emission value after (euro/dwelling)': emission_value_after,
                               'Emission value saved (euro/dwelling)': emission_value_saved
                               })

        cost_insulation = self.prepare_cost_insulation(cost_insulation_raw * self.surface_insulation)
        cost_insulation = reindex_mi(cost_insulation, index)
        potential_cost_insulation = (reindex_mi(self._surface, index) * cost_insulation.T).T

        output.update({'Cost insulation (euro/dwelling)': potential_cost_insulation,
                       'Cost insulation (euro/segment)': (potential_cost_insulation.T * self.stock).T
                       })

        index = self.stock.index
        energy = pd.Series(index.get_level_values('Heating system'), index=index).str.split('-').str[0].rename('Energy')
        energy_prices = prices.loc[self.year, :].reindex(energy).set_axis(index)

        bill_before = consumption_before * energy_prices
        bill_after = (consumption_after.T * energy_prices).T
        bill_saved = - bill_after.sub(bill_before, axis=0).dropna()

        output.update({'Bill before (euro/dwelling)': bill_before,
                       'Bill after (euro/dwelling)': bill_after,
                       'Bill saved (euro/dwelling)': bill_saved
                       })

        discount_rate, lifetime = 0.05, 30
        discount_factor = (1 - (1 + discount_rate) ** -lifetime) / discount_rate
        npv = bill_saved * discount_factor - potential_cost_insulation

        out = AgentBuildings.find_best_option(npv, {'bill_saved': bill_saved,
                                                    'cost': potential_cost_insulation,
                                                    'consumption_saved': consumption_saved,
                                                    'consumption_saved_agg': consumption_saved_agg,
                                                    'consumption_actual_saved_agg': consumption_actual_saved_agg
                                                    })
        output.update({'Best NPV': out})

        out = AgentBuildings.find_best_option(consumption_saved_agg, {'bill_saved': bill_saved,
                                                                      'cost': potential_cost_insulation,
                                                                      'consumption_saved': consumption_saved,
                                                                      'consumption_saved_agg': consumption_saved_agg,
                                                                      'consumption_actual_saved_agg': consumption_actual_saved_agg})

        output.update({'Max consumption saved': out})
        return output
