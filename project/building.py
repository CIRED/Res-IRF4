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
from numpy import exp, log, zeros, ones, append, arange, array
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import logging
from copy import deepcopy

from project.utils import make_plot, format_ax, save_fig, format_legend, reindex_mi, make_plots, get_pandas
from project.input.resources import resources_data
import project.thermal as thermal

from itertools import product

ACCURACY = 10**-5


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

    def __init__(self, stock, surface, ratio_surface, efficiency, income, consumption_ini, path=None, year=2018,
                 debug_mode=False):

        self.consumption_saving_retrofit = None
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
            self.path_calibration_renovation = os.path.join(self.path_calibration, 'renovation')
            if not os.path.isdir(self.path_calibration_renovation):
                os.mkdir(self.path_calibration_renovation)

        self._consumption_ini = consumption_ini
        self.coefficient_consumption = None

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

        # TODO only heating_intensity and calculate average in parse_output
        self.energy_poverty, self.heating_intensity = None, None
        self.consumption_3uses_building, self.consumption_sd_building, self.certificate_building = Series(
            dtype='float'), Series(dtype='float'), Series(dtype='float')
        self.consumption_sd_building_choice, self.consumption_3uses_building_choice, self.certificate_building_choice = Series(
            dtype='float'), Series(dtype='float'), Series(dtype='float')

        self.heating_intensity_avg = None
        self.consumption_heat_sd = None
        self.heat_consumption = None
        self.heat_consumption_calib = None
        self.heat_consumption_energy = None
        self.taxes_expenditure = None
        self.energy_expenditure = None
        self.taxes_list = []
        self.taxes_expenditure_details = {}
        self.stock_yrs = {}

        self.stock = stock

        self.consumption_before_retrofit = None

    @property
    def year(self):
        return self._year

    @year.setter
    def year(self, year):
        self._year = year
        self._surface = self._surface_yrs.loc[:, year]

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
        # self.housing_type = Series(stock.index.get_level_values('Housing type'), index=stock.index)

        heating_system = Series(stock.index.get_level_values('Heating system'), index=stock.index)
        self.energy = heating_system.str.split('-').str[0].rename('Energy')
        self.efficiency = to_numeric(heating_system.replace(self._efficiency))

        self.stock_yrs.update({self.year: self.stock})

        consumption_sd, _, certificate = self.consumption_standard(stock.index)
        self.consumption_heat_sd = reindex_mi(consumption_sd, stock.index)
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

    def consumption_heating(self, idx=None, freq='year', climate=None, smooth=False, temp_indoor=None, unit='kWh/m2.y',
                            full_output=False, efficiency_hour=False):
        """Calculation consumption standard of the current building stock [kWh/m2.a].

        Parameters
        ----------
        idx
        freq
        climate
        smooth
        temp_indoor
        unit
        full_output: bool, default False
        efficiency_hour

        Returns
        -------

        """
        if idx is None:
            levels = ['Housing type', 'Heating system', 'Wall', 'Floor', 'Roof', 'Windows']
            idx = self.stock.groupby(levels).sum().index

        wall = Series(idx.get_level_values('Wall'), index=idx)
        floor = Series(idx.get_level_values('Floor'), index=idx)
        roof = Series(idx.get_level_values('Roof'), index=idx)
        windows = Series(idx.get_level_values('Windows'), index=idx)
        heating_system = Series(idx.get_level_values('Heating system'), index=idx).astype('object')
        efficiency = to_numeric(heating_system.replace(self._efficiency))
        consumption = thermal.conventional_heating_final(wall, floor, roof, windows, self._ratio_surface.copy(),
                                                         efficiency, climate=climate, freq=freq, smooth=smooth,
                                                         temp_indoor=temp_indoor, efficiency_hour=efficiency_hour)

        if full_output is True:
            certificate, consumption_3uses = thermal.conventional_energy_3uses(wall, floor, roof, windows,
                                                                               self._ratio_surface.copy(),
                                                                               efficiency, idx)
            return consumption, certificate, consumption_3uses
        else:
            return consumption

    def consumption_standard(self, indexes, level_heater='Heating system', unit='kWh/m2.y'):
        """Pre-calculate space energy consumption based only on relevant levels.

        Parameters
        ----------
        indexes: MultiIndex, Index
            Used to estimate consumption standard.
        level_heater: {'Heating system', 'Heating system final'}, default 'Heating system'
        unit

        Returns
        -------
        """
        levels_consumption = ['Wall', 'Floor', 'Roof', 'Windows', level_heater, 'Housing type']
        index = indexes.to_frame().loc[:, levels_consumption].set_index(levels_consumption).index
        index = index[~index.duplicated()]

        index.rename({level_heater: 'Heating system'}, inplace=True)
        # remove index already calculated
        if not self.consumption_sd_building.empty:
            temp = self.consumption_sd_building.index.intersection(index)
            idx = index.drop(temp)
        else:
            idx = index

        if not idx.empty:

            consumption, certificate, consumption_3uses = self.consumption_heating(idx=idx, freq='year', climate=None,
                                                                                   full_output=True)

            self.consumption_sd_building = concat((self.consumption_sd_building, consumption))
            self.consumption_sd_building.index = MultiIndex.from_tuples(
                self.consumption_sd_building.index).set_names(consumption.index.names)
            self.consumption_3uses_building = concat((self.consumption_3uses_building, consumption_3uses))
            self.consumption_3uses_building.index = MultiIndex.from_tuples(
                self.consumption_3uses_building.index).set_names(consumption.index.names)
            self.certificate_building = concat((self.certificate_building, certificate))
            self.certificate_building.index = MultiIndex.from_tuples(
                self.certificate_building.index).set_names(consumption.index.names)

        levels_consumption = [i for i in indexes.names if i in levels_consumption]

        consumption_sd = self.consumption_sd_building.loc[index]
        consumption_sd.index.rename({'Heating system': level_heater}, inplace=True)
        consumption_sd = consumption_sd.reorder_levels(levels_consumption)
        consumption_3uses = self.consumption_3uses_building.loc[index]
        consumption_3uses.index.rename({'Heating system': level_heater}, inplace=True)
        consumption_3uses = consumption_3uses.reorder_levels(levels_consumption)
        certificate = self.certificate_building.loc[index]
        certificate.index.rename({'Heating system': level_heater}, inplace=True)
        certificate = certificate.reorder_levels(levels_consumption)

        return consumption_sd, consumption_3uses, certificate

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
            consumption = self.consumption_heat_sd.copy() * self.surface
        else:
            consumption = consumption.copy()

        energy_bill = AgentBuildings.energy_bill(prices, consumption)
        if isinstance(energy_bill, Series):
            budget_share = energy_bill / reindex_mi(self._income_tenant, self.stock.index)
            heating_intensity = thermal.heat_intensity(budget_share)
            consumption *= heating_intensity
            if store:
                self.heating_intensity = heating_intensity
                self.energy_poverty = (self.stock[self.stock.index.get_level_values(
                    'Income owner') == ('D1' or 'D2' or 'D3')])[budget_share >= 0.08].sum()
        elif isinstance(energy_bill, DataFrame):
            budget_share = (energy_bill.T / reindex_mi(self._income_tenant, self.stock.index)).T
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
                if energy is False:
                    return consumption.sum() / 10 ** 9
                else:
                    energy = self.energy.reindex(consumption.index)
                    return consumption.groupby(energy).sum() / 10**9

            if freq == 'hour':
                temp = self.consumption_heating(freq=freq, climate=climate, smooth=smooth, efficiency_hour=efficiency_hour)
                temp = reindex_mi(temp, self.stock.index)
                t = (temp.T * self.stock * self.surface).T
                # adding heating intensity
                t = (t.T * self.heating_intensity).T
                energy = temp.index.get_level_values('Heating system').str.split('-').str[0]
                t = t.groupby(energy).sum()
                t = (t.T * self.coefficient_consumption).T
                return t

    def calculate_consumption(self, prices, taxes=None, climate=None, temp_indoor=None):
        """Calculate energy indicators.

        Parameters
        ----------
        prices: Series
        taxes: Series
        climate
        temp_indoor

        Returns
        -------

        """

        consumption = self.consumption_heating(climate=climate, temp_indoor=temp_indoor)
        consumption = reindex_mi(consumption, self.stock.index) * self.surface

        _consumption_actual = self.consumption_actual(prices, consumption=consumption, store=True) * self.stock

        consumption_energy = _consumption_actual.groupby(self.energy).sum()
        if self.coefficient_consumption is None:

            consumption = concat((_consumption_actual, self.energy), axis=1).groupby(
                ['Housing type', 'Energy']).sum().iloc[:, 0] / 10**9

            # considering 20% of electricity got wood stove - 50% electricity
            electricity_wood = 0.2 * consumption[('Single-family', 'Electricity')] * 1
            consumption[('Single-family', 'Wood fuel')] += electricity_wood
            consumption[('Single-family', 'Electricity')] -= electricity_wood
            consumption.groupby('Energy').sum()

            _consumption_actual.groupby('Housing type').sum() / 10**9

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

            self.coefficient_consumption = self._consumption_ini * 10**9 / consumption_energy

            temp = self.coefficient_consumption.copy()
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

        coefficient = self.coefficient_consumption.reindex(self.energy).set_axis(self.stock.index, axis=0)
        self.heat_consumption_calib = (coefficient * _consumption_actual).copy()

        self.heat_consumption_energy = self.heat_consumption_calib.groupby(self.energy).sum()

        prices_reindex = prices.reindex(self.energy).set_axis(self.stock.index, axis=0)
        self.energy_expenditure = prices_reindex * self.heat_consumption_calib

        if taxes is not None:
            total_taxes = Series(0, index=prices.index)
            for tax in taxes:
                if self.year in tax.value.index:
                    if tax.name not in self.taxes_list:
                        self.taxes_list += [tax.name]
                    amount = tax.value.loc[self.year, :] * consumption_energy
                    self.taxes_expenditure_details[tax.name] = amount
                    total_taxes += amount
            self.taxes_expenditure = total_taxes

        if self.consumption_before_retrofit is not None:
            consumption_before_retrofit = self.consumption_before_retrofit
            self.consumption_before_retrofit = None
            consumption_after_retrofit = self.store_consumption(prices)
            self.consumption_saving_retrofit = {k: consumption_before_retrofit[k] - consumption_after_retrofit[k] for k
                                                in consumption_before_retrofit.keys()}
            # TODO: quantify rebound effect

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
        temp.index = temp.index.map(lambda x: 'Consumption standard {} (TWh)'.format(x))
        output.update(temp)

        temp = self.consumption_total(prices=prices, freq='year', standard=False, climate=None, smooth=False,
                                      existing=True, energy=True)
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

    def __init__(self, stock, surface, ratio_surface, efficiency, income, consumption_ini, preferences,
                 performance_insulation, path=None, year=2018, demolition_rate=0.0,
                 endogenous=True, exogenous=None, insulation_representative='market_share',
                 logger=None, debug_mode=False, calib_scale=True, full_output=None,
                 quintiles=None, financing_cost=True,
                 ):
        super().__init__(stock, surface, ratio_surface, efficiency, income, consumption_ini, path=path, year=year,
                         debug_mode=debug_mode)
        self.best_option = None
        self.constant_test = None
        self.certif_jump_all = None
        self.in_global_renovation_low_income = None
        self.in_global_renovation_high_income = None
        self.certificate_jump_heater = None
        self.global_renovation = None
        self.financing_cost = financing_cost
        self.subsidies_count_insulation, self.subsidies_average_insulation = dict(), dict()
        self.subsidies_count_heater, self.subsidies_average_heater = dict(), dict()

        self.prepared_cost_insulation = None
        self.certif_jump_all = None
        self.retrofit_with_heater = None
        self._calib_scale = calib_scale
        self.vta = 0.1
        self.lifetime_insulation = 30
        self._epc2int = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}

        self.quintiles = quintiles
        if full_output is None:
            full_output = True
        self.full_output = full_output

        if logger is None:
            logger = logging.getLogger()
        self.logger = logger
        self.policies = []

        # {'max', 'market_share'} define how to calculate utility_extensive
        self._insulation_representative = insulation_representative

        self.preferences_heater = deepcopy(preferences['heater'])
        self.preferences_insulation_int = deepcopy(preferences['insulation'])
        self.preferences_insulation_ext = deepcopy(preferences['insulation'])

        def monetary_unit(pref_dict):
            for key in pref_dict.keys():
                if key != 'investment':
                    pref_dict[key] = pref_dict[key] / abs(pref_dict['investment'])
            pref_dict['investment'] = -1

        # monetary_unit(self.preferences_heater)
        # monetary_unit(self.preferences_insulation_int)
        # monetary_unit(self.preferences_insulation_ext)


        # self.discount_rate = - self.pref_investment_insulation_ext / self.pref_bill_insulation_ext
        # self.discount_factor = (1 - (1 + self.discount_rate) ** -self.lifetime_insulation) / self.discount_rate

        self.scale = 1.0
        self.calibration_scale = 'cite'
        self.param_supply = None
        self.capacity_utilization = None
        self.factor_yrs = {}

        self._demolition_rate = demolition_rate
        self._demolition_total = (stock * self._demolition_rate).sum()
        self._target_demolition = ['E', 'F', 'G']

        self._choice_heater = None
        self._probability_replacement = None

        self._endogenous, self.param_exogenous = endogenous, exogenous

        choice_insulation = {'Wall': [False, True], 'Floor': [False, True], 'Roof': [False, True],
                             'Windows': [False, True]}
        names = list(choice_insulation.keys())
        choice_insulation = list(product(*[i for i in choice_insulation.values()]))
        choice_insulation.remove((False, False, False, False))
        choice_insulation = MultiIndex.from_tuples(choice_insulation, names=names)
        self._choice_insulation = choice_insulation
        self._performance_insulation = {i: min(val, self.stock.index.get_level_values(i).min()) for i, val in
                                        performance_insulation.items()}
        # min of self.stock
        self.surface_insulation = self._ratio_surface.copy()

        self.constant_insulation_extensive, self.constant_insulation_intensive, self.constant_heater = None, None, None

        self.cost_insulation_indiv, self.subsidies_heater_indiv, self.subsidies_insulation_indiv = None, None, None
        self.subsidies_details_insulation_indiv = None

        self.certificate_jump = None
        self.gest_nb = None

        self.global_renovation_high_income, self.global_renovation_low_income = None, None
        self.in_best, self.out_worst = None, None
        self.bonus_best, self.bonus_worst = None, None
        self.market_share = None
        self.replacement_heater, self.heater_replaced = None, None
        self.cost_heater, self.investment_heater = None, None
        self.tax_heater = None
        self.subsidies_details_heater, self.subsidies_heater = None, None

        self.replacement_insulation, self.retrofit_rate = None, None
        self.cost_component, self.investment_insulation = None, None
        self.tax_insulation, self.taxed_insulation = None, None
        self.subsidies_insulation = None, None
        self.subsidies_details_insulation = {}

        self.certificate_jump_all = None

        self.zil_count, self.zil_loaned_avg, self.zil_loaned = None, None, None

        self._share_decision_maker = stock.groupby(
            ['Occupancy status', 'Housing type', 'Income owner', 'Income tenant']).sum().unstack(
            ['Occupancy status', 'Income owner', 'Income tenant'])
        self._share_decision_maker = (self._share_decision_maker.T / self._share_decision_maker.sum(axis=1)).T

    @property
    def year(self):
        return self._year

    @year.setter
    def year(self, year):
        self._year = year
        self._surface = self._surface_yrs.loc[:, year]

        self.bonus_best, self.bonus_worst = 0, 0
        self.global_renovation_high_income, self.global_renovation_low_income = 0, 0
        self.replacement_insulation = None
        self.market_share, self.retrofit_rate = None, None
        self.certificate_jump_all = None
        self.subsidies_details_insulation, self.subsidies_count_insulation, self.subsidies_count_heater = {}, {}, {}
        self.subsidies_average_heater, self.subsidies_average_insulation = {}, {}

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

    def prepare_consumption(self, choice_insulation=None, performance_insulation=None, index=None,
                            level_heater='Heating system'):
        """Calculate standard energy consumption and energy performance certificate for each choice insulation for all
        households.

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

        if index is None:
            index = self.stock.index

        if not isinstance(choice_insulation, MultiIndex):
            choice_insulation = self._choice_insulation

        if not isinstance(performance_insulation, MultiIndex):
            performance_insulation = self._performance_insulation

        # only selecting useful levels
        indx = index.copy()
        indx = indx.droplevel([i for i in indx.names if i not in ['Housing type', 'Wall', 'Floor', 'Roof', 'Windows'] + [level_heater]])
        indx = indx[~indx.duplicated()]

        # remove idx already calculated
        if not self.consumption_sd_building_choice.empty:
            temp = self.consumption_sd_building_choice.index.intersection(indx)
            idx = indx.drop(temp)
        else:
            idx = indx

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
                 level_heater: 'string'})
            index = MultiIndex.from_frame(temp)
            # consumption based on insulated components
            consumption_sd, consumption_3uses, certificate = self.consumption_standard(index, level_heater=level_heater)

            rslt = dict()
            for key, temp in {'consumption_sd': consumption_sd, 'consumption_3uses': consumption_3uses, 'certificate': certificate}.items():
                temp = reindex_mi(temp, index).droplevel(['Wall', 'Floor', 'Roof', 'Windows']).unstack(
                    ['{} bool'.format(i) for i in ['Wall', 'Floor', 'Roof', 'Windows']])
                temp.index.rename({'Wall before': 'Wall', 'Floor before': 'Floor', 'Roof before': 'Roof', 'Windows before': 'Windows'},
                                  inplace=True)
                temp.columns.rename({'Wall bool': 'Wall', 'Floor bool': 'Floor', 'Roof bool': 'Roof', 'Windows bool': 'Windows'},
                                    inplace=True)
                rslt[key] = temp

            if self.consumption_sd_building_choice.empty:
                self.consumption_sd_building_choice = rslt['consumption_sd']
                self.consumption_3uses_building_choice = rslt['consumption_3uses']
                self.certificate_building_choice = rslt['certificate']
            else:
                self.consumption_sd_building_choice = concat((self.consumption_sd_building_choice, rslt['consumption_sd']))
                self.consumption_3uses_building_choice = concat((self.consumption_3uses_building_choice, rslt['consumption_3uses']))
                self.certificate_building_choice = concat((self.certificate_building_choice, rslt['certificate']))

        consumption_sd = self.consumption_sd_building_choice.loc[indx.rename({'Heating system': level_heater})]
        consumption_sd.index.rename({'Heating system': level_heater}, inplace=True)

        primary_consumption_3uses = self.consumption_3uses_building_choice.loc[indx]
        primary_consumption_3uses.index.rename({'Heating system': level_heater}, inplace=True)

        certificate = self.certificate_building_choice.loc[indx]
        certificate.index.rename({'Heating system': level_heater}, inplace=True)

        return consumption_sd, primary_consumption_3uses, certificate

    def heater_replacement(self, stock, prices, cost_heater, policies_heater, ms_heater=None,
                           probability_replacement=1/20):
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

        if ms_heater is not None:
            self._choice_heater = list(ms_heater.columns)

        if isinstance(probability_replacement, float):
            probability_replacement = Series(len(self._choice_heater) * [probability_replacement],
                                                Index(self._choice_heater, name='Heating system final'))

        index = stock.index

        # prohibited energies can be a string or a list of strings
        energy_regulations = [policy for policy in policies_heater if policy.policy == 'heater_regulation']
        prohibited_energies = Series(list(array([policy.name.replace('_elimination', "").replace("_", " ").capitalize()
                                             for policy in energy_regulations]).flat), index=[policy.name for policy in energy_regulations],
                                        dtype=object)

        for regulation in energy_regulations:
            if regulation.value is not None:
                heater = next(x for x in self._choice_heater if prohibited_energies[regulation.name] in x)
                probability_replacement[heater] = regulation.value

        self._probability_replacement = probability_replacement

        list_heaters = self._choice_heater
        for energy in prohibited_energies:
            list_heaters = list(set(list_heaters) & set([heater for heater in self._choice_heater if energy not in heater]))

        if energy_regulations:
            choice_heater_idx = Index(list_heaters, name='Heating system final')
        else:
            choice_heater_idx = Index(self._choice_heater, name='Heating system final')

        frame = Series(dtype=float, index=index).to_frame().dot(
            Series(dtype=float, index=choice_heater_idx).to_frame().T)
        cost_heater, tax_heater, subsidies_details, subsidies_total = self.apply_subsidies_heater(policies_heater,
                                                                                                  cost_heater.copy(),
                                                                                                  frame)
        if self._endogenous:
            subsidies_utility = subsidies_total.copy()
            if 'reduced_tax' in subsidies_details.keys():
                subsidies_utility -= subsidies_details['reduced_tax']
            market_share = self.endogenous_market_share_heater(index, prices, subsidies_utility, cost_heater,
                                                               ms_heater=ms_heater)

        else:
            market_share = self.exogenous_market_share_heater(index, choice_heater_idx)

        replacement = ((market_share * probability_replacement).T * stock).T

        stock_replacement = replacement.stack('Heating system final')
        to_replace = replacement.sum(axis=1)

        stock = stock - to_replace

        # adding heating system final equal to heating system because no switch
        stock = concat((stock, Series(stock.index.get_level_values('Heating system'), index=stock.index,
                                            name='Heating system final')), axis=1).set_index('Heating system final', append=True).squeeze()
        stock = concat((stock.reorder_levels(stock_replacement.index.names), stock_replacement),
                       axis=0, keys=[False, True], names=['Heater replacement'])
        assert round(stock.sum() - self.stock_mobile.sum(), 0) == 0, 'Sum problem'

        replaced_by = stock.droplevel('Heating system').rename_axis(index={'Heating system final': 'Heating system'})

        if self.full_output:
            self.store_information_heater(cost_heater, subsidies_total, subsidies_details, replacement, tax_heater,
                                          replaced_by)
        else:
            self.cost_heater = cost_heater

        return stock

    def apply_subsidies_heater(self, policies_heater, cost_heater, frame):
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

        subsidies_total = DataFrame(0, index=frame.index, columns=frame.columns)
        subsidies_details = {}

        tax = self.vta
        p = [p for p in policies_heater if 'reduced_tax' == p.policy]
        if p:
            tax = p[0].value
            sub = cost_heater * (self.vta - tax)
            subsidies_details.update({'reduced_tax': concat([sub] * frame.shape[0], keys=frame.index, axis=1).T})
            subsidies_total += subsidies_details['reduced_tax']

        tax_heater = cost_heater * tax
        cost_heater += tax_heater

        sub = None
        for policy in policies_heater:
            if policy.name not in self.policies and policy.policy in ['subsidy_target', 'subsidy_non_cumulative', 'subsidy_ad_volarem', 'subsidies_cap']:
                self.policies += [policy.name]
            if policy.policy == 'subsidy_target':
                sub = policy.value.reindex(frame.columns, axis=1).fillna(0)
                sub = reindex_mi(sub, frame.index)
            elif policy.policy == 'subsidy_ad_volarem':

                if isinstance(policy.value, (float, int)):
                    sub = policy.value * cost_heater
                    sub = concat([sub] * frame.shape[0], keys=frame.index, axis=1).T

                if isinstance(policy.value, DataFrame):
                    sub = policy.value * cost_heater
                    sub = reindex_mi(sub, frame.index).fillna(0)

                if isinstance(policy.value, Series):
                    if policy.by == 'index':
                        sub = policy.value.to_frame().dot(cost_heater.to_frame().T)
                        sub = reindex_mi(sub, frame.index).fillna(0)
                    elif policy.by == 'columns':
                        sub = (policy.value * cost_heater).fillna(0).reindex(frame.columns)
                        sub = concat([sub] * frame.shape[0], keys=frame.index, names=frame.index.names, axis=1).T
                    else:
                        raise NotImplemented
                if policy.cap:
                    sub[sub > policy.cap] = sub
            else:
                continue

            subsidies_details[policy.name] = sub
            subsidies_total += subsidies_details[policy.name]
        return cost_heater, tax_heater, subsidies_details, subsidies_total

    def store_information_heater(self, cost_heater, subsidies_total, subsidies_details, replacement, tax_heater,
                                 replaced_by):
        """Store information yearly heater replacement.

        Parameters
        ----------
        cost_heater: Series
            Cost of each heating system (€).
        subsidies_total: DataFrame
            Total amount of eligible subsidies by dwelling and heating system (€).
        subsidies_details: dict
            Amount of eligible subsidies by dwelling and heating system (€).
        replacement: DataFrame
            Number of heating system replacement by dwelling and heating system chosen.
        tax_heater: Series
            VTA tax of each heating system (€).
        replaced_by: Series
            Dwelling updated with a new heating system.
        """
        # information stored during
        self.cost_heater = cost_heater
        self.subsidies_heater_indiv = subsidies_total
        self.subsidies_details_heater = subsidies_details
        self.replacement_heater = replacement
        self.investment_heater = replacement * cost_heater
        self.tax_heater = replacement * tax_heater
        self.subsidies_heater = replacement * subsidies_total
        self.heater_replaced = replaced_by
        for key in self.subsidies_details_heater.keys():
            self.subsidies_details_heater[key] *= replacement

        for key, sub in self.subsidies_details_heater.items():
            mask = sub.copy()
            mask[mask > 0] = 1
            # self.subsidies_count_heater.update({key: (replacement.fillna(0) * mask).sum().sum()})
            self.subsidies_count_heater.update({key: (replacement.fillna(0) * mask).sum(axis=1).groupby('Housing type').sum()})
            self.subsidies_average_heater.update({key: sub.sum().sum() / replacement.fillna(0).sum().sum()})

    def endogenous_market_share_heater(self, index, prices, subsidies_total, cost_heater, ms_heater=None):

        def calibration_constant_heater(utility, ms_heater):
            """Constant to match the observed market-share.

            Market-share is defined by initial and final heating system.

            Parameters
            ----------
            utility: DataFrame
            ms_heater: DataFrame

            Returns
            -------
            DataFrame
            """

            # removing unnecessary level
            utility_ref = utility.droplevel(['Occupancy status']).copy()
            utility_ref = utility_ref[~utility_ref.index.duplicated(keep='first')]

            possible = reindex_mi(ms_heater, utility_ref.index)
            utility_ref[~(possible > 0)] = float('nan')

            stock = self.stock.groupby(utility_ref.index.names).sum()

            # initializing constant to 0
            constant = ms_heater.copy()
            constant[constant > 0] = 0
            market_share_ini, market_share_agg = None, None
            for i in range(50):
                constant.loc[constant['Electricity-Heat pump water'].notna(), 'Electricity-Heat pump water'] = 0
                constant.loc[constant['Electricity-Heat pump water'].isna(), 'Electricity-Heat pump air'] = 0

                utility_constant = reindex_mi(constant.reindex(utility_ref.columns, axis=1), utility.index)
                utility = utility_ref + utility_constant
                market_share = (exp(utility).T / exp(utility).sum(axis=1)).T
                agg = (market_share.T * stock).T.groupby(['Housing type', 'Heating system']).sum()
                market_share_agg = (agg.T / agg.sum(axis=1)).T
                if i == 0:
                    market_share_ini = market_share_agg.copy()
                constant = constant + log(ms_heater / market_share_agg)

                ms_heater = ms_heater.reindex(market_share_agg.index)

                if (market_share_agg.round(decimals=3) == ms_heater.round(decimals=3).fillna(0)).all().all():
                    self.logger.debug('Constant heater optim worked')
                    break

            constant.loc[constant['Electricity-Heat pump water'].notna(), 'Electricity-Heat pump water'] = 0
            constant.loc[constant['Electricity-Heat pump water'].isna(), 'Electricity-Heat pump air'] = 0

            details = concat((constant.stack(), market_share_ini.stack(), market_share_agg.stack(), ms_heater.stack()),
                             axis=1, keys=['constant', 'calcul ini', 'calcul', 'observed']).round(decimals=3)
            if self.path is not None:
                details.to_csv(os.path.join(self.path_calibration, 'calibration_constant_heater.csv'))

            return constant

        choice_heater = self._choice_heater
        choice_heater_idx = Index(choice_heater, name='Heating system final')
        energy = Series(choice_heater).str.split('-').str[0].set_axis(choice_heater_idx)

        temp = pd.Series(0, index=index, dtype='float').to_frame().dot(pd.Series(0, index=choice_heater_idx, dtype='float').to_frame().T)
        index_final = temp.stack().index
        consumption_heat_sd, _, certificate = self.consumption_standard(index_final, level_heater='Heating system final')
        consumption_heat_sd = reindex_mi(consumption_heat_sd.unstack('Heating system final'), index)
        prices_re = prices.reindex(energy).set_axis(consumption_heat_sd.columns)
        energy_bill_sd = ((consumption_heat_sd * prices_re).T * reindex_mi(self._surface, index)).T

        consumption_before = self.consumption_standard(index, level_heater='Heating system')[0]
        consumption_before = reindex_mi(consumption_before, index) * reindex_mi(self._surface, index)
        energy_bill_before = AgentBuildings.energy_bill(prices, consumption_before)

        bill_saved = - energy_bill_sd.sub(energy_bill_before, axis=0)
        utility_bill_saving = (bill_saved.T * reindex_mi(self.preferences_heater['bill_saved'], bill_saved.index)).T / 1000
        utility_bill_saving = utility_bill_saving.loc[:, choice_heater]

        certificate = reindex_mi(certificate.unstack('Heating system final'), index)
        certificate_before = self.consumption_standard(index)[2]
        certificate_before = reindex_mi(certificate_before, index)

        self.certificate_jump_heater = - certificate.replace(self._epc2int).sub(
            certificate_before.replace(self._epc2int), axis=0)

        utility_subsidies = subsidies_total * self.preferences_heater['subsidy'] / 1000

        cost_heater = cost_heater.reindex(utility_bill_saving.columns)
        pref_investment = reindex_mi(self.preferences_heater['investment'], utility_bill_saving.index).rename(None)
        utility_investment = pref_investment.to_frame().dot(cost_heater.to_frame().T) / 1000

        utility_inertia = DataFrame(0, index=utility_bill_saving.index, columns=utility_bill_saving.columns)
        for hs in choice_heater:
            utility_inertia.loc[
                utility_inertia.index.get_level_values('Heating system') == hs, hs] = self.preferences_heater['inertia']

        utility = utility_inertia + utility_investment + utility_bill_saving + utility_subsidies

        if (self.constant_heater is None) and (ms_heater is not None):
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
        _, _, certificate = self.consumption_standard(index_final, level_heater='Heating system final')
        certificate = reindex_mi(certificate.unstack('Heating system final'), index)
        certificate_before = self.consumption_standard(index)[2]
        certificate_before = reindex_mi(certificate_before, index)

        self.certificate_jump_heater = - certificate.replace(self._epc2int).sub(
            certificate_before.replace(self._epc2int), axis=0)

        return market_share

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
        _, _, certificate_before_heater = self.consumption_standard(index, level_heater='Heating system')
        # index only contains building with energy performance > B
        c_before = reindex_mi(certificate_before_heater, index)
        index = c_before[c_before > 'B'].index

        # before include the change of heating system
        _, consumption_3uses_before, certificate_before = self.consumption_standard(index,
                                                                                    level_heater='Heating system final')
        certificate_before = certificate_before[certificate_before > 'B']
        consumption_3uses_before = consumption_3uses_before.loc[certificate_before.index]
        # TODO: possible future bugs coming from this line
        temp = reindex_mi(certificate_before, index)
        index = temp[temp > 'B'].index
        stock = stock[index]

        condition = np.array([1] * stock.shape[0], dtype=bool)
        for k, v in self._performance_insulation.items():
            condition *= stock.index.get_level_values(k) == v
        stock = stock[~condition]
        index = stock.index

        surface = reindex_mi(self._surface, index)

        _, consumption_3uses, certificate = self.prepare_consumption(self._choice_insulation, index=index,
                                                                     level_heater='Heating system final')
        energy_saved_3uses = ((consumption_3uses_before - consumption_3uses.T) / consumption_3uses_before).T
        energy_saved_3uses.dropna(inplace=True)

        cost_insulation = self.prepare_cost_insulation(cost_insulation_raw * self.surface_insulation)
        cost_insulation = cost_insulation.T.multiply(self._surface, level='Housing type').T

        cost_insulation, tax_insulation, tax, subsidies_details, subsidies_total, condition, certificate_jump_all = self.apply_subsidies_insulation(
            index, policies_insulation, cost_insulation, surface, certificate, certificate_before, certificate_before_heater, energy_saved_3uses)

        if self.full_output:
            self.store_information_insulation(certificate_jump_all, condition, cost_insulation_raw, tax, cost_insulation,
                                              tax_insulation, subsidies_details, subsidies_total)
        else:
            self.subsidies_details_insulation_indiv = subsidies_details

        if self._endogenous:
            if 'reduced_tax' in subsidies_details.keys():
                subsidies_total -= subsidies_details['reduced_tax']

            if calib_renovation is not None:
                if calib_renovation['scale']['name'] == 'freeriders':
                    delta_subsidies = None
                    if (self.year in [self.first_year + 1]) and (self.scale is None):
                        delta_subsidies = subsidies_details[calib_renovation['scale']['target_policies']].copy()
                    calib_renovation['scale']['delta_subsidies'] = delta_subsidies

            retrofit_rate, market_share = self.endogenous_retrofit(stock, prices, subsidies_total,
                                                                   cost_insulation,
                                                                   calib_intensive=calib_intensive,
                                                                   calib_renovation=calib_renovation,
                                                                   financing_cost=financing_cost,
                                                                   min_performance=min_performance,
                                                                   subsidies_details=subsidies_details)


        else:
            retrofit_rate, market_share = self.exogenous_retrofit(stock)

        if self.retrofit_rate is None:
            self.retrofit_rate, self.market_share = retrofit_rate, market_share

        return retrofit_rate, market_share

    def apply_subsidies_insulation(self, index, policies_insulation, cost_insulation, surface, certificate,
                                   certificate_before, certificate_before_heater, energy_saved_3uses):
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

        Returns
        -------
        cost_insulation: DataFrame
        tax_insulation: DataFrame
        tax : float
        subsidies_details: dict
        subsidies_total: DataFrame
        condition: dict
        certificate_jump_all: DataFrame
        """

        def defined_condition(_index, _certificate, _certificate_before, _certificate_before_heater,
                              _energy_saved_3uses, _cost_insulation):
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

            Returns
            -------
            condition: dict
                Contains boolean DataFrame that established condition to get subsidies.
            certificate_jump: DataFrame
                Insulation (without account for heater replacement) allowed to jump of at least one certificate.
            certificate_jump_all: DataFrame
                Renovation (including heater replacement) allowed to jump of at least one certificate.
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

            condition_target = dict()

            self.out_worst = (~_certificate.isin(['G', 'F'])).T.multiply(_certificate_before.isin(['G', 'F'])).T
            self.out_worst = reindex_mi(self.out_worst, _index).fillna(False).astype('float')
            self.in_best = (_certificate.isin(['A', 'B'])).T.multiply(~_certificate_before.isin(['A', 'B'])).T
            self.in_best = reindex_mi(self.in_best, _index).fillna(False).astype('float')

            condition_target.update({'bonus_worst': self.out_worst})
            condition_target.update({'bonus_best': self.in_best})

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

            self.best_option = (best.T == best.min(axis=1)).T

            minimum_gest_condition, global_condition = 1, 2
            energy_condition = 0.35

            _certificate_jump = - _certificate.replace(self._epc2int).sub(_certificate_before.replace(self._epc2int),
                                                                          axis=0)
            _certificate_jump = reindex_mi(_certificate_jump, _index)
            certificate_jump_condition = _certificate_jump >= minimum_gest_condition

            _certificate_before_heater = reindex_mi(_certificate_before_heater, _index)
            _certificate = reindex_mi(_certificate, _index)
            _certificate_jump_all = - _certificate.replace(self._epc2int).sub(
                _certificate_before_heater.replace(self._epc2int),
                axis=0)

            condition_target.update({'certificate_jump': _certificate_jump_all >= minimum_gest_condition})
            condition_target.update({'global_renovation': _certificate_jump_all >= global_condition})

            low_income_condition = ['D1', 'D2', 'D3', 'D4']
            if self.quintiles:
                low_income_condition = ['C1', 'C2']
            low_income_condition = _index.get_level_values('Income owner').isin(low_income_condition)
            low_income_condition = pd.Series(low_income_condition, index=_index)

            high_income_condition = ['D5', 'D6', 'D7', 'D8', 'D9', 'D10']
            if self.quintiles:
                high_income_condition = ['C3', 'C4', 'C5']
            high_income_condition = _index.get_level_values('Income owner').isin(high_income_condition)
            high_income_condition = pd.Series(high_income_condition, index=_index)

            global_renovation_low_income = (low_income_condition & condition_target['global_renovation'].T).T
            condition_target.update({'global_renovation_low_income': global_renovation_low_income})

            global_renovation_high_income = (high_income_condition & condition_target['global_renovation'].T).T
            condition_target.update({'global_renovation_high_income': global_renovation_high_income})

            energy_condition = _energy_saved_3uses >= energy_condition

            condition_mpr_serenite = (reindex_mi(energy_condition, _index).T & low_income_condition).T
            condition_target.update({'mpr_serenite': condition_mpr_serenite})

            condition_zil = define_zil_target(_certificate, _certificate_before, _energy_saved_3uses)
            condition_target.update({'zero_interest_loan': condition_zil})

            return condition_target, _certificate_jump, _certificate_jump_all

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
            temp = self.constant_insulation_extensive.copy()
            temp = temp.drop(temp[temp.index.get_level_values(level) == idx_target].index)
            t = temp[temp.index.get_level_values(level) == idx_replace]
            t.rename(index={idx_replace: idx_target}, inplace=True)
            temp = pd.concat((temp, t)).loc[self.constant_insulation_extensive.index]
            self.constant_insulation_extensive = temp.copy()

        subsidies_total = DataFrame(0, index=index, columns=cost_insulation.columns)
        subsidies_details = {}

        tax = self.vta
        p = [p for p in policies_insulation if 'reduced_tax' == p.policy]
        if p:
            tax = p[0].value
            subsidies_details.update({p[0].name: reindex_mi(cost_insulation * (self.vta - tax), index)})
            subsidies_total += subsidies_details['reduced_tax']

        tax_insulation = cost_insulation * tax
        cost_insulation += tax_insulation

        condition, certificate_jump, certificate_jump_all = defined_condition(index, certificate, certificate_before,
                                                                              certificate_before_heater,
                                                                              energy_saved_3uses, cost_insulation)

        for policy in policies_insulation:
            if policy.name not in self.policies and policy.policy in ['subsidy_target', 'subsidy_non_cumulative', 'subsidy_ad_volarem', 'subsidies_cap']:
                self.policies += [policy.name]

            if policy.policy == 'subsidy_target':
                temp = (reindex_mi(self.prepare_subsidy_insulation(policy.value),
                                   index).T * surface).T
                subsidies_total += temp

                if policy.name in subsidies_details.keys():
                    subsidies_details[policy.name] = subsidies_details[policy.name] + temp
                else:
                    subsidies_details[policy.name] = temp.copy()

            elif policy.policy == 'bonus_best':
                temp = (reindex_mi(policy.value, condition['bonus_best'].index) * condition['bonus_best'].T).T
                subsidies_total += temp
                if policy.name in subsidies_details.keys():
                    subsidies_details[policy.name] = subsidies_details[policy.name] + temp

                else:
                    subsidies_details[policy.name] = temp.copy()

            elif policy.policy == 'bonus_worst':
                temp = (reindex_mi(policy.value, condition['bonus_worst'].index) * condition['bonus_worst'].T).T
                subsidies_total += temp

                if policy.name in subsidies_details.keys():
                    subsidies_details[policy.name] = subsidies_details[policy.name] + temp

                else:
                    subsidies_details[policy.name] = temp.copy()

            elif policy.policy == 'subsidy_ad_volarem':

                cost = policy.cost_targeted(reindex_mi(cost_insulation, index), target_subsidies=condition.get(policy.name),
                                            cost_included=self.cost_heater.copy())

                if isinstance(policy.value, (Series, float)):
                    temp = reindex_mi(policy.value, cost.index)
                    subsidies_details[policy.name] = (temp * cost.T).T
                    subsidies_total += subsidies_details[policy.name]
                else:
                    temp = self.prepare_subsidy_insulation(policy.value, policy=policy.policy)
                    temp = reindex_mi(temp, cost.index)
                    subsidies_details[policy.name] = temp * cost
                    subsidies_total += subsidies_details[policy.name]
                if policy.name == 'zero_interest_loan':
                    self.zil_loaned = cost.copy()

            elif policy.policy == 'zero_interest_loan':

                cost = policy.cost_targeted(reindex_mi(cost_insulation, index), target_subsidies=condition.get(policy.name),
                                            cost_included=self.cost_heater.copy())
                subsidies_details[policy.name] = policy.value * cost
                subsidies_total += subsidies_details[policy.name]
                self.zil_loaned = cost.copy()

        subsidies_non_cumulative = [p for p in policies_insulation if p.policy == 'subsidy_non_cumulative']
        if subsidies_non_cumulative is not []:
            for policy in subsidies_non_cumulative:
                sub = (reindex_mi(policy.value, condition[policy.name].index) * condition[policy.name].T).T
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

        if 'reduced_tax' in subsidies_details.keys():
            subsidies_uncaped -= subsidies_details['reduced_tax']

        zil = [p for p in policies_insulation if p.policy == 'subsidy_ad_volarem' and p.name == 'zero_interest_loan']
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

        return cost_insulation, tax_insulation, tax, subsidies_details, subsidies_total, condition, certificate_jump_all

    def store_information_insulation(self, certificate_jump_all, condition, cost_insulation_raw, tax, cost_insulation,
                                     tax_insulation, subsidies_details, subsidies_total):
        """Store insulation information.


        Information are post-treated to weight by the number of replacement.

        Parameters
        ----------
        certificate_jump_all
        condition: dict
        cost_insulation_raw: Series
            Cost of insulation for each envelope component of losses surface (€/m2).
        tax: float
            VTA to apply (%).
        cost_insulation: DataFrame
            Cost insulation for each dwelling and each insulation gesture (€).
        tax_insulation: DataFrame
            VTA applied to each insulation gesture cost (€).
        subsidies_details: dict
            Amount of subsidies for each dwelling and each insulation gesture (€).
        subsidies_total: DataFrame
            Total mount of subsidies for each dwelling and each insulation gesture (€).
        """

        # self.certificate_jump = condition['certificate_jump']
        self.certif_jump_all = certificate_jump_all
        self.global_renovation = condition['global_renovation']
        self.in_global_renovation_high_income = condition['global_renovation_high_income']
        self.in_global_renovation_low_income = condition['global_renovation_low_income']

        self.subsidies_details_insulation_indiv = subsidies_details
        self.subsidies_insulation_indiv = subsidies_total
        self.cost_insulation_indiv = cost_insulation
        self.tax_insulation = tax_insulation

        self.cost_component = cost_insulation_raw * self.surface_insulation * (1 + tax)

    def store_information_retrofit(self, replaced_by):
        """Calculate and store main statistics based on yearly retrofit.

        Parameters
        ----------
        replaced_by: DataFrame
            Retrofit flow for each dwelling (index) and each insulation gesture (columns).
            Dwelling must be defined with 'Heating system final' and 'Heater replacement'.
        """

        levels = [i for i in replaced_by.index.names if i not in ['Heater replacement', 'Heating system final']]
        if 'Heater replacement' not in replaced_by.index.names:
            replaced_by = concat([replaced_by], keys=[False], names=['Heater replacement'])
        if 'Heating system final' not in replaced_by.index.names:
            temp = replaced_by.reset_index('Heating system')
            temp['Heating system final'] = temp['Heating system']
            replaced_by = temp.set_index(['Heating system', 'Heating system final'], append=True).squeeze()

        replaced_by.index = replaced_by.index.reorder_levels(self.in_global_renovation_high_income.index.names)

        self.global_renovation_high_income += (replaced_by * self.in_global_renovation_high_income).sum().sum()
        self.global_renovation_low_income += (replaced_by * self.in_global_renovation_low_income).sum().sum()
        self.bonus_best += (replaced_by * self.in_best).sum().sum()
        self.bonus_worst += (replaced_by * self.out_worst).sum().sum()
        if self.replacement_insulation is None:
            self.replacement_insulation = replaced_by.groupby(levels).sum()
            self.investment_insulation = (replaced_by * self.cost_insulation_indiv).groupby(levels).sum()
            self.taxed_insulation = (replaced_by * self.tax_insulation).groupby(levels).sum()
            self.subsidies_insulation = (replaced_by * self.subsidies_insulation_indiv).groupby(levels).sum()

            for key in self.subsidies_details_insulation_indiv.keys():
                self.subsidies_details_insulation[key] = (replaced_by * reindex_mi(
                    self.subsidies_details_insulation_indiv[key], replaced_by.index)).groupby(levels).sum()

            rslt = {}
            l = unique(self.certif_jump_all.values.ravel('K'))
            for i in l:
                rslt.update({i: ((self.certif_jump_all == i) * replaced_by).sum(axis=1)})
            self.certificate_jump_all = DataFrame(rslt).groupby(levels).sum()

            gest = {1: [(False, False, False, True), (False, False, True, False), (False, True, False, False),
                        (True, False, False, False)],
                    2: [(False, False, True, True), (False, True, False, True), (True, False, False, True),
                        (True, False, True, False),
                        (True, True, False, False), (False, True, True, False)],
                    3: [(False, True, True, True), (True, False, True, True), (True, True, False, True),
                        (True, True, True, False)],
                    4: [(True, True, True, True)]}
            rslt = {i: 0 for i in range(1, 6)}
            for n, g in gest.items():
                rslt[n] += replaced_by.loc[:, g].xs(False, level='Heater replacement').sum().sum()
                rslt[n + 1] += replaced_by.loc[:, g].xs(True, level='Heater replacement').sum().sum()
            self.gest_nb = Series(rslt)

            self.retrofit_with_heater = replaced_by.xs(True, level='Heater replacement').sum().sum()

            for key, sub in self.subsidies_details_insulation_indiv.items():
                mask = sub.copy()
                mask[mask > 0] = 1
                # self.subsidies_count_insulation[key] = (replaced_by.fillna(0) * mask).sum().sum()
                self.subsidies_count_insulation[key] = (replaced_by.fillna(0) * mask).sum(axis=1).groupby('Housing type').sum()
                self.subsidies_average_insulation[key] = sub.sum().sum() / replaced_by.fillna(0).sum().sum()

                """if key == 'zero_interest_loan':
                    total_loaned = (replaced_by.fillna(0) * self.zil_loaned).sum().sum()
                    self.zil_loaned_avg = total_loaned / self.zil_count"""

        else:
            temp = replaced_by.groupby(levels).sum().reindex(self.replacement_insulation.index).fillna(0)
            self.replacement_insulation += temp
            temp = (replaced_by * self.cost_insulation_indiv).groupby(levels).sum().reindex(self.investment_insulation.index).fillna(0)
            self.investment_insulation += temp
            temp = (replaced_by * self.tax_insulation).groupby(levels).sum().reindex(self.taxed_insulation.index).fillna(0)
            self.taxed_insulation += temp
            temp = (replaced_by * self.subsidies_insulation_indiv).groupby(levels).sum().reindex(self.subsidies_insulation.index).fillna(0)
            self.subsidies_insulation += temp

            for key in self.subsidies_details_insulation_indiv.keys():
                temp = (replaced_by * reindex_mi(self.subsidies_details_insulation_indiv[key], replaced_by.index)).groupby(levels).sum().reindex(self.subsidies_details_insulation[key].index).fillna(0)
                self.subsidies_details_insulation[key] += temp

            rslt = {}
            l = unique(self.certif_jump_all.values.ravel('K'))
            for i in l:
                rslt.update({i: ((self.certif_jump_all == i) * replaced_by).sum(axis=1)})
            temp = DataFrame(rslt).groupby(levels).sum().reindex(self.certificate_jump_all.index).fillna(0)
            self.certificate_jump_all += temp

            gest = {1: [(False, False, False, True), (False, False, True, False), (False, True, False, False),
                        (True, False, False, False)],
                    2: [(False, False, True, True), (False, True, False, True), (True, False, False, True),
                        (True, False, True, False),
                        (True, True, False, False), (False, True, True, False)],
                    3: [(False, True, True, True), (True, False, True, True), (True, True, False, True),
                        (True, True, True, False)],
                    4: [(True, True, True, True)]}
            rslt = {i: 0 for i in range(1, 6)}
            for n, g in gest.items():
                rslt[n] += replaced_by.loc[:, g].xs(False, level='Heater replacement').sum().sum()
                # explain why - no heater replacement in flow obligation
                # rslt[n + 1] += replaced_by.loc[:, g].xs(True, level='Heater replacement').sum().sum()
            self.gest_nb += Series(rslt).reindex(self.gest_nb.index).fillna(0)

            # self.retrofit_with_heater += replaced_by.xs(True, level='Heater replacement').sum().sum()

            for key, sub in self.subsidies_details_insulation.items():
                mask = sub.copy()
                mask[mask > 0] = 1
                self.subsidies_count_insulation[key] += (replaced_by.fillna(0) * mask).sum(axis=1).groupby('Housing type').sum()

                # self.subsidies_count_insulation[key] += (replaced_by.fillna(0) * mask).sum().sum()
                # self.subsidies_average_insulation[key] += sub.sum().sum() / replaced_by.fillna(0).sum().sum()

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
        if self.prepared_cost_insulation is None:
            cost = DataFrame(0, index=cost_insulation.index, columns=self._choice_insulation)
            idx = IndexSlice
            cost.loc[:, idx[True, :, :, :]] = (cost.loc[:, idx[True, :, :, :]].T + cost_insulation['Wall']).T
            cost.loc[:, idx[:, True, :, :]] = (cost.loc[:, idx[:, True, :, :]].T + cost_insulation['Floor']).T
            cost.loc[:, idx[:, :, True, :]] = (cost.loc[:, idx[:, :, True, :]].T + cost_insulation['Roof']).T
            cost.loc[:, idx[:, :, :, True]] = (cost.loc[:, idx[:, :, :, True]].T + cost_insulation['Windows']).T
            self.prepared_cost_insulation = cost.copy()
            return cost
        else:
            # to reduce run time but doesn't seem to be very useful
            return self.prepared_cost_insulation

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

        if policy == 'subsidy_ad_volarem':
            # NotImplemented: ad_volarem with different subsidides rate
            value = [v for v in subsidies_insulation.stack().unique() if v != 0][0]
            subsidies[subsidies > 0] = value

        return subsidies

    def endogenous_retrofit(self, stock, prices, subsidies_total, cost_insulation, financing_cost=None,
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

            pref_sub = reindex_mi(self.preferences_insulation_int['subsidy'], _subsidies_total.index).rename(None)
            utility_subsidies = (_subsidies_total.T * pref_sub).T / 1000

            pref_investment = reindex_mi(self.preferences_insulation_int['investment'], _cost_total.index).rename(None)
            utility_investment = (_cost_total.T * pref_investment).T / 1000

            utility_bill_saving = (_bill_saved.T * reindex_mi(self.preferences_insulation_int['bill_saved'], _bill_saved.index)).T / 1000

            util_intensive = utility_bill_saving + utility_investment + utility_subsidies

            if self.constant_insulation_intensive is not None:
                util_intensive += self.constant_insulation_intensive

            ms_intensive = market_share_func(util_intensive)
            return ms_intensive, util_intensive

        def to_utility_extensive(_cost_total, _bill_saved, _subsidies_total, _market_share, _utility_intensive):
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

                dict_int['representative'] = rename_tuple(dict_int['columns'], utility_intensive.columns.names)
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

            utility_bill_saving = reindex_mi(self.preferences_insulation_ext['bill_saved'], _bill_saved.index) * _bill_saved / 1000

            pref_sub = reindex_mi(self.preferences_insulation_ext['subsidy'], _subsidies_total.index).rename(None)
            utility_subsidies = (pref_sub * _subsidies_total) / 1000

            pref_investment = reindex_mi(self.preferences_insulation_ext['investment'], _cost_total.index).rename(None)
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

            _investment_insulation, _bill_saved_insulation, _subsidies_insulation = to_utility_extensive(_cost_total,
                                                                                                         _bill_saved,
                                                                                                         _subsidies_total,
                                                                                                         _market_share,
                                                                                                         _utility_intensive)

            _renovation_rate, _utility = to_retrofit_rate(_bill_saved_insulation, _subsidies_insulation,
                                                        _investment_insulation)

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

        def calibration_intensive(util, _stock, market_share_ini, retrofit_rate_ini, iteration=1000):
            """Calibrate alternative-specific constant to match observed market-share.


            Parameters
            ----------
            _stock: Series
            util: DataFrame
            market_share_ini: Series
                Observed market-share.
            retrofit_rate_ini: Series
                Observed renovation rate.
            iteration: optional, int, default 100

            Returns
            -------
            Series
            """

            if 'Performance' in retrofit_rate_ini.index.names:
                _stock = self.add_certificate(_stock)

            f_retrofit = _stock * reindex_mi(retrofit_rate_ini, _stock.index)
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

            nb_renovation = (_stock * reindex_mi(retrofit_rate_ini, _stock.index)).sum()
            wtp = const / self.preferences_insulation_int['investment']
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

            coefficient_cost = abs(self.preferences_insulation_ext['investment'] * _scale)
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
            _retrofit_rate
            _stock
            _market_share
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
                if key == 'reduced_tax':
                    _c = _sub
                    _sub = 0

                c_total = _cost_total + _c
                assert (c_total >= _cost_total).all().all(), 'Cost issue'
                sub_total = _subsidies_total - _sub
                assert (sub_total <= _subsidies_total).all().all(), 'Subsidies issue'
                ms_sub, retrofit_sub = apply_endogenous_retrofit(_bill_saved, sub_total, c_total)
                f_retrofit_sub = retrofit_sub * stock
                f_replace_sub = (f_retrofit_sub * ms_sub.T).T
                avg_cost_global_sub = (f_replace_sub * cost_total).sum().sum() / f_retrofit_sub.sum()
                f_replace_sub = f_replace_sub[mask]
                avg_cost_benef_sub = (f_replace_sub * cost_total).sum().sum() / f_replace_sub.sum().sum()
                f_replace_benef = f_replace[mask]
                avg_cost_benef = (f_replace_benef * cost_total).sum().sum() / f_replace_benef.sum().sum()

                if key == 'reduced_tax':
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

        def calculation_intensive_margin(stock_segment, retrofit_rate_ini, bill_save, sub_total, cost_insul,
                                         delta_sub, target_invest=0.2):
            """ This function can be adapted to calibrate intensive margin on Risch 2020 result.


            However, for now just returns percentage of intensive margin difference

            Parameters
            ----------
            stock_segment
            retrofit_rate_ini
            bill_save: DataFrame
            sub_total: DataFrame
            cost_insul: DataFrame
            delta_sub: DataFrame, policies used to calibrate the scale.
            target_invest: float

            Returns
            -------

            """

            if 'Performance' in retrofit_rate_ini.index.names:
                stock_segment = self.add_certificate(stock_segment)
            f_retrofit = stock_segment * reindex_mi(retrofit_rate_ini, stock_segment.index)
            f_retrofit = f_retrofit.droplevel('Performance').dropna()

            def solve(scal, retrofit, b_save, sub_tot, c_insul, d_sub, target):
                ms_before, _ = to_market_share(b_save, sub_tot, c_insul)
                investment_insulation_before = (c_insul.reindex(ms_before.index) * ms_before).sum(axis=1)
                investment_insulation_before = (investment_insulation_before * retrofit).sum() / retrofit.sum()
                new_sub = sub_tot + d_sub
                ms_after, _ = to_market_share(b_save, new_sub, c_insul)
                investment_insulation_after = (c_insul.reindex(ms_after.index) * ms_after).sum(axis=1)
                investment_insulation_after = (investment_insulation_after * retrofit).sum() / retrofit.sum()

                delta_invest = (investment_insulation_before - investment_insulation_after) / investment_insulation_before
                return delta_invest - target

            return solve(1, f_retrofit, bill_save, sub_total, cost_insul, -0.5 * delta_sub,
                         target_invest) + target_invest

        index = stock.index

        cost_insulation = reindex_mi(cost_insulation, index)
        cost_total = cost_insulation.copy()
        discount_factor = None
        if financing_cost is not None and self.financing_cost:
            share_debt = financing_cost['share_debt'][0] + cost_insulation * financing_cost['share_debt'][1]
            cost_debt = financing_cost['interest_rate'] * share_debt * cost_insulation * financing_cost['duration']
            cost_saving = (financing_cost['saving_rate'] * reindex_mi(financing_cost['factor_saving_rate'], cost_insulation.index) * ((1 - share_debt) * cost_insulation).T).T * financing_cost['duration']
            cost_financing = cost_debt + cost_saving
            cost_total += cost_financing

            # discount = financing_cost['interest_rate'] * share_debt + (1 - share_debt) * financing_cost['factor_saving_rate']
            # discount_factor = (1 - (1 + discount) ** -30) / 30

        consumption_before = self.consumption_standard(index, level_heater='Heating system final')[0]
        consumption_before = reindex_mi(consumption_before, index) * reindex_mi(self._surface, index)
        energy_bill_before = AgentBuildings.energy_bill(prices, consumption_before, level_heater='Heating system final')

        consumption_sd = self.prepare_consumption(self._choice_insulation, index=index,
                                                  level_heater='Heating system final')[0]
        consumption_sd = reindex_mi(consumption_sd, index).reindex(self._choice_insulation, axis=1)

        energy = pd.Series(index.get_level_values('Heating system final'), index=index).str.split('-').str[0].rename('Energy')
        energy_prices = prices.reindex(energy).set_axis(index)
        energy_bill_sd = (consumption_sd.T * energy_prices * reindex_mi(self._surface, index)).T
        bill_saved = - energy_bill_sd.sub(energy_bill_before, axis=0).dropna()

        # and         if self.constant_insulation_extensive is None:
        if self.constant_insulation_intensive is None:
            self.logger.info('Calibration intensive and renovation rate')

            renovation_rate_ini = calib_renovation['renovation_rate_ini']
            ms_insulation_ini = calib_intensive['ms_insulation_ini']

            # initialization of market-share and renovation rate
            market_share, utility_intensive = to_market_share(bill_saved, subsidies_total, cost_total)
            renovation_rate = ms_insulation_ini
            # initialization calibration of market_share
            _, utility_intensive = to_market_share(bill_saved, subsidies_total, cost_total)
            self.constant_insulation_intensive = calibration_intensive(utility_intensive, stock, ms_insulation_ini,
                                                                       renovation_rate)
            market_share, utility_intensive = to_market_share(bill_saved, subsidies_total, cost_total)

            investment_insulation, bill_saved_insulation, subsidies_insulation = to_utility_extensive(cost_total,
                                                                                                      bill_saved,
                                                                                                      subsidies_total,
                                                                                                      market_share,
                                                                                                      utility_intensive)
            compare = None
            for k in range(10):
                # calibration of renovation rate
                self.constant_insulation_extensive = None
                _, utility = to_retrofit_rate(bill_saved_insulation, subsidies_insulation, investment_insulation)
                constant, scale = calibration_extensive(utility, stock, calib_renovation)
                self.constant_insulation_extensive = constant
                self.apply_scale(scale)
                renovation_rate, _ = to_retrofit_rate(bill_saved_insulation, subsidies_insulation, investment_insulation)

                # recalibration of market_share with new scale and renovation rate
                self.constant_insulation_intensive = None
                _, utility_intensive = to_market_share(bill_saved, subsidies_total, cost_total)
                self.constant_insulation_intensive = calibration_intensive(utility_intensive, stock, ms_insulation_ini,
                                                                           renovation_rate)
                # test market-share (test with other renovation rate so can differ)
                market_share, utility_intensive = to_market_share(bill_saved, subsidies_total, cost_total)

                # test renovation_rate (test with other utility extensive so can differ)
                investment_insulation, bill_saved_insulation, subsidies_insulation = to_utility_extensive(cost_total,
                                                                                                          bill_saved,
                                                                                                          subsidies_total,
                                                                                                          market_share,
                                                                                                          utility_intensive)

                renovation_rate, _ = to_retrofit_rate(bill_saved_insulation, subsidies_insulation, investment_insulation)

                flow = renovation_rate * stock
                rate = flow.groupby(renovation_rate_ini.index.names).sum() / stock.groupby(
                    renovation_rate_ini.index.names).sum()
                compare_rate = concat((rate.rename('Calculated'), renovation_rate_ini.rename('Observed')), axis=1).round(3)
                flow_insulation = (flow * market_share.T).T.sum()
                share = flow_insulation / flow_insulation.sum()
                compare_share = concat((share.rename('Calculated'), ms_insulation_ini.rename('Observed')), axis=1).round(3)

                compare = concat((compare_rate, compare_share), ignore_index=True)
                if (compare['Calculated'] == compare['Observed']).all():
                    self.logger.debug('Coupled optim worked')
                    break

            if self.path is not None:
                compare.to_csv(os.path.join(self.path_calibration, 'result_calibration.csv'))

            result = assess_policies(stock, subsidies_details, cost_total, bill_saved, subsidies_total)
            if self.path is not None:
                result.to_csv(os.path.join(self.path_calibration, 'result_policies_assessment.csv'))

            assess_sensitivity(stock, cost_total, bill_saved, subsidies_total, self.path_calibration)

        market_share, renovation_rate = apply_endogenous_retrofit(bill_saved, subsidies_total, cost_total)

        # min_performance = calib_intensive.get('minimum_performance')
        if min_performance is not None:
            certificate = reindex_mi(self.certificate_building_choice, market_share.index)
            market_share = market_share[certificate <= min_performance]
            market_share = (market_share.T / market_share.sum(axis=1)).T

        return renovation_rate, market_share

    def exogenous_retrofit(self, stock):
        """Format retrofit rate and market share for each segment.


        Global retrofit and retrofit rate to match exogenous numbers.
        Retrofit all heating system replacement dwelling.

        Parameters
        ----------
        stock: Series

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
            consumption = reindex_mi(self.consumption_sd_building, stock.index)
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
        market_share = self.best_option.astype(int)
        market_share = reindex_mi(market_share, retrofit_rate.index).dropna()

        assert market_share.loc[market_share.sum(axis=1) != 1].empty, 'Market share problem'

        return retrofit_rate, market_share

    def flow_obligation(self, policies_insulation, prices, cost_insulation,
                        financing_cost=True):
        """Account for flow obligation if defined in policies_insulation.

        Parameters
        ----------
        policies_insulation: list
            Check if obligation.
        prices: Series
        cost_insulation: Series
        rotation: Series, optional
        financing_cost: bool, optional

        Returns
        -------
        flow_obligation: Series
        """

        stock = self.stock_mobile.copy()

        obligation = [p for p in policies_insulation if p.name == 'obligation']
        if obligation == []:
            return None
        # only work if there is one obligation
        obligation = obligation[0]
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
        replaced_by.index = replaced_by.index.reorder_levels(self.market_share.index.names)

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
            self.store_information_retrofit(replaced_by)

        replaced_by = self.frame_to_flow(replaced_by)

        assert to_replace.sum().round(0) == replaced_by.sum().round(0), 'Sum problem'
        flow_obligation = concat((- to_replace, replaced_by), axis=0)
        flow_obligation = flow_obligation.groupby(flow_obligation.index.names).sum()
        return flow_obligation

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

    def flow_retrofit(self, prices, cost_heater, cost_insulation, policies_heater=None, policies_insulation=None,
                      ms_heater=None, financing_cost=None, calib_renovation=None, calib_intensive=None):
        """Compute heater replacement and insulation retrofit.


        1. Heater replacement based on current stock segment.
        2. Knowing heater replacement (and new heating system) calculating retrofit rate by segment and market
        share by segment.
        3. Then, managing inflow and outflow.

        Parameters
        ----------
        prices: Series
        cost_heater: Series
        ms_heater: DataFrame
        cost_insulation
        policies_heater: list
            Policies for heating system.
        policies_insulation: list
            Policies for insulation.
        financing_cost: optional, dict
        calib_renovation: dict, optional
        calib_intensive: dict, optional

        Returns
        -------
        Series
        """

        # store consumption before retrofit
        self.consumption_before_retrofit = self.store_consumption(prices)

        stock = self.stock_mobile.groupby([i for i in self.stock_mobile.index.names if i != 'Income tenant']).sum()

        # accounts for heater replacement - depends on energy prices, cost and policies heater
        stock = self.heater_replacement(stock, prices, cost_heater, policies_heater, ms_heater=ms_heater)

        self.logger.debug('Agents: {:,.0f}'.format(stock.shape[0]))
        stock_existing = stock.xs(True, level='Existing', drop_level=False)
        retrofit_rate, market_share = self.insulation_replacement(stock_existing, prices, cost_insulation,
                                                                  calib_renovation=calib_renovation,
                                                                  calib_intensive=calib_intensive,
                                                                  policies_insulation=policies_insulation,
                                                                  financing_cost=financing_cost)
        # heater replacement without insulation upgrade
        flow_only_heater = (1 - retrofit_rate.reindex(stock.index).fillna(0)) * stock
        flow_only_heater = flow_only_heater.xs(True, level='Heater replacement', drop_level=False).unstack('Heating system final')
        flow_only_heater_sum = flow_only_heater.sum().sum()

        # insulation upgrade
        flow = (retrofit_rate * stock).dropna()
        replacement_sum = flow.sum().sum()

        replaced_by = (flow * market_share.T).T

        assert round(replaced_by.sum().sum(), 0) == round(replacement_sum, 0), 'Sum problem'

        # energy performance certificate jump due to heater replacement without insulation upgrade
        only_heater = (stock - flow.reindex(stock.index, fill_value=0)).xs(True, level='Heater replacement')
        certificate_jump = self.certificate_jump_heater.stack()
        rslt = {}
        l = unique(certificate_jump)
        for i in l:
            rslt.update({i: ((certificate_jump == i) * only_heater).sum()})
        self.certificate_jump_heater = Series(rslt).sort_index()

        # storing information (flow, investment, subsidies)
        if self.full_output:
            self.store_information_retrofit(replaced_by)

        # removing heater replacement level
        replaced_by = replaced_by.groupby(
            [c for c in replaced_by.index.names if c != 'Heater replacement']).sum()
        flow_only_heater = flow_only_heater.groupby(
            [c for c in flow_only_heater.index.names if c != 'Heater replacement']).sum()

        # adding income tenant information
        share = (self.stock_mobile.unstack('Income tenant').T / self.stock_mobile.unstack('Income tenant').sum(axis=1)).T
        temp = concat([replaced_by] * share.shape[1], keys=share.columns, names=share.columns.names, axis=1)
        share = reindex_mi(share, temp.columns, axis=1)
        share = reindex_mi(share, temp.index)
        replaced_by = (share * temp).stack('Income tenant').dropna()
        assert round(replaced_by.sum().sum(), 0) == round(replacement_sum, 0), 'Sum problem'

        share = (self.stock_mobile.unstack('Income tenant').T / self.stock_mobile.unstack('Income tenant').sum(axis=1)).T
        temp = concat([flow_only_heater] * share.shape[1], keys=share.columns, names=share.columns.names, axis=1)
        share = reindex_mi(share, temp.columns, axis=1)
        share = reindex_mi(share, temp.index)
        flow_only_heater = (share * temp).stack('Income tenant').dropna()
        assert round(flow_only_heater.sum().sum(), 0) == round(flow_only_heater_sum, 0), 'Sum problem'

        flow_only_heater = flow_only_heater.stack('Heating system final')
        to_replace_only_heater = - flow_only_heater.droplevel('Heating system final')

        flow_replaced_by = flow_only_heater.droplevel('Heating system')
        flow_replaced_by.index = flow_replaced_by.index.rename('Heating system', level='Heating system final')
        flow_replaced_by = flow_replaced_by.reorder_levels(to_replace_only_heater.index.names)

        flow_only_heater = pd.concat((to_replace_only_heater, flow_replaced_by), axis=0)
        flow_only_heater = flow_only_heater.groupby(flow_only_heater.index.names).sum()
        assert round(flow_only_heater.sum(), 0) == 0, 'Sum problem'

        to_replace = replaced_by.droplevel('Heating system final').sum(axis=1).copy()
        to_replace = to_replace.groupby(to_replace.index.names).sum()
        assert round(to_replace.sum(), 0) == round(replacement_sum, 0), 'Sum problem'

        replaced_by = self.frame_to_flow(replaced_by)

        to_replace = to_replace.reorder_levels(replaced_by.index.names)
        flow_only_heater = flow_only_heater.reorder_levels(replaced_by.index.names)
        flow_retrofit = concat((-to_replace, replaced_by, flow_only_heater), axis=0)
        flow_retrofit = flow_retrofit.groupby(flow_retrofit.index.names).sum()

        assert round(flow_retrofit.sum(), 0) == 0, 'Sum problem'

        return flow_retrofit

    def flow_demolition(self):
        """Demolition of E, F and G buildings based on their share in the mobile stock.

        Returns
        -------
        Series
        """
        self.logger.info('Demolition')
        stock_demolition = self.stock_mobile[self.certificate.isin(self._target_demolition)]
        if stock_demolition.sum() < self._demolition_total:
            self._target_demolition = ['G', 'F', 'E', 'D']
            stock_demolition = self.stock_mobile[self.certificate.isin(self._target_demolition)]
            if stock_demolition.sum() < self._demolition_total:
                self._target_demolition = ['G', 'F', 'E', 'D', 'C']
                stock_demolition = self.stock_mobile[self.certificate.isin(self._target_demolition)]

        stock_demolition = stock_demolition / stock_demolition.sum()
        flow_demolition = (stock_demolition * self._demolition_total).dropna()
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

    def parse_output_run(self, inputs):
        """Parse output.

        Renovation : envelope
        Retrofit : envelope and/or heating system

        Parameters
        ----------
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

        Returns
        -------

        """

        stock = self.simplified_stock()

        output = dict()
        output['Consumption standard (TWh)'] = (self.consumption_heat_sd * self.surface * self.stock).sum() / 10 ** 9

        consumption = self.heat_consumption_calib
        output['Consumption (TWh)'] = consumption.sum() / 10 ** 9

        temp = consumption.groupby(self.energy).sum()
        temp.index = temp.index.map(lambda x: 'Consumption {} (TWh)'.format(x))
        output.update(temp.T / 10 ** 9)

        temp = consumption.groupby('Existing').sum()
        temp.rename(index={True: 'Existing', False: 'New'}, inplace=True)
        temp.index = temp.index.map(lambda x: 'Consumption {} (TWh)'.format(x))
        output.update(temp.T / 10 ** 9)

        temp = consumption.groupby(self.certificate).sum()
        temp.index = temp.index.map(lambda x: 'Consumption {} (TWh)'.format(x))
        output.update(temp.T / 10 ** 9)
        if self.consumption_saving_retrofit is not None:
            temp = {'{} saving'.format(k): i for k, i in self.consumption_saving_retrofit.items()}
            output.update(temp)

        c = self.add_energy(consumption)
        emission = reindex_mi(inputs['carbon_emission'].T.rename_axis('Energy', axis=0), c.index).loc[:,
                   self.year] * c

        output['Emission (MtCO2)'] = emission.sum() / 10 ** 12

        temp = emission.groupby('Existing').sum()
        temp.rename(index={True: 'Existing', False: 'New'}, inplace=True)
        temp.index = temp.index.map(lambda x: 'Emission {} (MtCO2)'.format(x))
        output.update(temp.T / 10 ** 12)

        temp = emission.groupby('Energy').sum()
        temp.index = temp.index.map(lambda x: 'Emission {} (MtCO2)'.format(x))
        output.update(temp.T / 10 ** 12)

        output['Stock (Million)'] = stock.sum() / 10 ** 6

        output['Surface (Million m2)'] = (self.stock * self.surface).sum() / 10 ** 6
        output['Surface (m2/person)'] = (
                    output['Surface (Million m2)'] / (inputs['population'].loc[self.year] / 10 ** 6))

        output['Consumption standard (kWh/m2)'] = (output['Consumption standard (TWh)'] * 10 ** 9) / (
                output['Surface (Million m2)'] * 10 ** 6)
        output['Consumption (kWh/m2)'] = (output['Consumption (TWh)'] * 10 ** 9) / (
                output['Surface (Million m2)'] * 10 ** 6)

        output['Heating intensity (%)'] = (self.stock * self.heating_intensity).sum() / self.stock.sum()

        output['Energy poverty (Million)'] = self.energy_poverty / 10 ** 6

        temp = self.stock.groupby(self.certificate).sum()
        temp.index = temp.index.map(lambda x: 'Stock {} (Million)'.format(x))
        output.update(temp.T / 10 ** 6)
        try:
            output['Stock efficient (Million)'] = output['Stock A (Million)'] + output['Stock B (Million)']
        except KeyError:
            output['Stock efficient (Million)'] = output['Stock B (Million)']

        output['Stock low-efficient (Million)'] = 0
        if 'Stock F (Million)' in output.keys():
            output['Stock low-efficient (Million)'] += output['Stock F (Million)']
        if 'Stock G (Million)' in output.keys():
            output['Stock low-efficient (Million)'] += output['Stock G (Million)']

        if self.year > self.first_year:
            temp = self.retrofit_rate.dropna(how='all')
            temp = temp.groupby([i for i in temp.index.names if i not in ['Heating system final']]).mean()

            if False in temp.index.get_level_values('Heater replacement'):
                t = temp.xs(False, level='Heater replacement')
                s_temp = self.stock
                s_temp = s_temp.groupby([i for i in s_temp.index.names if i != 'Income tenant']).sum()

                # Weighted average with stock to calculate real retrofit rate
                # TODO: doesn't work because self.stock have changed already - calculate before or remove
                output['Renovation rate (%)'] = ((t * s_temp).sum() / s_temp.sum())
                t_grouped = (t * s_temp).groupby(['Housing type', 'Occupancy status']).sum() / s_temp.groupby(
                    ['Housing type',
                     'Occupancy status']).sum()
                t_grouped.index = t_grouped.index.map(lambda x: 'Renovation rate {} - {} (%)'.format(x[0], x[1]))
                output.update(t_grouped.T)

            if True in temp.index.get_level_values('Heater replacement'):
                t = temp.xs(True, level='Heater replacement')
                s_temp = self.stock
                s_temp = s_temp.groupby([i for i in s_temp.index.names if i != 'Income tenant']).sum()
                output['Renovation rate w/ heater (%)'] = ((t * s_temp).sum() / s_temp.sum())

                t_grouped = (t * s_temp).groupby(['Housing type', 'Occupancy status']).sum() / s_temp.groupby(
                    ['Housing type',
                     'Occupancy status']).sum()
                t_grouped.index = t_grouped.index.map(lambda x: 'Renovation rate heater {} - {} (%)'.format(x[0], x[1]))
                output.update(t_grouped.T)

            # self.gest_nb: number of renovation types (number of insulated components by renovation)
            temp = self.gest_nb.copy()
            temp.index = temp.index.map(lambda x: 'Renovation types {} (Thousand households)'.format(x))
            output['Renovation (Thousand households)'] = temp.sum() / 10 ** 3
            output['Renovation with heater replacement (Thousand households)'] = self.retrofit_with_heater / 10 ** 3
            output['Replacement (Thousand renovating)'] = (self.gest_nb * self.gest_nb.index).sum() / 10 ** 3
            output.update(temp.T / 10 ** 3)
            output['Replacement total (Thousand)'] = output['Replacement (Thousand renovating)'] - output[
                'Renovation with heater replacement (Thousand households)'] + self.replacement_heater.sum().sum() / 10 ** 3

            output['Retrofit (Thousand households)'] = output['Renovation (Thousand households)'] - output[
                'Renovation with heater replacement (Thousand households)'] + self.replacement_heater.sum().sum() / 10 ** 3

            # output['Renovation (Thousand households)'] = self.certificate_jump.sum().sum() / 10 ** 3
            # We need them by income for freeriders ratios per income deciles

            # TODO: optimizing: only need certificate_jump summed or by index by not dataframe

            temp = self.certificate_jump_all.sum().squeeze().sort_index()
            temp = temp[temp.index.dropna()]
            o = {}
            for i in temp.index.union(self.certificate_jump_heater.index):
                t_renovation = 0
                if i in temp.index:
                    t_renovation = temp.loc[i]
                    o['Renovation {} EPC (Thousand households)'.format(i)] = t_renovation / 10 ** 3
                t_heater = 0
                if i in self.certificate_jump_heater.index:
                    t_heater = self.certificate_jump_heater.loc[i]
                o['Retrofit {} EPC (Thousand households)'.format(i)] = (t_renovation + t_heater) / 10 ** 3
            o = Series(o).sort_index(ascending=False)

            output['Renovation >= 1 EPC (Thousand households)'] = self.certificate_jump_all.loc[:,
                                                     [i for i in self.certificate_jump_all.columns if
                                                      i > 0]].sum().sum() / 10 ** 3
            output['Retrofit >= 1 EPC (Thousand households)'] = sum([o['Retrofit {} EPC (Thousand households)'.format(i)] for i in temp.index.unique() if i >=1])

            output.update(o.T)
            # output['Retrofit rate {} EPC (%)'.format(i)] = temp.sum() / stock.sum()

            # output['Efficient retrofits (Thousand)'] = Series(self.efficient_renovation_yrs) / 10**3
            output['Global renovation high income (Thousand households)'] = self.global_renovation_high_income / 10 ** 3
            output['Global renovation low income (Thousand households)'] = self.global_renovation_low_income / 10 ** 3
            output['Global renovation (Thousand households)'] = output['Global renovation high income (Thousand households)'] + output['Global renovation low income (Thousand households)']
            output['Bonus best renovation (Thousand households)'] = self.bonus_best / 10 ** 3
            output['Bonus worst renovation (Thousand households)'] = self.bonus_worst / 10 ** 3
            output['Percentage of global renovation (% households)'] = output['Global renovation (Thousand households)'] / output[
                'Renovation (Thousand households)']
            output['Percentage of bonus best renovation (% households)'] = output['Bonus best renovation (Thousand households)'] / output[
                'Renovation (Thousand households)']
            output['Percentage of bonus worst renovation (% households)'] = output['Bonus worst renovation (Thousand households)'] / output[
                'Renovation (Thousand households)']

            temp = self.certificate_jump_all.sum(axis=1)
            t = temp.groupby('Income owner').sum()
            t.index = t.index.map(lambda x: 'Renovation {} (Thousand households)'.format(x))
            output.update(t.T / 10 ** 3)

            # for replacement output need to be presented by technologies (what is used) and by agent (who change)
            temp = self.replacement_heater.sum()
            output['Replacement heater (Thousand households)'] = temp.sum() / 10 ** 3
            heat_pump = ['Electricity-Heat pump water', 'Electricity-Heat pump air']
            output['Replacement Heat pump (Thousand households)'] = temp[heat_pump].sum() / 10 ** 3

            heater_efficient = heat_pump + ['Wood fuel-Performance boiler', 'Natural gas-Performance boiler']
            output['Replacement Heater efficient (Thousand households)'] = temp[heater_efficient].sum().sum() / 10 ** 3
            t = temp.copy()
            t.index = t.index.map(lambda x: 'Replacement heater {} (Thousand households)'.format(x))
            output.update((t / 10 ** 3).T)

            """
            # summing accoridng to heating system beafore instead of final 
            temp = self.replacement_heater.sum(axis=1) 
            t = temp.groupby(['Heating system', 'Housing type']).sum()
            t.index = t.index.map(lambda x: 'Replacement heater {} {} (Thousand households)'.format(x[0], x[1]))
            output.update((t / 10 ** 3).T)
            """

            t = self.replacement_heater.groupby('Housing type').sum().loc['Multi-family']
            t.index = t.index.map(lambda x: 'Replacement heater Multi-family {} (Thousand households)'.format(x))
            output.update((t / 10 ** 3).T)

            t = self.replacement_heater.groupby('Housing type').sum().loc['Single-family']
            t.index = t.index.map(lambda x: 'Replacement heater Single-family {} (Thousand households)'.format(x))
            output.update((t / 10 ** 3).T)

            temp = self.replacement_insulation.sum(axis=1)
            output['Replacement insulation (Thousand households)'] = temp.sum() / 10 ** 3
            t = temp.groupby(['Housing type']).sum()
            t.index = t.index.map(lambda x: 'Replacement insulation {} (Thousand households)'.format(x))
            output.update((t / 10 ** 3).T)

            t = temp.groupby(['Housing type', 'Occupancy status']).sum()
            t.index = t.index.map(lambda x: 'Replacement insulation {} - {} (Thousand households)'.format(x[0], x[1]))
            output.update((t / 10 ** 3).T)
            t = temp.groupby('Income owner').sum()
            t.index = t.index.map(lambda x: 'Replacement insulation {} (Thousand households)'.format(x))
            output.update(t.T / 10 ** 3)

            """t.index = t.index.str.replace('Thousand', '%')
            s = stock.groupby(['Housing type', 'Occupancy status']).sum()
            s.index = s.index.map(lambda x: 'Replacement insulation {} - {} (%)'.format(x[0], x[1]))
            t = t / s
            output.update(t.T)"""
            o = {}
            for i in ['Wall', 'Floor', 'Roof', 'Windows']:
                temp = self.replacement_insulation.xs(True, level=i, axis=1).sum(axis=1)
                o['Replacement {} (Thousand households)'.format(i)] = temp.sum() / 10 ** 3

                cost = self.cost_component.loc[:, i]
                t = reindex_mi(cost, temp.index) * temp
                surface = reindex_mi(inputs['surface'].loc[:, self.year], t.index)
                o['Investment {} (Billion euro)'.format(i)] = (t * surface).sum() / 10 ** 9

                surface = reindex_mi(inputs['surface'].loc[:, self.year], temp.index)
                o['Embodied energy {} (TWh PE)'.format(i)] = (temp * surface *
                                                                   inputs['embodied_energy_renovation'][
                                                                       i]).sum() / 10 ** 9
                o['Carbon footprint {} (MtCO2)'.format(i)] = (temp * surface *
                                                                   inputs['carbon_footprint_renovation'][
                                                                       i]).sum() / 10 ** 9
            output['Replacement insulation (Thousand)'] = sum(
                [o['Replacement {} (Thousand households)'.format(i)] for i in
                 ['Wall', 'Floor', 'Roof', 'Windows']])

            o = Series(o).sort_index(ascending=False)
            output.update(o.T)

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

            temp = self.investment_heater.sum()
            output['Investment heater (Billion euro)'] = temp.sum() / 10 ** 9
            temp.index = temp.index.map(lambda x: 'Investment {} (Billion euro)'.format(x))
            output.update(temp.T / 10 ** 9)
            investment_heater = self.investment_heater.sum(axis=1)


            investment_insulation = self.investment_insulation.sum(axis=1)
            output['Investment insulation (Billion euro)'] = investment_insulation.sum() / 10 ** 9

            index = investment_heater.index.union(investment_insulation.index)
            investment_total = investment_heater.reindex(index, fill_value=0) + investment_insulation.reindex(index,
                                                                                                              fill_value=0)
            output['Investment total (Billion euro)'] = investment_total.sum() / 10 ** 9
            temp = investment_total.groupby('Income owner').sum()
            temp.index = temp.index.map(lambda x: 'Investment total {} (Billion euro)'.format(x))
            output.update(temp.T / 10 ** 9)
            temp = investment_total.groupby(['Housing type', 'Occupancy status']).sum()
            temp.index = temp.index.map(lambda x: 'Investment total {} - {} (Billion euro)'.format(x[0], x[1]))
            output.update(temp.T / 10 ** 9)

            subsidies_heater = self.subsidies_heater.sum(axis=1)
            output['Subsidies heater (Billion euro)'] = subsidies_heater.sum() / 10 ** 9

            subsidies_insulation = self.subsidies_insulation.sum(axis=1)
            output['Subsidies insulation (Billion euro)'] = subsidies_insulation.sum() / 10 ** 9

            index = subsidies_heater.index.union(subsidies_insulation.index)
            subsidies_total = subsidies_heater.reindex(index, fill_value=0) + subsidies_insulation.reindex(index,
                                                                                                           fill_value=0)
            output['Subsidies total (Billion euro)'] = subsidies_total.sum() / 10 ** 9
            temp = subsidies_total.groupby('Income owner').sum()
            temp.index = temp.index.map(lambda x: 'Subsidies total {} (Million euro)'.format(x))
            output.update(temp.T / 10 ** 6)
            temp = subsidies_total.groupby(['Housing type', 'Occupancy status']).sum()
            temp.index = temp.index.map(lambda x: 'Subsidies total {} - {} (Million euro)'.format(x[0], x[1]))
            output.update(temp.T / 10 ** 6)

            # policies amount and number of beneficiaries
            subsidies, subsidies_count, sub_count = None, None, None
            for gest, subsidies_details in {'heater': self.subsidies_details_heater,
                                            'insulation': self.subsidies_details_insulation}.items():
                if gest == 'heater':
                    sub_count = DataFrame(self.subsidies_count_heater, dtype=float)
                elif gest == 'insulation':
                    sub_count = DataFrame(self.subsidies_count_insulation, dtype=float)

                subsidies_details = Series({k: i.sum().sum() for k, i in subsidies_details.items()}, dtype='float64')

                for i in subsidies_details.index:
                    temp = sub_count[i]
                    temp.index = temp.index.map(lambda x: '{} {} {} (Thousand households)'.format(i.capitalize().replace('_', ' '), gest, x))
                    output.update(temp.T / 10 ** 3)
                    # output['{} {} (Thousand)'.format(i.capitalize().replace('_', ' '), gest)] =
                    output['{} {} (Billion euro)'.format(i.capitalize().replace('_', ' '), gest)] = \
                    subsidies_details.loc[i] / 10 ** 9
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
                    output['{} (Thousand households)'.format(i.capitalize().replace('_', ' '))] = temp.sum() / 10**3
                    temp.index = temp.index.map(lambda x: '{} {} (Thousand households)'.format(i.capitalize().replace('_', ' '), x))
                    output.update(temp.T / 10 ** 3)
                    output['{} (Billion euro)'.format(i.capitalize().replace('_', ' '))] = subsidies.loc[i] / 10 ** 9

            # output['Zero interest loan headcount'] = self.zil_count
            # output['Zero interest loan average amount'] = self.zil_loaned_avg
            taxes_expenditures = self.taxes_expenditure_details
            taxes_expenditures = DataFrame(taxes_expenditures).sum()
            taxes_expenditures.index = taxes_expenditures.index.map(
                lambda x: '{} (Billion euro)'.format(x.capitalize().replace('_', ' ').replace('Cee', 'Cee tax')))
            output.update((taxes_expenditures / 10 ** 9).T)
            output['Taxes expenditure (Billion euro)'] = taxes_expenditures.sum() / 10 ** 9

            energy_expenditure = self.energy_expenditure
            output['Energy expenditures (Billion euro)'] = energy_expenditure.sum() / 10 ** 9
            temp = energy_expenditure.groupby('Income tenant').sum()
            temp.index = temp.index.map(lambda x: 'Energy expenditures {} (Billion euro)'.format(x))
            output.update(temp.T / 10 ** 9)

            output['VTA heater (Billion euro)'] = self.tax_heater.sum().sum() / 10 ** 9

            output['VTA insulation (Billion euro)'] = self.taxed_insulation.sum().sum() / 10 ** 9
            output['VTA (Billion euro)'] = output['VTA heater (Billion euro)'] + output['VTA insulation (Billion euro)']

            output['Investment total HT (Billion euro)'] = output['Investment total (Billion euro)'] - output[
                'VTA (Billion euro)']

            output['Carbon value (Billion euro)'] = (self.heat_consumption_energy * inputs['carbon_value_kwh'].loc[
                                                                                         self.year,
                                                                                         :]).sum() / 10 ** 9

            output['Health cost (Billion euro)'], o = self.health_cost(inputs)
            output.update(o)

            output['Income state (Billion euro)'] = output['VTA (Billion euro)'] + output[
                'Taxes expenditure (Billion euro)']
            output['Expenditure state (Billion euro)'] = output['Subsidies heater (Billion euro)'] + output[
                'Subsidies insulation (Billion euro)']
            output['Balance state (Billion euro)'] = output['Income state (Billion euro)'] - output[
                'Expenditure state (Billion euro)']

            levels = ['Occupancy status', 'Income owner', 'Housing type']
            for level in levels:
                temp = subsidies_total.groupby(level).sum() / investment_total.groupby(level).sum()
                temp.index = temp.index.map(lambda x: 'Share subsidies {} (%)'.format(x))
                output.update(temp.T)

            output['Investment total HT / households (Thousand euro)'] = output['Investment total HT (Billion euro)'] * 10**6 / (output['Retrofit (Thousand households)'] * 10**3)
            output['Investment total / households (Thousand euro)'] = output['Investment total (Billion euro)'] * 10**6 / (output['Retrofit (Thousand households)'] * 10**3)
            output['Investment insulation / households (Thousand euro)'] = output['Investment insulation (Billion euro)'] * 10**6 / (output['Renovation (Thousand households)'] * 10**3)

        output = Series(output).rename(self.year)
        stock = stock.rename(self.year)
        return stock, output

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

        consumption_before = self.consumption_standard(index)[0]
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

    def remove_calibration(self):
        self.preferences_insulation_int['subsidy'] /= self.scale
        self.preferences_insulation_int['investment'] /= self.scale
        self.preferences_insulation_int['bill_saved'] /= self.scale
        self.preferences_insulation_ext['subsidy'] /= self.scale
        self.preferences_insulation_ext['investment'] /= self.scale
        self.preferences_insulation_ext['bill_saved'] /= self.scale
        self.coefficient_consumption = None
        self.constant_heater = None
        self.constant_insulation_intensive = None
        self.constant_insulation_extensive = None
        self.scale = None

    def apply_scale(self, scale):

        self.scale *= scale
        self.preferences_insulation_ext['subsidy'] *= scale
        self.preferences_insulation_ext['investment'] *= scale
        self.preferences_insulation_ext['bill_saved'] *= scale

        self.preferences_insulation_int['subsidy'] *= scale
        self.preferences_insulation_int['investment'] *= scale
        self.preferences_insulation_int['bill_saved'] *= scale

    def calibration_exogenous(self, coefficient_consumption=None, constant_heater=None,
                              constant_insulation_intensive=None, constant_insulation_extensive=None, scale=None,
                              energy_prices=None, taxes=None):
        """Function calibrating buildings object with exogenous data.


        Parameters
        ----------
        coefficient_consumption: Series
        constant_heater: Series
        constant_insulation_intensive: Series
        constant_insulation_extensive: Series
        energy_prices: Series
            Energy prices for year y. Index are energy carriers {'Electricity', 'Natural gas', 'Oil fuel', 'Wood fuel'}.
        taxes: Series
            Energy taxes for year y.
        """

        # calibration energy consumption first year
        if (coefficient_consumption is None) and (energy_prices is not None) and (taxes is not None):
            self.calculate_consumption(energy_prices.loc[self.first_year, :], taxes)
        else:
            self.coefficient_consumption = coefficient_consumption

        if constant_heater is None:
            constant_heater = get_pandas('project/input/calibration/calibration_constant_heater.csv',
                                                     lambda x: pd.read_csv(x, index_col=[0, 1, 2]).squeeze())
            constant_heater = constant_heater.unstack('Heating system final')

        elif isinstance(constant_heater, str):
            constant_heater = read_csv(constant_heater, index_col=[0, 1, 2]).squeeze()
            constant_heater = constant_heater.unstack('Heating system final')

        self.constant_heater = constant_heater
        self._choice_heater = list(self.constant_heater.columns)

        if constant_insulation_intensive is None:
            constant_insulation_intensive = get_pandas('project/input/calibration/calibration_constant_insulation.csv',
                                                       lambda x: pd.read_csv(x, index_col=[0, 1, 2, 3]).squeeze())
        elif isinstance(constant_insulation_intensive, str):
            constant_insulation_intensive = read_csv(constant_insulation_intensive, index_col=[0, 1, 2, 3]).squeeze()

        self.constant_insulation_intensive = constant_insulation_intensive.dropna()

        if constant_insulation_extensive is None:
            constant_insulation_extensive = get_pandas('project/input/calibration/calibration_constant_extensive.csv',
                                                         lambda x: pd.read_csv(x, index_col=[0, 1, 2, 3]).squeeze())
        elif isinstance(constant_insulation_extensive, str):
            constant_insulation_extensive = read_csv(constant_insulation_extensive, index_col=[0, 1, 2, 3]).squeeze()

        self.constant_insulation_extensive = constant_insulation_extensive.dropna()

        self.scale = scale
