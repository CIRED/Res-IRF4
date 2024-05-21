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
import sys

import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame, MultiIndex, Index, IndexSlice, concat, to_numeric, unique
from numpy import exp, log, append, array, allclose
from numpy.testing import assert_almost_equal
from scipy.optimize import fsolve
import logging
from copy import deepcopy
from itertools import product

from project.utils import make_plot, reindex_mi, make_plots, calculate_annuities, deciles2quintiles_dict, size_dict, \
    get_size, compare_bar_plot, make_sensitivity_tables, cumulated_plot, find_discount_rate, conditional_expectation, get_json
from project.utils import make_hist, reverse_dict, select, make_grouped_scatterplots, calculate_average, make_scatter_plot, add_no_renovation
from project.utils import factor_annuities

import project.thermal as thermal

ACCURACY = 10 ** -5
EPC2INT = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
VAT = 0.1
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

    Attributes:
    ----------

    """

    def __init__(self, stock, surface, ratio_surface, efficiency, income, path=None, year=2018,
                 resources_data=None, detailed_output=None, figures=None, residual_rate=0, temp_sink=None):

        # default values
        self.heating_intensity_max = None
        self.hi_threshold = None
        if figures is None:
            figures = True
        if detailed_output is None:
            detailed_output = True

        if isinstance(stock, MultiIndex):
            stock = Series(index=stock, dtype=float)

        self._resources_data = resources_data

        self._efficiency_yrs = efficiency
        if isinstance(efficiency, DataFrame):
            self._efficiency = self._efficiency_yrs.loc[:, year]
        else:
            self._efficiency = efficiency

        self._temp_sink_yrs = temp_sink
        if temp_sink is not None:
            self._temp_sink = self._temp_sink_yrs.loc[year]
        else:
            self._temp_sink = None

        self._ratio_surface = ratio_surface
        self.path, self.path_ini, self.path_calibration = path, None, None

        if path is not None and detailed_output:
            self.path_calibration = os.path.join(path, 'calibration')
            if not os.path.isdir(self.path_calibration):
                os.mkdir(self.path_calibration)
            if figures:
                self.path_ini = os.path.join(path, 'ini')
                if not os.path.isdir(self.path_ini):
                    os.mkdir(self.path_ini)

        self.coefficient_global, self.coefficient_backup = None, None

        self._surface_yrs = surface
        self._surface = surface.loc[:, year]

        self._income_yrs = income.copy()
        # self._income_ini = income.loc[:, year].copy()
        self._income, self._income_owner, self._income_tenant = None, None, None
        self.income = self._income_yrs.loc[:, year]

        self._residual_rate = residual_rate
        self._stock_residual = self._residual_rate * stock
        self.stock_mobile = stock - self._stock_residual

        self.first_year = year
        self._year = year

        self.taxes_list = []

        self.consumption_before_retrofit = None
        self.intensity_before_retrofit = None

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
        if self.path_ini is not None:
            stock = self.add_certificate(stock).groupby('Performance').sum() / 10 ** 6
            compare_performance = concat((stock, self._resources_data['performance_stock']), axis=1,
                                         keys=['Model', 'SDES-2018'])
            compare_bar_plot(compare_performance, 'Stock by performance (Million dwelling)',
                             save=os.path.join(self.path_ini, 'stock_performance.png'))

    @property
    def year(self):
        return self._year

    @year.setter
    def year(self, year):
        self._year = year
        self._surface = self._surface_yrs.loc[:, year]
        self.consumption_before_retrofit = None
        self.intensity_before_retrofit = None
        self.income = self._income_yrs.loc[:, year]
        if isinstance(self._efficiency_yrs, DataFrame):
            self._efficiency = self._efficiency_yrs.loc[:, year]
        if self._temp_sink_yrs is not None:
            self._temp_sink = self._temp_sink_yrs.loc[year]

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
        self.energy = self.to_energy(stock).astype('category')
        self._resources_data['index']['Energy'] = [i for i in self._resources_data['index']['Energy'] if
                                                   i in self.energy.unique()]

        consumption_sd, _, certificate = self.consumption_heating_store(stock.index)
        self.certificate = reindex_mi(certificate, stock.index).astype('category')

    @property
    def income(self):
        return self._income

    @income.setter
    def income(self, income):
        self._income = income
        self._income_owner = self._income.copy()
        self._income_owner.index.rename('Income owner', inplace=True)
        self._income_tenant = self._income.copy()
        self._income_tenant.index.rename('Income tenant', inplace=True)

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
        energy = self.to_energy(stock).rename('Energy')
        stock = concat((stock, certificate, energy), axis=1).set_index(['Performance', 'Energy'], append=True).squeeze()
        if energy_level:
            stock = stock.groupby(
                ['Occupancy status', 'Income owner', 'Income tenant', 'Housing type', 'Heating system',
                 'Energy', 'Performance']).sum()

        else:
            stock = stock.groupby(
                ['Occupancy status', 'Income owner', 'Income tenant', 'Housing type', 'Heating system',
                 'Performance']).sum()

        stock = stock[stock > 0]
        return stock

    @staticmethod
    def to_energy(df, level_heater='Heating system'):
        return Series(df.index.get_level_values(level_heater), index=df.index).str.split('-').str[0].rename(
            'Energy')

    @staticmethod
    def to_emission(df, emission):
        if 'Heating system' in df.index.names:
            return emission.reindex(ThermalBuildings.to_energy(df)).set_axis(df.index) * df
        elif 'Energy' in df.index.names:
            return emission.reindex(df.index.get_level_values('Energy')).set_axis(df.index) * df

    def add_certificate(self, df):
        """Add energy performance certificate to df index.

        Parameters
        ----------
        df

        Returns
        -------

        """
        columns = None
        if isinstance(df, DataFrame):
            columns = df.columns
        certificate = self.certificate.rename('Performance')
        lvl = [i for i in certificate.index.names if i in df.index.names]
        certificate = certificate.groupby(lvl).first()

        certificate = reindex_mi(certificate, df.index)
        df = concat((df, certificate), axis=1).set_index('Performance', append=True).squeeze()

        if isinstance(df, DataFrame):
            df.columns = columns

        return df

    def add_energy(self, df):
        columns = None
        if isinstance(df, DataFrame):
            columns = df.columns
        energy = self.energy.rename('Energy')
        lvl = [i for i in energy.index.names if i in df.index.names]
        energy = energy.groupby(lvl).first()
        energy = reindex_mi(energy, df.index)
        df = concat((df, energy), axis=1).set_index('Energy', append=True).squeeze()

        if isinstance(df, DataFrame):
            df.columns = columns

        return df

    def need_heating(self, climate=2006, smooth=False, freq='year', hourly_profile=None, unit='kWh/y'):
        """Calculate heating need of the current building stock.

        Returns
        -------
        DataFrame
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
            if isinstance(heating_need, (Series, float, int)):
                heating_need = heating_need * self.stock * self.surface
            elif isinstance(heating_need, DataFrame):
                heating_need = (heating_need.T * self.stock * self.surface).T

            return heating_need

    def size_heater(self, index=None):
        if index is None:
            levels = ['Housing type', 'Heating system', 'Wall', 'Floor', 'Roof', 'Windows']
            index = self.stock.groupby(levels).sum().index

        levels_consumption = ['Wall', 'Floor', 'Roof', 'Windows', 'Housing type']
        _index = index.to_frame().loc[:, levels_consumption].set_index(levels_consumption).index
        _index = _index[~_index.duplicated()]

        wall = Series(_index.get_level_values('Wall'), index=_index)
        floor = Series(_index.get_level_values('Floor'), index=_index)
        roof = Series(_index.get_level_values('Roof'), index=_index)
        windows = Series(_index.get_level_values('Windows'), index=_index)

        size_heating_system = thermal.size_heating_system(wall, floor, roof, windows, self._ratio_surface.copy())
        size_heating_system = reindex_mi(size_heating_system, index)
        size_heating_system *= reindex_mi(self._surface, index)

        return size_heating_system / 1e3

    def consumption_heating(self, index=None, freq='year', climate=None, smooth=False,
                            full_output=False, efficiency_hour=False, level_heater='Heating system',
                            method='5uses', hourly_profile=None, temp_sink=None):
        """Calculation consumption standard of the current building stock [kWh/m2.a].

        Parameters
        ----------
        index
        freq
        climate
        smooth
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
                                                         efficiency_hour=efficiency_hour, hourly_profile=hourly_profile,
                                                         temp_sink=temp_sink)

        consumption = reindex_mi(consumption, index)

        if full_output is True:
            certificate, consumption_3uses = thermal.conventional_energy_3uses(wall, floor, roof, windows,
                                                                               self._ratio_surface.copy(),
                                                                               efficiency, _index,
                                                                               method=method)
            certificate = reindex_mi(certificate, index)
            consumption_3uses = reindex_mi(consumption_3uses, index)

            return consumption, certificate, consumption_3uses
        else:
            return consumption

    def consumption_heating_store(self, index, level_heater='Heating system', full_output=True):
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
            self._consumption_store['consumption_3uses'] = concat(
                (self._consumption_store['consumption_3uses'], consumption_3uses))
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

    def _to_heating_intensity(self, index, prices, consumption=None, level_heater='Heating system', bill_rebate=0,
                             full_output=False):
        """Calculate heating intensity of index based on energy prices.

        Parameters
        ----------
        index: MultiIndex or Index
        prices: Series
        consumption: Series, default None
        level_heater: {'Heating system', 'Heating system final'}
        bill_rebate: float or Series, default 0

        Returns
        -------
        Series
            Heating intensity
        """
        if consumption is None:
            consumption = reindex_mi(self.consumption_heating_store(index, full_output=False), index) * reindex_mi(
                self._surface, index)
        energy_bill = AgentBuildings.energy_bill(prices, consumption, level_heater=level_heater,
                                                 bill_rebate=bill_rebate)

        if isinstance(energy_bill, Series):
            budget_share = energy_bill / reindex_mi(self._income_tenant, index)
            heating_intensity = thermal.heat_intensity(budget_share)
        elif isinstance(energy_bill, DataFrame):
            budget_share = (energy_bill.T / reindex_mi(self._income_tenant, energy_bill.index)).T
            heating_intensity = thermal.heat_intensity(budget_share)
        else:
            raise NotImplemented

        if self.coefficient_global is not None:
            heating_intensity *= self.coefficient_global

        if not full_output:
            return heating_intensity
        else:
            return heating_intensity, budget_share

    def to_heating_intensity(self, index, prices, consumption=None, level_heater='Heating system', bill_rebate=0,
                             full_output=False, rho=5):
        """Calculate heating intensity of index based on energy prices.

        Parameters
        ----------
        index: MultiIndex or Index
        prices: Series
        consumption: Series, default None
        level_heater: {'Heating system', 'Heating system final'}
        bill_rebate: float or Series, default 0

        Returns
        -------
        Series
            Heating intensity
        """
        if consumption is None:
            consumption = reindex_mi(self.consumption_heating_store(index, full_output=False), index) * reindex_mi(
                self._surface, index)
        energy_bill = self.energy_bill(prices, consumption, level_heater=level_heater,
                                       bill_rebate=bill_rebate)
        if isinstance(energy_bill, Series):
            budget_share = energy_bill / reindex_mi(self._income_tenant, index)
        elif isinstance(energy_bill, DataFrame):
            budget_share = (energy_bill.T / reindex_mi(self._income_tenant, energy_bill.index)).T

        energy_bill[energy_bill < 0] = 0.1
        heating_intensity = energy_bill ** (-1/rho)
        heating_intensity[heating_intensity > self.heating_intensity_max] = self.heating_intensity_max

        if self.coefficient_global is not None:
            heating_intensity *= self.coefficient_global

        if not full_output:
            return heating_intensity
        else:
            return heating_intensity, budget_share

    def to_thermal_comfort(self, heating_intensity, rho=5):
        A = self.coefficient_global**rho * abs(self.preferences_insulation['cost']) / 1e3
        return A * heating_intensity**(1 - rho) / (1 - rho)

    def consumption_actual(self, prices, consumption=None, full_output=False, bill_rebate=0):
        """Space heating consumption based on standard space heating consumption and heating intensity (kWh/building.a).


        Space heating consumption is in kWh/building.y
        Equation is based on Allibe (2012).

        Parameters
        ----------
        prices: Series
        consumption: Series or None, optional
            kWh/building.a
        full_output: bool, default False
        bill_rebate: float, default 0

        Returns
        -------
        Series
        """

        if consumption is None:
            index = self.stock.index
            consumption = self.consumption_heating_store(index, full_output=False)
            consumption = reindex_mi(consumption, index) * reindex_mi(self._surface, index)
        else:
            consumption = consumption.copy()
            index = consumption.index

        heating_intensity, budget_share = self.to_heating_intensity(index, prices, consumption=consumption,
                                                                    full_output=True, bill_rebate=bill_rebate)
        consumption = consumption * heating_intensity

        if full_output is False:
            return consumption
        else:
            return consumption, heating_intensity, budget_share

    def consumption_agg(self, prices=None, freq='year', climate=None, smooth=False,
                        standard=False, efficiency_hour=False, existing=False, agg='all', bill_rebate=0,
                        hourly_profile=None):
        """Aggregated final energy consumption (TWh final energy).

        Parameters
        ----------
        prices: Series
            Energy prices.
        freq
        climate
        smooth
        standard: bool, default False
            If yes, consumption standard (conventional). Otherwise, consumption actual.
        efficiency_hour: bool, default False
        existing: bool, default False
            If yes, calculate consumption only for existing buildings.
        agg: {'all', 'energy', 'heater'}, default 'all'
            Aggregation level.
        bill_rebate: float, default 0
        hourly_profile: DataFrame, optional

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

                if agg == 'all':
                    return consumption.sum() / 10 ** 9
                elif agg == 'energy':
                    energy = self.energy.reindex(consumption.index)
                    return consumption.groupby(energy).sum() / 10 ** 9
                elif agg == 'heater':
                    return consumption.groupby('Heating system').sum() / 10 ** 9

        if standard is False:
            if freq == 'year':
                # TODO: if climate is none consumption_heating_store ?
                consumption = self.consumption_heating(freq=freq, climate=climate, temp_sink=self._temp_sink)
                consumption = reindex_mi(consumption, self.stock.index) * self.surface
                if existing is True:
                    consumption = consumption[consumption.index.get_level_values('Existing')]
                consumption = self.consumption_actual(prices, consumption=consumption, bill_rebate=bill_rebate) * self.stock

                if agg == 'all':
                    consumption = self.apply_calibration(consumption, agg='energy') / 10 ** 9
                    consumption = consumption.sum()
                else:
                    consumption = self.apply_calibration(consumption, agg=agg) / 10 ** 9

                return consumption

            if freq == 'hour':
                consumption = self.consumption_heating(freq=freq, climate=climate, smooth=smooth,
                                                       efficiency_hour=efficiency_hour, hourly_profile=hourly_profile,
                                                       temp_sink=self._temp_sink)
                consumption = (reindex_mi(consumption, self.stock.index).T * self.surface).T
                heating_intensity = self.to_heating_intensity(consumption.index, prices,
                                                              consumption=consumption.sum(axis=1),
                                                              bill_rebate=bill_rebate)
                consumption = (consumption.T * heating_intensity * self.stock).T
                consumption = self.apply_calibration(consumption)
                return consumption

    def apply_calibration(self, consumption, level_heater='Heating system', agg='energy'):
        if self.coefficient_global is None:
            raise AttributeError

        if level_heater == 'Heating system final':
            if 'Heating system' in consumption.index.names:
                consumption = consumption.rename_axis(index={'Heating system': 'Heating system before'})
            consumption = consumption.rename_axis(index={'Heating system final': 'Heating system'})

        consumption_heater = consumption.groupby('Heating system').sum()
        # consumption_heater *= self.coefficient_global, included in heating intensity

        if isinstance(consumption, Series):
            _consumption_heater = self.coefficient_backup * consumption_heater
            consumption_backup = (1 - self.coefficient_backup) * consumption_heater
            _consumption_energy = _consumption_heater.groupby(
                _consumption_heater.index.str.split('-').str[0].rename('Energy')).sum()
            _consumption_energy['Wood fuel'] += consumption_backup.sum()

            _consumption_heater['Wood fuel-Performance boiler'] += consumption_backup.sum()
            if agg == 'energy':
                return _consumption_energy
            elif agg == 'heater':
                return _consumption_heater
            else:
                raise ValueError

        elif isinstance(consumption, DataFrame):
            _consumption_heater = (consumption_heater.T * self.coefficient_backup).T
            consumption_backup = (consumption_heater.T * (1 - self.coefficient_backup)).T.sum()
            _consumption_energy = _consumption_heater.groupby(
                _consumption_heater.index.str.split('-').str[0].rename('Energy')).sum()
            _consumption_energy.loc['Wood fuel', :] += consumption_backup

            return _consumption_energy

    def calibration_consumption(self, prices, consumption_ini, health_cost_income, health_cost_dpe, climate=None):
        """Calculate energy indicators.

        Parameters
        ----------
        prices: Series
        consumption_ini: Series
        climate

        Returns
        -------

        """

        if self.coefficient_global is None:
            consumption, certificate, consumption_3uses = self.consumption_heating(climate=climate, full_output=True)
            s = self.stock.groupby(consumption.index.names).sum()
            if self.path_ini is not None:
                df = concat((consumption_3uses, s), axis=1, keys=['Consumption', 'Stock'])
                df = self.add_certificate(df).reset_index('Performance')
                make_hist(df, 'Consumption', 'Performance', 'Housing (Million)',
                          format_y=lambda y, _: '{:.0f}'.format(y / 10 ** 6),
                          save=os.path.join(self.path_ini, 'consumption_primary_ini.png'),
                          palette=self._resources_data['colors'],
                          kde=True)

                df = concat((consumption, s), axis=1, keys=['Consumption', 'Stock'])
                df = self.add_certificate(df).reset_index('Performance')
                make_hist(df, 'Consumption', 'Performance', 'Housing (Million)',
                          format_y=lambda y, _: '{:.0f}'.format(y / 10 ** 6),
                          save=os.path.join(self.path_ini, 'consumption_final_ini.png'),
                          palette=self._resources_data['colors'], kde=True)

            consumption = reindex_mi(consumption, self.stock.index) * self.surface

            _consumption_actual, heating_intensity, budget_share = self.consumption_actual(prices,
                                                                                           consumption=consumption,
                                                                                           full_output=True)
            # Test to calibrate health cost based on heating intensity
            if False:
                # calibration health_cost on heating intensity
                total_health_cost = self.health_cost(health_cost_dpe, health_cost_income, prices,
                                                     method_health_cost='epc')
                def find_threshold(x, df, total_health_cost):
                    d = df.loc[df['Heating intensity'] <= float(x), 'Stock']
                    cost = (d * health_cost_income).sum() / 1e9
                    return cost - total_health_cost

                df = concat((heating_intensity, self.stock), axis=1, keys=['Heating intensity', 'Stock'])
                root = fsolve(find_threshold, 0.5, args=(df, total_health_cost), xtol=1e-3)
                self.hi_threshold = root[0]

                # heating intensity threshold by income, performance group

                try:
                    df = pd.concat((heating_intensity, self.stock), axis=1, keys=['Heating intensity', 'Stock'])
                    df = df.reset_index('Income tenant')
                    make_hist(df, 'Heating intensity', 'Income tenant', 'Housing (Million)',
                              format_y=lambda y, _: '{:.1f}'.format(y / 10 ** 6),
                              save=os.path.join(self.path_ini, 'heating_intensity_ini.png'),
                              palette=self._resources_data['colors'], kde=True)

                    df = pd.concat((budget_share, self.stock), axis=1, keys=['Budget share', 'Stock'])
                    df = df.reset_index('Income tenant')
                    make_hist(df, 'Budget share', 'Income tenant', 'Housing (Million)',
                              format_y=lambda y, _: '{:.0f}'.format(y / 10 ** 6),
                              save=os.path.join(self.path_ini, 'budget_share_ini.png'),
                              palette=self._resources_data['colors'], kde=True)
                except:
                    pass

            _consumption_actual *= self.stock

            consumption_before_calibration = _consumption_actual.copy()

            consumption_ini = consumption_ini.groupby('Energy').sum()

            # 0. Test before any calibration
            consumption_energy = self.add_energy(_consumption_actual)
            consumption_energy = consumption_energy.groupby(consumption_ini.index.names).sum()
            consumption_energy = consumption_energy[consumption_energy > 0]
            test = concat((consumption_ini, consumption_energy / 1e9), axis=1, keys=['Observed', 'Simulated'])

            # 1. consumption total
            consumption_ini = consumption_ini[consumption_ini > 0]
            coefficient_global = consumption_ini.sum() * 10 ** 9 / _consumption_actual.sum()
            self.coefficient_global = coefficient_global
            _consumption_actual *= self.coefficient_global

            heating_intensity = self.to_heating_intensity(_consumption_actual.index, prices)
            self.heating_intensity_max = heating_intensity.max()

            # 2. coefficient among energies
            consumption_energy = self.add_energy(_consumption_actual)
            consumption_energy = consumption_energy.groupby(consumption_ini.index.names).sum()
            consumption_energy = consumption_energy[consumption_energy > 0]

            test = concat((consumption_ini, consumption_energy / 1e9), axis=1, keys=['Observed', 'Simulated'])
            coefficient = consumption_ini * 10 ** 9 / consumption_energy

            # coefficient[abs(1 - coefficient) < 0.05] = 1
            coefficient[coefficient.index.get_level_values('Energy') == 'Wood fuel'] = 1
            coefficient[coefficient.index.get_level_values('Energy') == 'Heating'] = 1

            # test
            _consumption_energy = coefficient * consumption_energy
            _consumption_energy['Wood fuel'] += ((1 - coefficient) * consumption_energy).sum()
            test = concat((consumption_ini, _consumption_energy / 10 ** 9), axis=1, keys=['Observed', 'Simulated'])

            # 3. coefficient energy is the share of the consumption used for other uses
            
            idx_heater = _consumption_actual.index.get_level_values('Heating system').unique()

            energy = idx_heater.str.split('-').str[0].rename('Energy')
            self.coefficient_backup = coefficient.reindex(energy).set_axis(idx_heater)

            self.coefficient_backup['Electricity-Heat pump air'] = 1
            self.coefficient_backup['Electricity-Heat pump water'] = 1
            self.coefficient_backup['Heating-District heating'] = 1

            temp = self.apply_calibration(_consumption_actual)
            validation = dict()
            # stock initial
            temp = concat((self.stock, self.energy), axis=1).groupby(
                ['Housing type', 'Energy']).sum().iloc[:, 0] / 10 ** 3
            temp.index = temp.index.map(lambda x: 'Stock {} {} (Thousands)'.format(x[0], x[1]))
            validation.update(temp)
            temp = self.stock.groupby('Housing type').sum() / 10 ** 3
            temp.index = temp.index.map(lambda x: 'Stock {} (Thousands)'.format(x))
            validation.update(temp)
            validation.update({'Stock (Thousands)': self.stock.sum() / 10 ** 3})

            # surface initial
            temp = concat((self.stock * self.surface, self.energy), axis=1).groupby(
                ['Housing type', 'Energy']).sum().iloc[:, 0] / 10 ** 6
            temp.index = temp.index.map(lambda x: 'Surface {} {} (Million m2)'.format(x[0], x[1]))
            validation.update(temp)
            temp = (self.stock * self.surface).groupby('Housing type').sum() / 10 ** 6
            temp.index = temp.index.map(lambda x: 'Surface {} (Million m2)'.format(x))
            validation.update(temp)
            validation.update({'Surface (Million m2)': (self.stock * self.surface).sum() / 10 ** 6})

            # heating consumption initial
            temp = concat((_consumption_actual, self.energy), axis=1).groupby(
                ['Housing type', 'Energy']).sum().iloc[:, 0] / 10 ** 9
            temp.index = temp.index.map(lambda x: 'Consumption {} {} (TWh)'.format(x[0], x[1]))
            validation.update(temp)
            temp = _consumption_actual.groupby('Housing type').sum() / 10 ** 9
            temp.index = temp.index.map(lambda x: 'Consumption {} (TWh)'.format(x))
            validation.update(temp)
            validation.update({'Consumption (TWh)': _consumption_actual.sum() / 10 ** 9})

            validation.update({'Coefficient calibration global (%)': self.coefficient_global})

            temp = self.coefficient_backup.copy()
            temp.index = temp.index.map(lambda x: 'Coefficient calibration {} (%)'.format(x))
            validation.update(temp)

            temp = consumption_energy / 10 ** 9
            temp.index = temp.index.map(lambda x: 'Consumption {} (TWh)'.format(x))
            validation.update(temp)

            validation = Series(validation)
            if self._resources_data['data_calibration'] is not None:
                validation = concat((validation, self._resources_data['data_calibration']), keys=['Calcul', 'Data'],
                                    axis=1)
                validation['Error'] = (validation['Calcul'] - validation['Data']) / validation['Data']

            if self.path_calibration is not None:
                validation.round(2).to_csv(os.path.join(self.path_calibration, 'validation_stock.csv'))

    @staticmethod
    def energy_bill(prices, consumption, level_heater='Heating system', bill_rebate=0):
        """Calculate energy bill by dwelling for each stock segment (€/dwelling.a).

        Parameters
        ----------
        prices: Series or DataFrame or float
            Energy prices for year (€/kWh)
        consumption: Series
            Energy consumption by dwelling (kWh/dwelling.a)
        level_heater
            Heating system level to calculate the bill. Enable to calculate energy bill before or after change of
            heating system.
        bill_rebate: float or Series, default 0
            Bill rebate (€/dwelling.a)

        Returns
        -------
        Series
            Energy bill by dwelling for each stock segment (€/dwelling)
        """

        index = consumption.index

        heating_system = Series(index.get_level_values(level_heater), index=index)
        energy = heating_system.str.split('-').str[0].rename('Energy')

        prices = prices.rename('Energy').reindex(energy)
        prices.index = index

        bill_rebate = reindex_mi(bill_rebate, index).fillna(0)

        if isinstance(consumption, Series):
            # * reindex_mi(self._surface, index)
            return reindex_mi(consumption, index) * prices - bill_rebate
        else:
            # * reindex_mi(self._surface, index)
            return (reindex_mi(consumption, index).T * prices - bill_rebate).T

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
        temp_optimal = Series(temp_optimal)

        temp = concat((consumption_actual, consumption_sd, temp_optimal), axis=1,
                      keys=['actual', 'conventional', 'temp optimal'])

        return temp_optimal

    def store_consumption(self, prices, carbon_content, bill_rebate=0):
        """Store energy consumption.


        Useful to calculate energy saving and rebound effect.

        Parameters
        ----------
        prices
        carbon_content
        bill_rebate
        """
        output = dict()
        temp = self.consumption_agg(freq='year', standard=True, existing=True, agg='energy')
        temp = temp.reindex(prices.index).fillna(0)
        output.update({'Consumption standard (TWh)': temp.sum()})
        temp.index = temp.index.map(lambda x: 'Consumption standard {} (TWh)'.format(x))
        output.update(temp)

        temp = self.consumption_agg(prices=prices, freq='year', standard=False, climate=None, smooth=False,
                                    existing=True, agg='energy', bill_rebate=bill_rebate)
        temp = temp.reindex(prices.index).fillna(0)
        output.update({'Consumption (TWh)': temp.sum()})
        emission = (temp * carbon_content).sum() / 10 ** 3
        temp.index = temp.index.map(lambda x: 'Consumption {} (TWh)'.format(x))
        output.update(temp)

        output.update({'Emission (MtCO2)': emission})

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
        Initial energy consumption by dwelling type.
    preferences: dict
        Preferences parameters.
    performance_insulation: dict
        Performance insulation parameters.
    path: str, optional
        Path to save results.
    year: int, default: 2018
        Year of the stock.
    demolition_rate: float, default 0.0
        Demolition rate.
    endogenous: bool, default True
        Enable to calculate endogenous renovation.
    number_exogenous: float or int, default 300000
        Number of exogenous renovation.
    expected_utility: {'market_share', 'max}
    logger: default
    calib_scale: bool, default True
    full_output: None
    quintiles: bool or None, default None
    financing_cost: bool, default True
    rational_behavior_insulation: bool, default None
    rational_behavior_heater: bool, default None
    resources_data: dict, default None
    detailed_output: bool, default True
    figures: dict, default None



    Attributes
    ----------
    """

    def __init__(self, stock, surface, ratio_surface, efficiency, income, preferences,
                 performance_insulation_renovation, lifetime_heater=None, path=None, year=2018,
                 endogenous=True, exogenous=None, expected_utility=None,
                 logger=None, calib_scale=True, quintiles=None,
                 rational_behavior_insulation=None, rational_behavior_heater=None,
                 resources_data=None, detailed_output=True, figures=None,
                 method_health_cost=None, residual_rate=0, constraint_heat_pumps=True,
                 variable_size_heater=True, temp_sink=None, social_discount_rate=0.032,
                 lifetime_insulation=30, vat_heater=VAT, no_friction=None, belief_engineering_calculation=None
                 ):
        super().__init__(stock, surface, ratio_surface, efficiency, income, path=path, year=year,
                         resources_data=resources_data, detailed_output=detailed_output, figures=figures,
                         residual_rate=residual_rate, temp_sink=temp_sink)

        self._distortion_store = {}
        if logger is None:
            logger = logging.getLogger()
        self.logger = logger

        self.memory = {}
        self.quintiles = quintiles
        if self.quintiles:
            self._resources_data = deciles2quintiles_dict(self._resources_data)

        self._flow_obligation = {}
        self.policies = []

        self._target_demolition = ['E', 'F', 'G']

        self.vat_heater = vat_heater

        choice_insulation = {'Wall': [False, True], 'Floor': [False, True], 'Roof': [False, True],
                             'Windows': [False, True]}
        names = list(choice_insulation.keys())
        choice_insulation = list(product(*[i for i in choice_insulation.values()]))
        choice_insulation.remove((False, False, False, False))
        choice_insulation = MultiIndex.from_tuples(choice_insulation, names=names)
        self._choice_insulation = choice_insulation
        self._performance_insulation_renovation = performance_insulation_renovation
        self.surface_insulation = self._ratio_surface.copy()

        self._endogenous, self.param_exogenous = endogenous, exogenous

        # method to calculate expected utility
        if expected_utility is not None:
            self._expected_utility = expected_utility
        else:
            self._expected_utility = 'market_share'

        self.social_discount_rate = social_discount_rate
        self.lifetime_insulation = lifetime_insulation

        self.preferences_heater = deepcopy(preferences['heater'])
        discount_factor = (1 - (1 + self.preferences_heater['present_discount_rate']) ** -self.preferences_heater[
            'lifetime']) / self.preferences_heater['present_discount_rate']
        self.preferences_heater['discount_factor'] = discount_factor.copy()
        self.preferences_heater['bill_saved'] = abs(self.preferences_heater['cost']) * discount_factor
        discount_factor_social = (1 - (1 + self.social_discount_rate) ** -self.preferences_heater[
            'lifetime']) / self.social_discount_rate
        self.preferences_heater['discount_factor_perfect'] = discount_factor_social
        self.preferences_heater['bill_saved_perfect'] = abs(self.preferences_heater['cost']) * discount_factor_social

        self.preferences_insulation = deepcopy(preferences['insulation'])
        discount_factor = (1 - (1 + self.preferences_insulation['present_discount_rate']) ** -self.preferences_insulation[
            'lifetime']) / self.preferences_insulation['present_discount_rate']
        self.preferences_insulation['discount_factor'] = discount_factor.copy()
        self.preferences_insulation['bill_saved'] = abs(self.preferences_insulation['cost']) * discount_factor
        discount_factor_social = (1 - (1 + self.social_discount_rate) ** -self.preferences_insulation[
            'lifetime']) / self.social_discount_rate
        self.preferences_insulation['discount_factor_perfect'] = discount_factor_social
        self.preferences_insulation['bill_saved_perfect'] = abs(self.preferences_insulation['cost']) * discount_factor_social

        self._calib_scale = calib_scale
        self.constant_insulation_extensive, self.constant_insulation_intensive, self.constant_insulation = None, None, None
        self.constant_heater = None
        self.scale_insulation, self.scale_heater = 1.0, 1.0
        self.hidden_cost, self.hidden_cost_insulation, self.hidden_cost_renovation = None, None, None
        self.landlord_dilemma, self.multifamily_friction = None, None
        self.hidden_cost_heater = None
        self.utility_hidden_cost = None

        self.number_firms_insulation, self.number_firms_heater = None, None

        self._list_condition_subsidies = []
        self._stock_ref = self.stock_mobile.copy()
        self._replaced_by = None
        self._only_heater = None
        self._heater_store = {}
        self._renovation_store = {}
        self._condition_store = None
        self._heating_intensity_avg = None

        if belief_engineering_calculation is None:
            self._belief_engineering_calculation = False
        else:
            self._belief_engineering_calculation = belief_engineering_calculation

        if no_friction is None:
            self.no_friction = False
        else:
            self.no_friction = no_friction

        self._variable_size_heater = variable_size_heater
        self._constraint_heat_pumps = constraint_heat_pumps

        self._markup_insulation_store, self._markup_heater_store = 1, 1

        self._annuities_store = {'investment_owner': Series(dtype=float),
                                 'rent_tenant': Series(dtype=float),
                                 'stock_annuities': Series(dtype=float),
                                 }

        self.consumption_saving_heater = None
        self.consumption_saving_insulation = None
        self.cost_rebound = None
        self.rebound = None

        self.rational_behavior_insulation = rational_behavior_insulation
        self.rational_hidden_cost = None
        self.rational_behavior_heater = rational_behavior_heater

        self.cost_curve_heater = None
        self.store_over_years = {year: {}}
        self.expenditure_store = {}
        self.taxes_revenues = {}
        self.bill_rebate = {}
        self._balance_state_ini = None

        # self.lifetime_heater = lifetime_heater
        temp = self.stock.groupby('Heating system').sum()
        heater_vintage = dict()
        for i in lifetime_heater.index:
            if i not in temp.index:
                temp.loc[i] = 0
            heater_vintage.update({i: Series([temp.loc[i] / lifetime_heater.loc[i]] * lifetime_heater.loc[i],
                                  index=range(1, lifetime_heater.loc[i] + 1))})
        self.heater_vintage = DataFrame(heater_vintage).T.rename_axis(index='Heating system', columns='Year')
        self.lifetime_heater = lifetime_heater

        # 'epc', 'heating_intensity'
        if method_health_cost is None:
            method_health_cost = 'epc'
        self.method_health_cost = method_health_cost

    @property
    def year(self):
        return self._year

    @year.setter
    def year(self, year):
        self._year = year
        self._stock_ref = self.stock_mobile.copy()
        self._surface = self._surface_yrs.loc[:, year]
        self.consumption_before_retrofit = None
        self.intensity_before_retrofit = None

        self._list_condition_subsidies = []
        self._condition_store = None
        self.income = self._income_yrs.loc[:, year]

        if isinstance(self._efficiency_yrs, DataFrame):
            self._efficiency = self._efficiency_yrs.loc[:, year]
        if self._temp_sink_yrs is not None:
            self._temp_sink = self._temp_sink_yrs.loc[year]

        self._replaced_by = None
        self._only_heater = None
        self._flow_obligation = {}

        ini = {
            'subsidies_details': {},
            'subsidies_count': {},
            'subsidies_average': {},
            'cost_average': {},
            'replacement_eligible': {}
        }

        for k, item in ini.items():
            self._heater_store[k] = item

        self.store_over_years.update({self.year: {}})

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

        df = df.fillna(0)

        # to check consistency
        df_sum = None
        if isinstance(df, Series):
            df_sum = df.sum()
        elif isinstance(df, DataFrame):
            df_sum = df.sum().sum()

        share = (ref.unstack(level).T / ref.unstack(level).sum(axis=1)).T
        temp = concat([df] * share.shape[1], keys=share.columns, names=share.columns.names, axis=1)
        share = reindex_mi(share, temp.columns, axis=1)
        share = reindex_mi(share, temp.index)
        df = (share * temp).stack(level).dropna()

        try:
            if isinstance(df, Series):
                assert_almost_equal(df.sum(), df_sum, decimal=0)
            elif isinstance(df, DataFrame):
                assert_almost_equal(df.sum().sum(), df_sum, decimal=0)
        except AssertionError:
            pass

        return df

    def add_attribute(self, df, levels):
        if isinstance(levels, str):
            levels = [levels]
        for lvl in levels:
            df = concat([df] * len(self._resources_data['index'][lvl]), keys=self._resources_data['index'][lvl],
                        names=[lvl],
                        axis=1).stack(lvl).squeeze()
        return df

    @staticmethod
    def sum_by_insulation_type(df):
        rslt = {}
        for i in df.columns.names:
            rslt.update({'{} renovated'.format(i): df.xs(True, level=i, drop_level=False, axis=1).sum(axis=1)})
        rslt = DataFrame(rslt)
        return rslt

    @staticmethod
    def to_cumac(consumption_saved, discount=0.035, duration=30):
        discount_factor = (1 - (1 + discount) ** -duration) / discount
        consumption_saved_cumac = (consumption_saved * discount_factor) / 1000
        return consumption_saved_cumac

    def frame_to_flow(self, replaced_by, default_quality=None):
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

        performance_insulation = self._performance_insulation_renovation.copy()
        if default_quality is not None:
            performance_insulation = {k: i * (1 + default_quality) for k, i in performance_insulation.items()}

        for component in ['Wall', 'Floor', 'Roof', 'Windows']:
            replaced_by[component] = replaced_by['{} before'.format(component)]
            replaced_by.loc[replaced_by['{} after'.format(component)], component] = performance_insulation[component]

        replaced_by.drop(
            ['Wall before', 'Wall after', 'Roof before', 'Roof after', 'Floor before', 'Floor after', 'Windows before',
             'Windows after'], axis=1, inplace=True)

        replaced_by = replaced_by.set_index(self.stock.index.names).loc[:, 'Data']
        replaced_by = replaced_by.groupby(replaced_by.index.names).sum()

        assert replaced_by.sum().round(0) == replaced_by_sum.round(0), 'Sum problem'

        return replaced_by

    def weighted_average(self, df, weights, levels=None, technology='insulation'):

        if levels is None:
            levels = ['Occupancy status', 'Housing type', 'Performance', 'Energy', 'Income owner']

        if 'Energy' in levels:
            df = self.add_energy(df)
            weights = self.add_energy(weights)
        if 'Performance' in levels:
            df = self.add_certificate(df)
            weights = self.add_certificate(weights)

        weights = weights.reorder_levels(df.index.names)

        if isinstance(df, DataFrame):
            df_grouped = (df.T * weights).T
            df_grouped = (df_grouped.groupby(levels).sum().T / weights.groupby(levels).sum()).T
            df_grouped.dropna(how='all', inplace=True)
            if technology == 'insulation':
                names = df_grouped.columns.names
                df_grouped.columns = df_grouped.columns.map(lambda row: ", ".join([name for name, value in zip(names, row) if value]))
            df_grouped = df_grouped.rename_axis('Technology', axis=1)
            df_grouped = df_grouped.stack('Technology').squeeze()

            return df_grouped
        else:
            raise NotImplementedError

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

    @staticmethod
    def select_deep_renovation(certificate_after):
        """Select deep renovation.

        Parameters
        ----------
        certificate_after: DataFrame
            Certificate after renovation.

        Returns
        -------
        DataFrame
        """
        condition = certificate_after.isin(['A', 'B'])
        index = condition.index[(~condition).all(axis=1)]
        condition.drop(index, inplace=True)
        condition = concat((condition, certificate_after.loc[index, :].isin(['C'])), axis=0)
        index = condition.index[(~condition).all(axis=1)]
        condition.drop(index, inplace=True)
        condition = concat((condition, certificate_after.loc[index, :].isin(['D'])), axis=0)
        return condition

    def prepare_consumption(self, choice_insulation=None, performance_insulation=None, index=None, method_epc='5uses',
                            level_heater='Heating system', full_output=True, store=True, climate=None, default_quality=None):
        """Standard energy consumption and energy performance certificate for each renovation works option.

        Standard energy consumption only depends on building characteristics.

        Parameters
        ----------
        choice_insulation: MultiIndex
            Insulation choice.
        performance_insulation: MultiIndex
            Insulation performance.
        index: MultiIndex
            Index to calculate consumption.
        method_epc: {'3uses', '5uses'}
            Method to calculate energy performance certificate.
        level_heater: str
            Heating system level to calculate the consumption.
        full_output: bool, default True
            Enable to return full output.
        store: bool, default True
            Enable to store results.
        climate: int
            Climate year.

        Returns
        -------
        DataFrame
            Final consumption standard.
        DataFrame
            Primary consumption standard 3 uses.
        DataFrame
            Energy performance certificate.
        """

        # default option: store = True means that the result is stored in the object
        if climate is not None:
            store = False

        if method_epc == '3uses':
            store = False

        if index is None:
            index = self.stock.index

        if not isinstance(choice_insulation, MultiIndex):
            choice_insulation = self._choice_insulation

        if not isinstance(performance_insulation, MultiIndex):
            performance_insulation = self._performance_insulation_renovation

        if default_quality is not None:
            performance_insulation = {k: i * (1 + default_quality) for k, i in performance_insulation.items()}

        # only selecting useful levels as theoretical energy consumption only depends on building characteristics
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
            s = concat([Series(index=idx, dtype=float)] * len(choice_insulation), axis=1).set_axis(choice_insulation,
                                                                                                   axis=1)
            # choice_insulation = choice_insulation.drop(no_insulation) # only for
            s.index.rename(
                {'Wall': 'Wall before', 'Floor': 'Floor before', 'Roof': 'Roof before', 'Windows': 'Windows before'},
                inplace=True)
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
            if climate is not None or method_epc == '3uses':
                consumption, certificate, consumption_3uses = self.consumption_heating(index=index, climate=climate,
                                                                                       level_heater='Heating system',
                                                                                       method=method_epc, full_output=True)
            else:
                consumption, consumption_3uses, certificate = self.consumption_heating_store(index,
                                                                                             level_heater='Heating system')


            rslt = dict()
            t = {'consumption': consumption}
            if full_output is True:
                t.update({'consumption_3uses': consumption_3uses, 'certificate': certificate})
            for key, temp in t.items():
                temp = reindex_mi(temp, index).droplevel(['Wall', 'Floor', 'Roof', 'Windows']).unstack(
                    ['{} bool'.format(i) for i in ['Wall', 'Floor', 'Roof', 'Windows']])
                temp.index.rename({'Wall before': 'Wall', 'Floor before': 'Floor', 'Roof before': 'Roof',
                                   'Windows before': 'Windows'},
                                  inplace=True)
                temp.columns.rename(
                    {'Wall bool': 'Wall', 'Floor bool': 'Floor', 'Roof bool': 'Roof', 'Windows bool': 'Windows'},
                    inplace=True)
                rslt[key] = temp

            if store is True:
                if self._consumption_store['consumption_renovation'].empty:
                    self._consumption_store['consumption_renovation'] = rslt['consumption']
                    self._consumption_store['consumption_3uses_renovation'] = rslt['consumption_3uses']
                    self._consumption_store['certificate_renovation'] = rslt['certificate']
                else:
                    self._consumption_store['consumption_renovation'] = concat(
                        (self._consumption_store['consumption_renovation'], rslt['consumption']))
                    self._consumption_store['consumption_3uses_renovation'] = concat(
                        (self._consumption_store['consumption_3uses_renovation'], rslt['consumption_3uses']))
                    self._consumption_store['certificate_renovation'] = concat(
                        (self._consumption_store['certificate_renovation'], rslt['certificate']))
            else:
                consumption = rslt['consumption'].loc[_index]
                consumption.index.rename({'Heating system': level_heater}, inplace=True)
                if full_output is False:
                    return consumption
                else:
                    consumption_3uses = rslt['consumption_3uses'].loc[_index]
                    consumption_3uses.index.rename({'Heating system': level_heater}, inplace=True)
                    certificate = rslt['certificate'].loc[_index]
                    certificate.index.rename({'Heating system': level_heater}, inplace=True)
                    return consumption, consumption_3uses, certificate

        # formatting using results stored in the object
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

    def calculate_financing(self, cost, subsidies, financing_cost, policies=None):
        """Apply financing cost.

        Parameters
        ----------
        cost
        subsidies
        financing_cost
        policies: list

        Returns
        -------
        cost_total
        cost_financing
        amount_debt
        amount_saving
        discount
        subsidies
        """

        cost_total, cost_financing, amount_debt, amount_saving, discount = cost, None, None, None, None
        to_pay = cost - subsidies

        financing_options = {'debt': {'price': financing_cost['interest_rate'].loc[self.year],
                                      'max': None,
                                      'duration': financing_cost['duration'],
                                      'type': 'debt'
                                      },
                             'saving': {'price': financing_cost['saving_rate'].loc[self.year],
                                        'max': financing_cost['upfront_max'],
                                        'duration': financing_cost['duration'],
                                        'type': 'saving'
                                        },
                             }

        # only work with one zero-interest-loan
        if policies:
            for policy in policies:
                if isinstance(policy.value, dict):
                    value = policy.value[self.year]
                else:
                    value = policy.value

                if value is None:
                    # Meaning we are assessing the zero_interest_loan
                    value = financing_cost['interest_rate'].loc[self.year] - 1e-4

                if isinstance(policy.cost_max, dict):
                    cost_max = policy.cost_max[self.year]
                else:
                    cost_max = policy.cost_max

                # Only when policy.target otherwise cost_max = 0
                if policy.target is not None:
                    cost_max = cost_max * policy.target

                financing_options.update({policy.name: {'price': value,
                                                        'max': cost_max,
                                                        'duration': policy.duration,
                                                        'type': 'debt'}})

        # sort financing_options by price
        financing_options = {k: v for k, v in sorted(financing_options.items(), key=lambda item: item[1]['price'])}

        if financing_cost is not None:

            remaining = to_pay.copy()
            rslt = {}
            cost_financing, debt_reimbursement = DataFrame(0, index=to_pay.index, columns=to_pay.columns), DataFrame(0, index=to_pay.index, columns=to_pay.columns)
            for key, item in financing_options.items():
                if item['max'] is not None:
                    c_max = reindex_mi(item['max'], remaining.index)
                    if isinstance(c_max, Series):
                        c_max = concat([c_max] * remaining.shape[1], keys=to_pay.columns, axis=1)
                    amount = remaining.where(remaining < c_max, c_max)
                else:
                    amount = remaining

                if item['type'] == 'debt':
                    if item['price'] == 0:
                        reimbursement = amount / item['duration']
                    else:
                        reimbursement = calculate_annuities(amount, lifetime=item['duration'], discount_rate=item['price'])
                    debt_reimbursement += reimbursement

                # could be better calculated
                c = (calculate_annuities(amount, lifetime=item['duration'], discount_rate=item['price']) * item['duration'] - amount)
                cost_financing += c
                rslt.update({key: amount.copy()})
                remaining -= amount


            assert remaining.sum().sum().round(0) == 0, 'Not all cost have been financed'
            assert allclose(sum(rslt.values()), to_pay), 'Not all cost have been financed'

            share = {k: i / to_pay for k, i in rslt.items()}
            share = {k: i.fillna(0) for k, i in share.items()}

            # assert allclose(sum(share.values()).round(2), DataFrame(1, index=to_pay.index, columns=to_pay.columns)), 'Share problem'
            discount = sum([share[k] * financing_options[k]['price'] for k in rslt.keys()])

            amount_debt = sum([rslt[k] for k in financing_options.keys() if financing_options[k]['type'] == 'debt'])
            amount_saving = sum([rslt[k] for k in financing_options.keys() if financing_options[k]['type'] == 'saving'])
            assert allclose(amount_debt + amount_saving, to_pay), 'Not all cost have been financed'

            cost_total = cost + cost_financing

            subsidies = dict()
            if policies:
                for policy in policies:
                    subsidies.update({policy.name: rslt[policy.name] * policy.public_cost * financing_options[policy.name]['duration']})

        return cost_total, cost_financing, amount_debt, amount_saving, discount, subsidies

    def credit_constraint(self, amount_debt, financing_cost, bill_saved=None):

        condition = DataFrame(True, index=amount_debt.index, columns=amount_debt.columns)

        sub = None
        if financing_cost['debt_income_ratio']:
            debt_reimbursement = calculate_annuities(amount_debt, lifetime=financing_cost['duration'],
                                                     discount_rate=financing_cost['interest_rate'].loc[self.year])
            debt_income_ratio = (debt_reimbursement.T / reindex_mi(self._income_owner, amount_debt.index)).T
            condition = condition & (debt_income_ratio <= financing_cost['debt_income_ratio'])

            factor = factor_annuities(lifetime=financing_cost['duration'],
                                      discount_rate=financing_cost['interest_rate'].loc[self.year])

            sub = (amount_debt.T - reindex_mi(self._income_owner * financing_cost['debt_income_ratio'] / factor, amount_debt.index)).T
            sub = sub.where(sub > 0, 0)

            if bill_saved is not None:
                condition = condition & (debt_reimbursement <= bill_saved)

        return condition

    def apply_subsidies_heater(self, index, policies_heater, cost_heater, consumption_saved, emission_saved,
                               bill_saved):
        """Calculate subsidies for each dwelling and each heating system.

        Parameters
        ----------
        index: Index or MultiIndex
        policies_heater: list
        cost_heater: DataFrame
        consumption_saved: DataFrame
        emission_saved: DataFrame

        Returns
        -------
        """

        subsidies_details = {}

        vat_base = self.vat_heater
        if isinstance(vat_base, Series):
            vat_base = vat_base.reindex(cost_heater.columns)

        p = [p for p in policies_heater if 'tax_status_quo' == p.policy]
        if p:
            value = self.preferences_heater['status_quo_bias'] / abs(self.preferences_heater['cost']) * 1000
            value = Series(value, index=Index(['Natural gas-Performance boiler'], name='Heating system final'))
            value = value.reindex(cost_heater.columns).fillna(0)
            t = ((cost_heater + value) / cost_heater).mean() - 1
            vat_base += t.reindex(vat_base.index).fillna(0)

        p = [p for p in policies_heater if 'reduced_vat' == p.policy]
        if p:
            vat = p[0].value
            if isinstance(vat, dict):
                vat = vat[self.year]
            sub = cost_heater * (vat_base - vat)
            subsidies_details.update({'reduced_vat': sub.copy()})
        else:
            vat = vat_base

        vat_heater = cost_heater * vat
        cost_heater += vat_heater

        policies_incentive = ['subsidy_ad_valorem', 'subsidy_target', 'subsidy_proportional', 'subsidies_cap',
                              'subsidy_present_bias']
        self.policies += [i.name for i in policies_heater if i.policy in policies_incentive and i.name not in self.policies]

        cost_heater.sort_index(inplace=True)

        for policy in [p for p in policies_heater if p.policy in ['subsidy_ad_valorem', 'subsidy_target', 'subsidy_proportional',
                                                                  'subsidy_present_bias']]:

            if isinstance(policy.value, dict):
                value = policy.value[self.year]
            else:
                value = policy.value

            if policy.policy == 'subsidy_target':
                value = value.stack()
                value = value[value.index.get_level_values('Heating system final').isin(cost_heater.columns)]
                value = reindex_mi(value, cost_heater.stack().index).unstack('Heating system final').fillna(0)

            elif policy.policy == 'subsidy_present_bias':
                value = bill_saved.copy()
                value[value < 0] = 0
                value = (reindex_mi((self.preferences_heater['discount_factor_perfect'] -
                                     self.preferences_heater['discount_factor']), value.index) * value.T).T

            elif policy.policy == 'subsidy_ad_valorem':
                if isinstance(policy.value, dict):
                    value = policy.value[self.year]
                else:
                    value = policy.value

                if isinstance(value, (float, int)):
                    value = value * cost_heater

                elif isinstance(value, DataFrame):
                    value = value.stack()
                    value = value[value.index.get_level_values('Heating system final').isin(cost_heater.columns)]
                    value = (cost_heater.stack() * value).unstack('Heating system final').fillna(0)

                elif isinstance(value, Series):
                    raise NotImplemented('Error by should be index or columns')

            elif policy.policy == 'subsidy_proportional':
                # reindex_mi(cost_insulation, index) / consumption_saved_cumac
                if policy.proportional == 'tCO2_cumac':
                    emission_saved_cumac = self.to_cumac(emission_saved, discount=self.social_discount_rate,
                                                         duration=self.lifetime_heater) / 10**3
                    emission_saved_cumac[emission_saved_cumac < 0] = 0
                    value = value * emission_saved_cumac
                    value = value.fillna(0).loc[:, emission_saved_cumac.columns]
                elif policy.proportional == 'MWh_cumac':
                    consumption_saved_cumac = self.to_cumac(consumption_saved)
                    consumption_saved_cumac[consumption_saved_cumac < 0] = 0
                    value = value * consumption_saved_cumac
                else:
                    raise NotImplemented

            else:
                raise NotImplemented

            assert (value >= 0).all().all(), 'Subsidies need to be negative'

            # cap for all policies
            value.fillna(0, inplace=True)
            value = value.loc[:, [i for i in cost_heater.columns if i in value.columns]]
            value = value.reorder_levels(cost_heater.index.names)
            value.sort_index(inplace=True)
            value = value.where(value <= cost_heater, cost_heater)

            if policy.cap:
                value = value.where(value <= policy.cap, policy.cap)

            subsidies_details[policy.name] = value.copy()

        subsidies_bonus = [p for p in policies_heater if p.policy == 'bonus']
        for policy in subsidies_bonus:
            if isinstance(policy.value, dict):
                value = policy.value[self.year]
            else:
                value = policy.value

            value = reindex_mi(value, index)
            value = value.reindex(cost_heater.columns, axis=1).fillna(0)

            if policy.name in subsidies_details.keys():
                subsidies_details[policy.name] = subsidies_details[policy.name] + value
            else:
                subsidies_details[policy.name] = value

        subsidies_total = [subsidies_details[k] for k in subsidies_details.keys() if
                           k not in ['reduced_vat', 'over_cap']]
        if subsidies_total:
            subsidies_total = sum(subsidies_total)
        else:
            subsidies_total = pd.DataFrame(0, index=index, columns=cost_heater.columns)

        # store eligible before cap
        eligible = {}
        for policy in subsidies_details:
            eligible.update({policy: (subsidies_details[policy] > 0).any(axis=1)})

        # overall cap for cumulated amount of subsidies
        subsidies_cap = [p for p in policies_heater if p.policy == 'subsidies_cap']
        if subsidies_cap:
            # only one subsidy cap
            subsidies_cap = subsidies_cap[0]
            policies_target = subsidies_cap.target
            subsidies_cap = reindex_mi(subsidies_cap.value, subsidies_total.index)
            cap = (reindex_mi(cost_heater, index).T * subsidies_cap).T
            over_cap = subsidies_total > cap
            subsidies_details['over_cap'] = (subsidies_total - cap)[over_cap].fillna(0)
            subsidies_details['over_cap'].sort_index(inplace=True)
            remaining = subsidies_details['over_cap'].copy()

            for policy_name in policies_target:
                if policy_name in subsidies_details.keys():
                    self.logger.debug('Capping amount for {}'.format(policy_name))
                    temp = remaining.where(remaining <= subsidies_details[policy_name], subsidies_details[policy_name])
                    subsidies_details[policy_name] -= temp
                    assert (subsidies_details[policy_name].values >= 0).all(), '{} got negative values'.format(
                        policy_name)
                    remaining -= temp
                    if allclose(remaining, 0, rtol=10 ** -1):
                        break

            assert allclose(remaining, 0, rtol=10 ** -1), 'Over cap'
            subsidies_total -= subsidies_details['over_cap']

        regulation = [p for p in policies_heater if p.policy == 'regulation']
        if 'status_quo_bias' in [p.name for p in regulation]:
            self.preferences_heater['status_quo_bias'] = 0
        if 'present_bias' in [p.name for p in regulation]:
            v = [p for p in regulation if p.name == 'present_bias'][0].value
            discount_factor = (1 - (1 + v) ** -self.preferences_heater['lifetime']) / v
            idx = self.preferences_heater['bill_saved'].index
            self.preferences_heater['bill_saved'] = Series(discount_factor * abs(self.preferences_heater['cost']),
                                                           index=idx)

        return cost_heater, vat_heater, subsidies_details, subsidies_total, eligible

    def calculate_distortion_heater(self, stock, cost_heater, emission_saved, carbon_value, bill_saved, cap=True):

        distortion_details = {}
        # 1. distortion present-bias
        value = bill_saved.copy()
        value[value < 0] = 0
        value = (reindex_mi((self.preferences_heater['discount_factor_perfect'] -
                             self.preferences_heater['discount_factor']), value.index) * value.T).T
        value.fillna(0, inplace=True)
        distortion_details.update({'bill_saved': value.copy()})

        # 2. distortion emission
        emission_saved_cumac = self.to_cumac(emission_saved, discount=self.social_discount_rate,
                                             duration=self.lifetime_heater) / 10 ** 3
        emission_saved_cumac[emission_saved_cumac < 0] = 0
        value = carbon_value * emission_saved_cumac
        value = value.fillna(0).loc[:, emission_saved.columns]
        distortion_details.update({'emission_saved': value.copy()})

        # 3. distortion status quo
        value = pd.DataFrame(0, index=cost_heater.index, columns=cost_heater.columns)
        for hs in value.columns:
            value.loc[
                value.index.get_level_values('Heating system') == hs, hs] = - self.preferences_heater['status_quo_bias'] / abs(self.preferences_heater['cost']) * 1000
        distortion_details.update({'status_quo': value.copy()})

        distortion_total = sum([i for k, i in distortion_details.items()])
        hs_to_drop = ['Heating-District heating', 'Oil fuel-Performance boiler']
        distortion_total.drop(hs_to_drop, inplace=True, axis=1)
        distortion_total = (distortion_total.T - distortion_total.min(axis=1)).T

        if cap is True:
            distortion_total = distortion_total.loc[:, [i for i in cost_heater.columns if i in distortion_total.columns]]
            c = cost_heater.loc[:, [i for i in cost_heater.columns if i in distortion_total.columns]]
            distortion_total = distortion_total.where(distortion_total <= c, c)

            distortion_total[distortion_total < 0] = 0

        levels = ['Occupancy status', 'Housing type', 'Performance', 'Energy', 'Income owner']
        distortion_grouped = self.weighted_average(distortion_total, stock.copy(), levels=levels, technology='heater')
        return distortion_grouped

    def endogenous_market_share_heater(self, index, _bill_saved, _subsidies_total, _cost_heater, calib_heater=None,
                                       _cost_financing=None, condition=None, flow_replace=None, drop_heater=None,
                                       _credit_constraint=None):

        def test_sensitivity_hp(_utility_cost, _utility_bill_saving, _utility_financing, _utility_inertia, _condition,
                                _flow_replace, _cost_heater):
            # assessing sensitivity to subsidies_heat_pump
            _utility = _utility_cost + _utility_bill_saving + _utility_financing + _utility_inertia
            _utility_constant = reindex_mi(self.constant_heater.reindex(_utility.columns, axis=1), _utility.index)
            _utility += _utility_constant

            c = _cost_heater.copy()
            c.loc[:, [i for i in _cost_heater.columns if i != 'Electricity-Heat pump water']] = 0

            dict_rslt = dict()

            rslt = dict()
            for sub in range(0, 110, 10):
                sub /= 100
                u_sub = c * sub * self.preferences_heater['subsidy'] / 1000
                u = _utility + u_sub

                if _condition is not None:
                    _condition = _condition.astype(float)
                    _condition = _condition.replace(0, float('nan'))
                    u *= _condition

                _market_share = (exp(u).T / exp(u).sum(axis=1)).T
                temp = (_flow_replace * _market_share.T).T
                rslt.update({sub: temp.loc[:, 'Electricity-Heat pump water'].sum()})
            dict_rslt.update({'With inertia': Series(rslt) / 10 ** 3})

            _utility -= _utility_inertia
            rslt = dict()
            for sub in range(0, 110, 10):
                sub /= 100
                u_sub = c * sub * self.preferences_heater['subsidy'] / 1000
                u = _utility + u_sub

                if _condition is not None:
                    _condition = _condition.astype(float)
                    _condition = _condition.replace(0, float('nan'))
                    u *= _condition

                _market_share = (exp(u).T / exp(u).sum(axis=1)).T
                temp = (_flow_replace * _market_share.T).T
                rslt.update({sub: temp.loc[:, 'Electricity-Heat pump water'].sum()})
            dict_rslt.update({'Without inertia': Series(rslt) / 10 ** 3})

            make_plots(dict_rslt, 'Heat pumps function of ad valorem subsidy (Thousand per year)',
                       save=os.path.join(self.path_calibration, 'sensi_hp.png'), integer=False, legend=True)

        def calibration_heater(_utility, _calib_heater, _utility_shock, _flow_replace):

            def calibration(x, _ms, _utility_ini, _flow, _idx, _ref, _u_shock, _target,
                            _option='price_elasticity'):

                if _option is not None:
                    _scale = abs(x[0])
                else:
                    _scale = 1

                cst = pd.Series(x[1:], index=_idx)
                cst = concat((cst, Series(0, index=_ref)))

                cst = cst.unstack('Heating system final')
                _u = _utility_ini * _scale + cst
                _market_share = (exp(_u).T / exp(_u).sum(axis=1)).T
                _agg = (_market_share.T * _flow).T.groupby(cst.index.names).sum()
                _market_share_agg = (_agg.T / _agg.sum(axis=1)).T
                _market_share_agg = _market_share_agg.stack()
                _market_share_agg = _market_share_agg[_market_share_agg > 0]

                _rslt = _market_share_agg - _ms
                _rslt.drop(_ref, inplace=True)
                _rslt = _rslt.loc[_idx].to_numpy()

                if _option == 'shock':
                    _temp = _u + _u_shock * _scale
                    # temp = _utility_ini + _u_shock
                    _market_share = (exp(_temp).T / exp(_temp).sum(axis=1)).T
                    _agg = (_market_share.T * _flow).T
                    heat_pump = _agg.loc[:, 'Electricity-Heat pump water'].sum() / _flow.sum()
                    _rslt = append(heat_pump - _target, _rslt)

                elif _option == 'price_elasticity':
                    heater_ref = 'Electricity-Heat pump water'
                    _price_elasticity = self.preferences_heater['cost'] * _scale * _cost_heater.loc[:, heater_ref].iloc[0] / 1000 * (1 - _market_share.loc[:, heater_ref])
                    _price_elasticity_average = (_flow * _price_elasticity).sum() / _flow.sum()
                    _rslt = append(_price_elasticity_average - _target, _rslt)

                else:
                    _rslt = append(0, _rslt)

                return _rslt

            # target
            _ms_heater = _calib_heater['ms_heater']
            option = _calib_heater['scale']['option']
            target = _calib_heater['scale']['target']

            # simplifying market-share
            simplify = True
            if simplify:
                flow = _flow_replace.groupby(_ms_heater.index.names).sum()
                ms = (flow * _ms_heater.T).T.groupby('Housing type').sum()
                ms = (ms.T / ms.sum(axis=1)).T
                ms = ms.stack()
                ms = ms[ms > 0]
            else:
                ms = _ms_heater.copy()

            ref = MultiIndex.from_tuples([('Multi-family', 'Electricity-Direct electric'), ('Single-family', 'Electricity-Direct electric')], names=['Housing type', 'Heating system final'])
            x0 = pd.Series(0, index=ms.index).drop(ref)
            idx = x0.index
            x0 = x0.copy().to_numpy()
            x0 = append(1, x0)

            root, info_dict, ier, mess = fsolve(calibration, x0, args=(ms, _utility.copy(), _flow_replace, idx, ref,
                                                                       _utility_shock, target, option), full_output=True)
            if ier == 1:
                self.logger.debug('Calibration investment decision heater worked')
            else:
                raise ValueError('Calibration investment decision heater did not work')

            scale = abs(root[0])
            constant = concat((Series(root[1:], index=idx), Series(0, index=ref))).unstack('Heating system final')
            ms = ms.unstack('Heating system final')

            self.constant_heater = constant.copy()
            self.apply_scale(scale, gest='heater')

            # no calibration
            temp = (exp(_utility).T / exp(_utility).sum(axis=1)).T
            agg = (temp.T * _flow_replace).T.groupby(ms.index.names).sum()
            ms_no_calibration = (agg.T / agg.sum(axis=1)).T

            # with shock
            temp_shock = (exp(_utility + _utility_shock).T / exp(_utility + _utility_shock).sum(axis=1)).T
            agg_shock = (temp_shock.T * _flow_replace).T.groupby(ms.index.names).sum()
            ms_no_calibration_agg_shock = (agg_shock.T / agg_shock.sum(axis=1)).T

            temp = (exp(_utility * scale + constant).T / exp(_utility * scale + constant).sum(axis=1)).T
            assert (temp.sum(axis=1).round(1) == 1.0).all(), 'Market-share issue'
            agg = (temp.T * _flow_replace).T.groupby(ms.index.names).sum()
            market_share_agg = (agg.T / agg.sum(axis=1)).T

            # with shock
            temp_shock = (exp((_utility + _utility_shock) * scale + constant).T / exp((_utility + _utility_shock) * scale + constant).sum(axis=1)).T
            assert (temp.sum(axis=1).round(1) == 1.0).all(), 'Market-share issue'
            agg_shock = (temp_shock.T * _flow_replace).T.groupby(ms.index.names).sum()
            market_share_agg_shock = (agg_shock.T / agg_shock.sum(axis=1)).T

            wtp = constant / self.preferences_heater['cost']
            hidden_benefits = constant / abs(self.preferences_heater['cost'])

            details = concat((constant.stack(), ms_no_calibration.stack(), market_share_agg.stack(), ms.stack(),
                              agg.stack() / 10 ** 3, wtp.stack(), hidden_benefits.stack()),
                             axis=1, keys=['constant', 'no_calibration', 'calcul', 'observed', 'thousand', 'wtp',
                                           'hidden_benefits']).round(decimals=3)

            if self.path_calibration is not None:
                details.to_csv(os.path.join(self.path_calibration, 'calibration_constant_heater.csv'))

                # export utility coefficient
                cost = Series(self.preferences_heater['cost'])
                cost = cost.reset_index().set_axis(['Attribute', 'Value'], axis=1)

                preference_benefits = self.preferences_heater['bill_saved']
                preference_benefits = preference_benefits.reset_index().set_axis(['Attribute', 'Value'], axis=1)

                status_quo = Series(self.preferences_heater['status_quo_bias'])
                status_quo = status_quo.reset_index().set_axis(['Attribute', 'Value'], axis=1)

                constant_heater = self.constant_heater.stack().copy()
                constant_heater.index = constant_heater.index.map(lambda x: '{} | {}'.format(x[0], x[1]))
                constant_heater = constant_heater.reset_index().set_axis(['Attribute', 'Value'], axis=1)

                temp = {'Cost': cost,
                        'Benefits': preference_benefits,
                        'Status quo': status_quo,
                        'Constant measure': constant_heater}
                temp = concat(temp, axis=0).droplevel(-1)
                temp.to_csv(os.path.join(self.path_calibration, 'coefficient_utility_heater.csv'))

            return constant

        _bill_saved, _subsidies_total, _cost_heater = _bill_saved.loc[index, :], _subsidies_total.loc[index, :], _cost_heater.loc[index, :]

        choice_heater = _cost_heater.columns
        if drop_heater:
            choice_heater = [i for i in choice_heater if i not in drop_heater]

        if condition is not None:
            condition = condition.loc[index, choice_heater]

        utility_bill_saving = (_bill_saved.T * reindex_mi(self.preferences_heater['bill_saved'],
                                                         _bill_saved.index)).T / 1000
        utility_bill_saving = utility_bill_saving.loc[:, choice_heater]

        utility_subsidies = _subsidies_total * self.preferences_heater['subsidy'] / 1000

        _cost_heater = _cost_heater.reindex(index).reindex(choice_heater, axis=1)
        pref_investment = reindex_mi(self.preferences_heater['cost'], index)
        utility_cost = (pref_investment * _cost_heater.T).T / 1000

        utility_inertia = DataFrame(0, index=utility_bill_saving.index, columns=utility_bill_saving.columns)
        for hs in choice_heater:
            utility_inertia.loc[
                utility_inertia.index.get_level_values('Heating system') == hs, hs] = self.preferences_heater['status_quo_bias']

        utility = utility_inertia + utility_cost + utility_bill_saving + utility_subsidies

        if _cost_financing is not None:
            _cost_financing = _cost_financing.loc[index, :]
            pref_financing = reindex_mi(self.preferences_heater['cost'], _cost_financing.index).rename(None)
            utility_financing = (_cost_financing.T * pref_financing).T / 1000
            utility += utility_financing

        if condition is not None:
            condition = condition.loc[index, :]
            condition = condition.astype(float)
            condition = condition.replace(0, float('nan'))
            utility *= condition
            utility.dropna(how='all', axis=1, inplace=True)

        if _credit_constraint is not None:
            _credit_constraint = _credit_constraint.loc[index, utility.columns]
            _credit_constraint = _credit_constraint.astype(float)
            _credit_constraint = _credit_constraint.replace(0, float('nan'))
            utility_constraint = utility * _credit_constraint

            # there are cases where hh cannot afford any investment, we therefore let them keep the same heating system
            temp = utility_constraint.loc[utility_constraint.isna().all(axis=1), :]
            t = DataFrame(index=temp.index, columns=temp.columns, dtype=float)
            for i in temp.index.get_level_values('Heating system').unique():
                idx = temp[temp.index.get_level_values('Heating system') == i].index
                t.loc[idx, i] = utility.loc[idx, i]

            # u = utility.copy()
            utility *= _credit_constraint
            utility.dropna(how='all', inplace=True)
            utility = concat((utility, t), axis=0)

        utility = utility.loc[index, :]
        if (self.constant_heater is None) and (calib_heater is not None):
            self.logger.info('Calibration market-share heating system')
            sub_shock = - utility_cost - utility_financing
            sub_shock.loc[:, [i for i in sub_shock.columns if i != 'Electricity-Heat pump water']] = 0
            utility_shock = - utility_subsidies + sub_shock
            utility_shock = utility_shock.loc[index, :]

            if condition is not None:
                utility_shock *= condition
                utility_shock.dropna(how='all', axis=1, inplace=True)

            # ms_heater.dropna(how='all', inplace=True)
            calib_heater['ms_heater'] = calib_heater['ms_heater'].loc[:, utility.columns]

            calibration_heater(utility, calib_heater, utility_shock, flow_replace)

            utility_cost *= self.scale_heater
            utility_bill_saving *= self.scale_heater
            utility_financing *= self.scale_heater
            utility_inertia *= self.scale_heater
            if self.path_calibration:
                test_sensitivity_hp(utility_cost, utility_bill_saving, utility_financing, utility_inertia, condition,
                                      flow_replace, _cost_heater)

            utility *= self.scale_heater

        utility_constant = reindex_mi(self.constant_heater.reindex(utility.columns, axis=1), utility.index)

        utility += utility_constant
        # x = (log(exp(utility).sum(axis=1)) - utility.T).T
        error_conditional = (log(exp(utility).sum(axis=1)) + (- utility + utility_constant).T).T

        market_share = (exp(utility).T / exp(utility).sum(axis=1)).T
        return market_share, error_conditional

    def exogenous_market_share_heater(self, index, choice_heater_idx):
        """Define exogenous market-share.

        Market-share is defined by _market_share_exogenous attribute.

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
                                  'Electricity-Direct electric': 'Electricity-Heat pump air',
                                  'Electricity-Heat pump air': 'Electricity-Heat pump air',
                                  'Electricity-Heat pump water': 'Electricity-Heat pump water',
                                  }

        market_share = Series(index=index, dtype=float).to_frame().dot(
            Series(index=choice_heater_idx, dtype=float).to_frame().T)

        for initial, final in market_share_exogenous.items():
            market_share.loc[market_share.index.get_level_values('Heating system') == initial, final] = 1

        temp = Series(0, index=index, dtype='float').to_frame().dot(
            Series(0, index=choice_heater_idx, dtype='float').to_frame().T)
        index_final = temp.stack().index
        _, _, certificate = self.consumption_heating_store(index_final, level_heater='Heating system final')
        certificate = reindex_mi(certificate.unstack('Heating system final'), index)
        certificate_before = self.consumption_heating_store(index)[2]
        certificate_before = reindex_mi(certificate_before, index)

        self._heater_store['epc_upgrade'] = - certificate.replace(EPC2INT).sub(
            certificate_before.replace(EPC2INT), axis=0)

        return market_share

    def store_information_heater(self, cost_heater, subsidies_total, bill_saved, subsidies_details, replacement, vat_heater,
                                 cost_financing, amount_debt, amount_saving, discount, consumption_saved,
                                 consumption_before, consumption_no_rebound, consumption_actual, epc_upgrade,
                                 flow_premature_replacement, subsidies_loan,
                                 eligible, hidden_benefits_heater):
        """Store information yearly heater replacement.

        Parameters
        ----------
        cost_heater: DataFrame
            Cost of each heating system (EUR).
        subsidies_total: DataFrame
            Total amount of eligible subsidies by dwelling and heating system (EUR).
        subsidies_details: dict
            Amount of eligible subsidies by dwelling and heating system (EUR).
        replacement: Series
            Dwelling updated with a new heating system.
        vat_heater: Series
            vat tax of each heating system (EUR).
        flow_premature_replacement: int
            Number of dwelling that replace their heating system before the end of its lifetime.
        cost_financing: DataFrame
            Cost of financing by dwelling (EUR).
        amount_debt: DataFrame
            Amount of debt by dwelling and income owner (EUR).
        amount_saving: DataFrame
            Amount of saving by dwelling and income owner (EUR).
        discount: float
            Discount rate.
        consumption_saved: DataFrame
            Consumption saved by dwelling and heating system (kWh).
        consumption_before: DataFrame
            Consumption before replacement by dwelling and heating system (kWh).
        consumption_no_rebound: DataFrame
            Consumption after replacement without rebound effect by dwelling and heating system (kWh).
        consumption_actual: DataFrame
            Consumption after replacement with rebound effect by dwelling and heating system (kWh).
        epc_upgrade: DataFrame
            Jump in EPC by dwelling and heating system (EPC).
        subsidies_loan: DataFrame
            Amount of eligible subsidies by dwelling and heating system (EUR).

        Returns
        -------
        None
        """

        # information stored during
        self._heater_store.update(
            {
                'cost_households': cost_heater,
                'subsidies_households': subsidies_total,
                'bill_saved_households': bill_saved,
                'replacement': replacement,
                'cost': replacement * cost_heater,
                'cost_financing': replacement * cost_financing,
                'vat': (replacement * vat_heater).sum().sum(),
                'subsidies': replacement * subsidies_total,
                'subsidies_loan': replacement * subsidies_loan,
                'debt': (replacement * amount_debt).sum(axis=1).groupby('Income owner').sum(),
                'saving': (replacement * amount_saving).sum(axis=1).groupby('Income owner').sum(),
                'discount': discount,
                'epc_upgrade': epc_upgrade,
                'flow_premature_replacement': flow_premature_replacement,
                'eligible': eligible,
                'hidden_cost_agg':  replacement * (- hidden_benefits_heater)
            }
        )

        self._heater_store['consumption_saved'] = (consumption_saved * replacement).sum().sum()

        temp = self.add_level(replacement.fillna(0), self._stock_ref, 'Income tenant')
        temp = temp.reorder_levels(consumption_before.index.names)

        consumption_before = self.apply_calibration((consumption_before * temp.T).T.sum(axis=1))
        consumption_actual = self.apply_calibration((consumption_actual * temp).sum(),
                                                    level_heater='Heating system final')
        consumption_saved_actual = (consumption_before - consumption_actual.T).T
        self._heater_store['consumption_saved_actual'] = consumption_saved_actual

        consumption_no_rebound = self.apply_calibration((consumption_no_rebound * temp).sum(),
                                                        level_heater='Heating system final')
        consumption_saved_no_rebound = (consumption_before - consumption_no_rebound.T).T
        self._heater_store['consumption_saved_no_rebound'] = consumption_saved_no_rebound

        self._heater_store['rebound'] = consumption_saved_no_rebound - consumption_saved_actual

        for key, item in subsidies_details.items():
            self._heater_store['subsidies_details'][key] = replacement * item

        for key, sub in self._heater_store['subsidies_details'].items():
            if key in self._heater_store['eligible'].keys():
                eligible = self._heater_store['eligible'][key].copy()
            else:
                eligible = sub.copy()
                eligible[eligible > 0] = 1
                eligible = eligible.any(axis=1)

            # Low carbon heating system to define extensive margin
            low_carbon_heating_system = ['Electricity-Heat pump water',
                                         'Electricity-Heat pump air',
                                         'Wood fuel-Performance boiler']

            replacement_eligible = replacement.fillna(0).sum(axis=1) * eligible
            cost = (((cost_heater - vat_heater) * replacement.fillna(0)).T * eligible).T

            self._heater_store['replacement_eligible'].update(
                {key: replacement_eligible.groupby('Housing type').sum()})

            if eligible.sum().sum() > 0:
                self._heater_store['subsidies_average'].update({key: sub.sum().sum() / replacement_eligible.sum()})
                self._heater_store['cost_average'].update({key: cost.sum().sum() / replacement_eligible.sum()})
            else:
                self._heater_store['subsidies_average'].update({key: 0})
                self._heater_store['cost_average'].update({key: 0})

    def heater_replacement(self, stock, prices, cost_heater, policies_heater, calib_heater=None,
                           step=1, financing_cost=None, district_heating=None, premature_replacement=None,
                           prices_before=None, supply=None, store_information=True, bill_rebate=0,
                           carbon_content=None, carbon_value=None, credit_constraint=True):
        """Function returns building stock updated after switching heating system.


        Parameters
        ----------
        stock: Series
        prices: Series
        cost_heater: Series
        calib_heater: DataFrame, optional
        policies_heater: list
        calib_heater: DataFrame, optional
        step: int, optional
        financing_cost: dict, optional
        district_heating: Series, optional
        premature_replacement: Series, optional
        prices_before: Series, optional
        supply: Series, optional
        store_information: bool, optional
        carbon_content: Series, optional
        carbon_value: float, optional
        bill_rebate: float, optional

        Returns
        -------
        Series
        """

        def apply_rational_choice(_consumption_saved, _subsidies_total, _cost_total, _bill_saved, _carbon_saved,
                                  _stock, social=False, discount_social=0.032):

            if social:
                _discount = discount_social

            # subsidies do not change market-share
            _bill_saved[_bill_saved == 0] = float('nan')

            ratio = (_cost_total - _subsidies_total) / _bill_saved
            if social:
                ratio = (_cost_total - _subsidies_total) / (_bill_saved + _carbon_saved)

            best_option = AgentBuildings.find_best_option(ratio, {'consumption_saved': _consumption_saved}, func='min')
            _market_share = DataFrame(0, index=ratio.index, columns=ratio.columns)
            for i in _market_share.index:
                _market_share.loc[i, best_option.loc[i, 'columns']] = 1

            assert (_market_share.sum(axis=1) == 1).all(), 'Market-share issue'

            return _market_share

        index = stock.index

        cost_heater_raw = cost_heater.copy()

        probability = self.heater_vintage.loc[:, 1] / self.heater_vintage.sum(axis=1)
        probability.dropna(inplace=True)
        probability *= step

        # premature replacement
        premature_heater = [p for p in policies_heater if p.policy == 'premature_heater']
        for premature in premature_heater:
            print('Premature replacement not Implemented')
            # temp = [i for i in probability.index if '{}'.format(i.split('-')[0]) in premature.target]
            # probability.loc[temp] = premature.value

        flow_replace = stock * reindex_mi(probability, stock.index)

        # condition
        condition = Series(True, index=index, dtype='float').to_frame().dot(Series(True, index=cost_heater.index).to_frame().T)
        condition = condition.astype(bool)

        # policies restriction
        restriction = [p for p in policies_heater if p.policy in ['restriction_energy', 'restriction_heater']]
        for policy in restriction:
            if isinstance(policy.value, str):
                temp = [policy.value]
            else:
                temp = policy.value
            if policy.policy == 'restriction_energy':
                temp = [i for i in cost_heater.index if '{}'.format(i.split('-')[0]) in temp]
            if temp:
                idx = condition.index
                if policy.target is not None:
                    idx = condition.index.get_level_values('Housing type') == policy.target
                condition.loc[idx, temp] = False

        # size heating system
        if self._variable_size_heater:
            size_heater = self.size_heater(index=index)
        else:
            size_heater = Series(8, index=index)

        # technical restriction - removing heat-pump for low-efficient buildings
        condition.columns.names = ['Heating system final']
        if self._constraint_heat_pumps:
            if isinstance(self._constraint_heat_pumps, list):
                _, certificate, _ = self.consumption_heating(method='3uses', full_output=True)
                condition = concat((condition, reindex_mi(certificate.rename('Performance'), condition.index)), axis=1)
                condition = condition.set_index('Performance', append=True)
                idx = (condition.index.get_level_values('Performance').isin(['F', 'G'])) & (
                    ~condition.index.get_level_values('Heating system').isin(self._resources_data['index']['Heat pumps']))
                condition.loc[idx, [i for i in self._resources_data['index']['Heat pumps'] if i in condition.columns]] = False
                condition = condition.droplevel('Performance')
            else:
                raise NotImplementedError
                size = self.size_heater(index=index)
                idx = (size > self._constraint_heat_pumps) & (
                    ~size.index.get_level_values('Heating system').isin(self._resources_data['index']['Heat pumps']))
                condition.loc[idx, [i for i in self._resources_data['index']['Heat pumps'] if i in condition.columns]] = False

        # technical restriction - heat pump can only be switch with heat pump
        idx = condition.index.get_level_values('Heating system').isin(self._resources_data['index']['Heat pumps'])
        condition.loc[idx, [i for i in condition.columns if i not in self._resources_data['index']['Heat pumps']]] = False

        # technical restriction - direct electric cannot switch to fossil
        fossil_boilers = ['Natural gas-Performance boiler', 'Oil fuel-Performance boiler']
        idx = condition.index.get_level_values('Heating system') == 'Electricity-Direct electric'
        condition.loc[idx, [i for i in condition.columns if i in fossil_boilers]] = False

        # technical restriction - wood boiler can only be switch with wood boiler
        wood_boilers = ['Wood fuel-Performance boiler', 'Wood fuel-Performance boiler']
        idx = condition.index.get_level_values('Heating system').isin(wood_boilers)
        condition.loc[idx, [i for i in condition.columns if i not in wood_boilers]] = False

        if False:
            # only dwelling fuel with natural gas can get natural gas
            idx = condition.index.get_level_values('Heating system').isin([i for i in condition.columns if i != 'Natural gas-Performance boiler'])
            condition.loc[idx, 'Natural gas-Performance boiler'] = False

            # multi-family not heated by wood cannot
            idx = condition.index.get_level_values('Heating system').isin([i for i in condition.columns if i != 'Wood fuel-Performance boiler']) & (condition.index.get_level_values('Housing type') == 'Multi-family')
            condition.loc[idx, 'Wood fuel-Performance boiler'] = False

        # bill saving
        choice_heater = cost_heater.index
        choice_heater_idx = Index(choice_heater, name='Heating system final')

        temp = Series(0, index=index, dtype='float').to_frame().dot(Series(0, index=choice_heater_idx, dtype='float').to_frame().T)
        index_final = temp.stack().index

        consumption_std, _, certificate = self.consumption_heating_store(index_final, level_heater='Heating system final')
        consumption_std = reindex_mi(consumption_std.unstack('Heating system final'), index)
        consumption_std = (reindex_mi(self._surface, consumption_std.index) * consumption_std.T).T
        bill_std = self.energy_bill(prices, consumption_std)
        """temp = [i.split('-')[0] for i in consumption.columns]
        prices_re = prices.reindex(temp).set_axis(consumption.columns)
        bill = consumption * prices_re"""

        consumption_std_before = self.consumption_heating_store(index, level_heater='Heating system')[0]
        consumption_std_before = reindex_mi(consumption_std_before, index) * reindex_mi(self._surface, index)
        bill_std_before = self.energy_bill(prices, consumption_std_before)

        bill_std_saved = - bill_std.sub(bill_std_before, axis=0)
        bill_saved_std = bill_std_saved.copy()

        certificate = reindex_mi(certificate.unstack('Heating system final'), index)
        certificate_before = self.consumption_heating_store(index)[2]
        certificate_before = reindex_mi(certificate_before, index)

        consumption_std_saved = (consumption_std_before - consumption_std.T).T

        consumption_std_before = self.add_attribute(consumption_std_before, 'Income tenant')
        consumption_std_before = consumption_std_before.reorder_levels(self.stock.index.names)
        index_consumption = consumption_std_before.index.intersection(self.stock.index)
        consumption_std_before = consumption_std_before.loc[index_consumption]
        heating_intensity_before = self.to_heating_intensity(consumption_std_before.index, prices,
                                                             consumption=consumption_std_before,
                                                             level_heater='Heating system',
                                                             bill_rebate=bill_rebate)
        bill_std_saved = self.add_attribute(bill_std_saved, 'Income tenant')
        bill_std_saved = bill_std_saved.reorder_levels(self.stock.index.names)
        bill_std_saved = bill_std_saved.loc[index_consumption]
        # x = pd.concat((consumption_std_before, reindex_mi(certificate_before, index=)))

        # bill saved that include rebound effect so increase of thermal comfort
        bill_saved = (bill_std_saved.T * heating_intensity_before).T
        bill_saved = bill_saved.groupby(index.names).mean()

        # emission saved
        consumption_before = consumption_std_before * heating_intensity_before
        emission_before = self.energy_bill(carbon_content, consumption_before, level_heater='Heating system')

        consumption_std = self.add_attribute(consumption_std, 'Income tenant')
        consumption_std = consumption_std.reorder_levels(self.stock.index.names)
        consumption_std = consumption_std.loc[index_consumption]
        consumption_std = consumption_std.stack('Heating system final')

        heating_intensity = self.to_heating_intensity(consumption_std.index, prices,
                                                      consumption=consumption_std,
                                                      level_heater='Heating system final',
                                                      bill_rebate=bill_rebate)
        consumption = (consumption_std.T * heating_intensity).T
        emission = self.energy_bill(carbon_content, consumption, level_heater='Heating system final')

        consumption = consumption.unstack('Heating system final')
        emission = emission.unstack('Heating system final')

        consumption_saved = (consumption_before - consumption.T).T
        consumption_saved = consumption_saved.groupby(index.names).mean()
        emission_saved = (emission_before - emission.T).T
        emission_saved = emission_saved.groupby(index.names).mean()

        # subsidies
        cost_heater = size_heater.to_frame().dot(cost_heater.to_frame().T)
        cost_heater, vat_heater, subsidies_details, subsidies_total, eligible = self.apply_subsidies_heater(index,
                                                                                                            policies_heater,
                                                                                                            cost_heater.copy(),
                                                                                                            consumption_std_saved,
                                                                                                            emission_saved,
                                                                                                            bill_saved)

        # cost financing
        p = [p for p in policies_heater if p.policy == 'zero_interest_loan']
        cost_total, cost_financing, amount_debt, amount_saving, discount, subsidies = self.calculate_financing(
            cost_heater,
            subsidies_total,
            financing_cost, policies=p)
        subsidies_loan = DataFrame(0, index=cost_total.index, columns=cost_total.columns)
        for policy in p:
            subsidies_details.update({policy.name: subsidies[policy.name]})
            subsidies_loan += subsidies[policy.name]

        if store_information:
            if prices_before is None:
                prices_before = prices

            heating_intensity_before = self.to_heating_intensity(consumption_std_before.index, prices_before,
                                                                 consumption=consumption_std_before,
                                                                 level_heater='Heating system',
                                                                 bill_rebate=bill_rebate)
            consumption_before = consumption_std_before * heating_intensity_before

            heating_intensity_after = self.to_heating_intensity(consumption_std.index, prices_before,
                                                                consumption=consumption_std,
                                                                level_heater='Heating system final',
                                                                bill_rebate=bill_rebate)
            consumption_actual = (consumption_std * heating_intensity_after)

            consumption_actual = consumption_actual.unstack('Heating system final')
            consumption_no_rebound = (consumption_std.unstack('Heating system final').T * heating_intensity_before).T

        epc_upgrade = - certificate.replace(EPC2INT).sub(
            certificate_before.replace(EPC2INT), axis=0)

        p = [p for p in policies_heater if p.policy == 'credit_constraint']
        if credit_constraint and not p:
            credit_constraint = self.credit_constraint(amount_debt, financing_cost, bill_saved=None)
        else:
            credit_constraint = None

        # district heating are automatically excluded from endogenous market share
        index_endogenous = index
        if 'Heating-District heating' in flow_replace.index.get_level_values('Heating system'):
            flow_replace_sum = flow_replace.sum()
            replace_district_heating = select(flow_replace, {'Heating system': 'Heating-District heating'})
            flow_replace = flow_replace.drop('Heating-District heating', level='Heating system')
            assert round(flow_replace_sum, 0) == round(flow_replace.sum() + replace_district_heating.sum(), 0), 'Error in flow replace'

            index_endogenous = index.drop(replace_district_heating.index)

        if self._endogenous:
            bill_saved_expectation = bill_saved.copy()
            if self._belief_engineering_calculation is None:
                bill_saved_expectation = bill_saved_std.copy()

            market_share, error_conditional = self.endogenous_market_share_heater(index_endogenous, bill_saved_expectation,
                                                                                  subsidies_total, cost_heater,
                                                                                  calib_heater=calib_heater,
                                                                                  _cost_financing=cost_financing,
                                                                                  condition=condition,
                                                                                  flow_replace=flow_replace,
                                                                                  drop_heater=[
                                                                                      'Heating-District heating'],
                                                                                  _credit_constraint=credit_constraint)

            hidden_benefits_heater = (error_conditional / abs(self.preferences_heater['cost'])) * 1000
            # to reduce number of combination if market_shares for one technology is too small it is removed
            market_share[market_share < 10 ** -2] = 0
            market_share = (market_share.T / market_share.sum(axis=1)).T


        else:
            market_share = self.exogenous_market_share_heater(index, cost_heater.columns)

        assert (market_share.sum(axis=1).round(0) == 1).all(), 'Market-share issue'

        # first considering the case where the heating system is replaced by district heating
        if district_heating is not None:
            temp = flow_replace[~flow_replace.index.get_level_values('Heating system').isin(
                ['Electricity-Heat pump water', 'Electricity-Heat pump air'])]
            temp = select(temp, {'Housing type': 'Multi-family'})
            if district_heating > temp.sum():
                district_heating = temp.sum()
            to_district_heating = district_heating * temp / temp.sum()
            to_district_heating = to_district_heating.reindex(flow_replace.index).fillna(0)
            flow_replace = flow_replace - to_district_heating

        replacement = (market_share.T * flow_replace).T

        if district_heating is not None:
            # adding heating system switching to district heating
            to_district_heating = concat((to_district_heating, replace_district_heating), axis=0)
            replacement = concat((replacement, to_district_heating.rename('Heating-District heating')), axis=1)

        assert_almost_equal(replacement.sum().sum(), flow_replace_sum, decimal=0)

        # indicator
        temp = replacement.fillna(0).groupby(['Housing type', 'Heating system']).sum()
        temp = (temp.T / temp.sum(axis=1)).T

        replacement = replacement.groupby(replacement.columns, axis=1).sum()
        replacement.columns.names = ['Heating system final']

        stock_replacement = replacement.stack('Heating system final')
        to_replace = replacement.sum(axis=1)

        stock = stock - to_replace
        # correct rounding error
        stock[stock < 0] = 0

        if self.year == self.first_year + 2:
            distortion_grouped = self.calculate_distortion_heater(stock, cost_heater, emission_saved, carbon_value,
                                                                  bill_saved)

            levels = [k for k in distortion_grouped.index.names if k != 'Technology']
            subsidies_grouped = self.weighted_average(subsidies_total, stock.copy(), levels=levels,
                                                      technology='heater')
            s = self.add_certificate(stock)
            s = self.add_energy(s)
            stock_grouped = s.groupby([i for i in subsidies_grouped.index.names if i in s.index.names]).sum()
            stock_grouped = stock_grouped[stock_grouped > 0]
            temp = concat((subsidies_grouped, distortion_grouped), axis=1, keys=['Subsidies', 'Distortion'])

            self._distortion_store.update({'heater': temp.copy()})

            if self.path_calibration is not None:
                from project.utils import make_scatter_plot
                temp = self._distortion_store['heater'].copy()
                temp = temp.reset_index()
                col_colors = 'Technology'
                temp[col_colors] = temp[col_colors].apply(lambda x: self._resources_data['colors'][x])
                #col_colors = None

                make_scatter_plot(temp, 'Subsidies', 'Distortion', 'Subsidies (Thousand euro)',
                                  'Distortion (Thousand euro)',
                                  annotate=False, save=os.path.join(self.path_calibration, 'subsidies_distortion_heater.png'),
                                  format_y=lambda y, _: '{:.0f}'.format(y / 1e3),
                                  format_x=lambda x, _: '{:.1f}'.format(x / 1e3),
                                  s=10, diagonal_line=True, col_colors=col_colors)

        # adding heating system final equal to heating system because no switch
        flow_premature_replacement = 0
        if premature_replacement is not None:
            def premature_func(x, alpha=10 ** -3):
                return 1 / (1 + exp(-alpha * x))

            alpha = fsolve(lambda a: premature_func(-500, alpha=a) - 0.10, 10 ** -3)[0]

            bill_saved[bill_saved <= 0] = float('nan')

            npv = - cost_total + subsidies_total + 3 * bill_saved
            npv = npv.loc[:, [i for i in npv.columns if i in self._resources_data['index']['Heat pumps']]]
            npv = npv.dropna(axis=1, how='all')

            # not implemented yet
            if supply is not None:
                if self.number_firms_heater is None:
                    self._markup_heater_store = supply['markup_heater']
                cost = cost_heater['Electricity-Heat pump water']
                weight = stock / stock.sum()

                def mark_up(price, u=npv, n=2, cost=cost, weight=weight):
                    u = u.squeeze()
                    u += - (price - cost)
                    proba = premature_func(u, alpha=alpha)
                    proba = proba.reindex(weight.index).fillna(0)
                    elasticity = - alpha * price * (1 - proba)
                    elasticity_global = (elasticity * weight).sum()
                    return 1 / (1 + 1 / (elasticity_global * n))

                if False and self.number_firms_heater is None:
                    mark_up_target = supply['markup_heater']
                    number_firms = fsolve(
                        lambda x: mark_up(cost * self._markup_heater_store, u=npv.copy(), n=x) - mark_up_target, 3)[0]

                    self.number_firms_heater = number_firms
                    self.logger.info('Number of firms heater market: {:.1f}'.format(number_firms))

                self.number_firms_heater = 2

                def cournot_equilibrium(price, n=self.number_firms_heater, cost=cost, u=npv.copy()):
                    return price / cost - mark_up(price, u=u, n=n)


                price = fsolve(cournot_equilibrium, cost * self._markup_heater_store)[0]
                if price != 0:
                    markup = price / cost
                    self._markup_heater_store = markup
                    self.logger.info('Equilibrium heater found. Markup: {:.2f}'.format(markup))
                else:
                    self.logger.info('ERROR No Equilibrium. Keeping the same markup. Markup: {:.2f}'.format(
                        self._markup_heater_store))

            proba_replace = premature_func(npv, alpha=alpha) * premature_replacement['information_rate']
            proba_replace.dropna(inplace=True)
            to_replace = (stock[flow_replace.index] * proba_replace.T).T

            if to_replace.sum().sum() > 1:
                # TODO: does not work if multiple heater
                flow_premature_replacement = to_replace.sum()

                flow_replace = to_replace.idxmin(axis=1).rename('Heating system final')
                flow_replace = concat((flow_replace, to_replace), axis=1)
                stock_replacement_premature = flow_replace.set_index('Heating system final', append=True).squeeze(
                    axis=1)
                stock = stock - to_replace.squeeze().reindex(stock.index).fillna(0)
                stock_replacement = concat((stock_replacement, stock_replacement_premature), axis=0)
                stock_replacement = stock_replacement.groupby(stock_replacement.index.names).sum()

        stock_replacement = stock_replacement[stock_replacement > 0]
        stock = concat((stock, Series(stock.index.get_level_values('Heating system'), index=stock.index,
                                      name='Heating system final')), axis=1).set_index('Heating system final', append=True).squeeze()
        stock = concat((stock.reorder_levels(stock_replacement.index.names), stock_replacement),
                       axis=0, keys=[False, True], names=['Heater replacement'])
        stock.sort_index(inplace=True)

        assert round(stock.sum() - self.stock_mobile.xs(True, level='Existing', drop_level=False).sum(),
                     0) == 0, 'Sum problem'

        if store_information:
            self.store_information_heater(cost_heater.copy(), subsidies_total, bill_saved, subsidies_details,
                                          stock_replacement.unstack('Heating system final'),
                                          vat_heater, cost_financing, amount_debt, amount_saving, discount,
                                          consumption_std_saved,
                                          consumption_before, consumption_no_rebound, consumption_actual, epc_upgrade,
                                          flow_premature_replacement, subsidies_loan, eligible, hidden_benefits_heater)
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
            index = subsidies_insulation.index
            if 'Housing type' in subsidies_insulation.index.names:
                index = subsidies_insulation[subsidies_insulation.index.get_level_values('Housing type') == i].index

            subsidy = DataFrame(0, index=index, columns=self._choice_insulation)
            subsidy.loc[index, idx[True, :, :, :]] = subsidy.loc[index, idx[True, :, :, :]].add(
                subsidies_insulation['Wall'] * self.surface_insulation.loc[i, 'Wall'], axis=0)
            subsidy.loc[index, idx[:, True, :, :]] = subsidy.loc[index, idx[:, True, :, :]].add(
                subsidies_insulation['Floor'] * self.surface_insulation.loc[i, 'Floor'], axis=0)
            subsidy.loc[index, idx[:, :, True, :]] = subsidy.loc[index, idx[:, :, True, :]].add(
                subsidies_insulation['Roof'] * self.surface_insulation.loc[i, 'Roof'], axis=0)
            subsidy.loc[index, idx[:, :, :, True]] = subsidy.loc[index, idx[:, :, :, True]].add(
                subsidies_insulation['Windows'] * self.surface_insulation.loc[i, 'Windows'], axis=0)
            subsidies[i] = subsidy.copy()

        if 'Housing type' in subsidies_insulation.index.names:
            subsidies = concat(list(subsidies.values()), axis=0)
        else:
            subsidies = concat(list(subsidies.values()), axis=0, keys=self.surface_insulation.index,
                               names=self.surface_insulation.index.names)

        if policy == 'subsidy_ad_valorem':
            # NotImplemented: ad_valorem with different subsididies rate
            value = [v for v in subsidies_insulation.stack().unique() if v != 0][0]
            subsidies[subsidies > 0] = value

        return subsidies

    def apply_subsidies_insulation(self, index, policies_insulation, cost_insulation, surface, certificate,
                                   certificate_before, certificate_before_heater, energy_saved_3uses,
                                   consumption_saved, carbon_content, calculate_condition=True, health_cost_saved=None,
                                   bill_saved=None):
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
        vat_insulation: DataFrame
        tax : float
        subsidies_details: dict
        subsidies_total: DataFrame
        condition: dict
        eligible: dict
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

            _condition = dict()
            if 'out_worst' in _list_conditions:
                out_worst = (~_certificate.isin(['G', 'F'])).T.multiply(_certificate_before.isin(['G', 'F'])).T
                out_worst = reindex_mi(out_worst, _index).fillna(False).astype('float')
                _condition.update({'out_worst': out_worst})

            if 'reach_best' in _list_conditions:
                in_best = (_certificate.isin(['A', 'B'])).T.multiply(~_certificate_before.isin(['A', 'B'])).T
                in_best = reindex_mi(in_best, _index).fillna(False).astype('float')
                _condition.update({'reach_best': in_best})

            if 'mitigation_option' in _list_conditions:
                # selecting best cost efficiency opportunities to reach A or B
                _c_saved = _consumption_saved.copy().replace(0, float('nan'))
                levels = ['Heating system final' if i == 'Heating system' else i for i in self._resources_data['index']['Dwelling']]
                _c_saved = _c_saved.groupby(levels).first()
                _c_insulation = reindex_mi(_cost_insulation, _c_saved.index)
                cost_saving = _c_insulation / _c_saved
                # cost_saving = reindex_mi(cost_saving, _index)
                _temp = reindex_mi(_certificate, cost_saving.index)
                # TODO: select_deep_renovation
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

                _condition.update({'mitigation_option': (best.T == best.min(axis=1)).T})

            if [i for i in ['best_efficiency', 'best_efficiency_fg', 'efficiency_100', 'fg'] if i in _list_conditions]:
                _cost_annualized = calculate_annuities(_cost_insulation)
                cost_saving = reindex_mi(_cost_annualized, _consumption_saved.index) / _consumption_saved
                best_efficiency = (cost_saving.T == cost_saving.min(axis=1)).T
                if 'best_efficiency' in _list_conditions:
                    _condition.update({'best_efficiency': best_efficiency.copy()})
                if 'best_efficiency_fg' in _list_conditions:
                    best_efficiency_fg = (best_efficiency.T & _certificate_before.isin(['G', 'F'])).T
                    _condition.update({'best_efficiency_fg': best_efficiency_fg})
                if 'efficiency_100' in _list_conditions:
                    _condition.update({'efficiency_100': cost_saving < 0.1})
                if 'fg' in _list_conditions:
                    fg = cost_saving.copy()
                    fg[fg > 0] = False
                    fg.loc[reindex_mi(_certificate_before, fg.index).isin(['G', 'F']), :] = True
                    _condition.update({'fg': fg})

            deep_condition = 2
            energy_condition_35, energy_condition_50 = 0.35, 0.5

            _epc_upgrade = - _certificate.replace(EPC2INT).sub(_certificate_before.replace(EPC2INT), axis=0)
            _epc_upgrade = reindex_mi(_epc_upgrade, _index)
            _certificate_before_heater = reindex_mi(_certificate_before_heater, _index)
            _certificate = reindex_mi(_certificate, _index)
            _epc_upgrade_all = - _certificate.replace(EPC2INT).sub(
                _certificate_before_heater.replace(EPC2INT),
                axis=0)
            _condition.update({'epc_upgrade_all': _epc_upgrade_all})
            # if 'deep_renovation' in _list_conditions:
            _condition.update({'deep_renovation': _epc_upgrade_all >= deep_condition})

            if 'epc_upgrade_min' in _list_conditions:
                _condition.update({'epc_upgrade_min': _epc_upgrade_all >= 1})

            if 'deep_renovation_fg' in _list_conditions:
                condition_deep_renovation = _epc_upgrade_all >= deep_condition
                _condition.update(
                    {'deep_renovation_fg': (condition_deep_renovation.T & _certificate_before.isin(['G', 'F'])).T})

            if 'deep_renovation_fge' in _list_conditions:
                condition_deep_renovation = _epc_upgrade_all >= deep_condition
                _condition.update({'deep_renovation_fge': (
                            condition_deep_renovation.T & _certificate_before.isin(['G', 'F', 'E'])).T})

            if 'fossil' in _list_conditions:
                fossil = ['Oil fuel-Performance boiler', 'Oil fuel-Standard boiler', 'Oil fuel-Collective boiler',
                          'Natural gas-Performance boiler', 'Natural gas-Standard boiler', 'Natural gas-Collective boiler'
                          ]
                _temp = pd.Series(_certificate.index.get_level_values('Heating system').isin(fossil), index=_certificate.index)
                _temp = reindex_mi(_temp, _index).fillna(False).astype('float')
                _condition.update({'fossil': _temp})

            if 'deep_renovation_low_income' in _list_conditions:
                low_income_condition = ['D1', 'D2', 'D3', 'D4', 'C1', 'C2']
                low_income_condition = _index.get_level_values('Income owner').isin(low_income_condition)
                low_income_condition = Series(low_income_condition, index=_index)
                condition_deep_renovation = _epc_upgrade_all >= deep_condition
                _condition.update(
                    {'deep_renovation_low_income': (low_income_condition & condition_deep_renovation.T).T})

            if 'deep_renovation_high_income' in _list_conditions:
                condition_deep_renovation = _epc_upgrade_all >= deep_condition
                high_income_condition = ['D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'C3', 'C4', 'C5']
                high_income_condition = _index.get_level_values('Income owner').isin(high_income_condition)
                high_income_condition = Series(high_income_condition, index=_index)
                _condition.update(
                    {'deep_renovation_high_income': (high_income_condition & condition_deep_renovation.T).T})

            if 'energy_condition_35' in _list_conditions:
                energy_condition_35 = _energy_saved_3uses >= energy_condition_35
                _condition.update({'energy_condition_35': reindex_mi(energy_condition_35, _index)})

            if 'energy_condition_50' in _list_conditions:
                energy_condition_50 = _energy_saved_3uses >= energy_condition_50
                _condition.update({'energy_condition_50': reindex_mi(energy_condition_50, _index)})

            if 'mpr_no_fg' in _list_conditions:
                _condition.update({'mpr_no_fg': ~_certificate_before.isin(['G', 'F'])})

            if 'heater_replacement' in _list_conditions:
                _temp = pd.Series(_certificate.index.get_level_values('Heater replacement'), index=_certificate.index)
                _condition.update({'heater_replacement': _temp})

            if 'heater_replacement_low_carbon' in _list_conditions:
                _temp = pd.Series(_certificate.index.get_level_values('Heater replacement'), index=_certificate.index)
                _temp = _temp & pd.Series(_certificate.index.get_level_values('Heating system final').isin(
                    self._resources_data['index']['Low Carbon']), index=_certificate.index)
                _temp = (_condition['deep_renovation'].T & _temp).T
                _condition.update({'heater_replacement_low_carbon': _temp})

            if 'heater_replacement_heat_pump' in _list_conditions:
                _temp = pd.Series(_certificate.index.get_level_values('Heater replacement'), index=_certificate.index)
                _temp = _temp & pd.Series(_certificate.index.get_level_values('Heating system final').isin(
                    self._resources_data['index']['Heat pumps']), index=_certificate.index)
                _temp = (_condition['deep_renovation'].T & _temp).T
                _condition.update({'heater_replacement_heat_pump': _temp})

            if 'mpr_no_fg_heater_replacement' in _list_conditions:
                _temp = pd.Series(_certificate.index.get_level_values('Heater replacement'), index=_certificate.index)
                _temp = _temp & reindex_mi(~_certificate_before.isin(['G', 'F']), _temp.index)
                """_temp = _certificate.loc[_certificate.index.get_level_values('Heater replacement'), :]
                _temp = ~_temp.isna()
                temp = (_temp.T * reindex_mi(~_certificate_before.isin(['G', 'F']), _temp.index)).T"""
                _condition.update({'mpr_no_fg_heater_replacement': _temp})

            if 'mpr_serenite_nb' in _list_conditions:
                nb_measures = Series([sum(i) for i in _certificate.columns], index=_certificate.columns)
                _temp = Series(0, index=_certificate.index)
                _temp[_temp.index.get_level_values('Heater replacement')] += 1
                _temp = concat([_temp] * nb_measures.shape[0], axis=1).set_axis(nb_measures.index, axis=1)
                nb_measures = _temp + nb_measures
                _condition.update({'mpr_serenite_nb': nb_measures >= 3})

            return _condition

        subsidies_details = {}

        vat = VAT
        p = [p for p in policies_insulation if 'reduced_vat' == p.policy]
        if p:
            vat = p[0].value
            if isinstance(vat, dict):
                vat = vat[self.year]
            subsidies_details.update({p[0].name: reindex_mi(cost_insulation * (VAT - vat), index)})

        vat_insulation = cost_insulation * vat
        cost_insulation += vat_insulation

        if calculate_condition:
            self._list_condition_subsidies = [i.target for i in policies_insulation if
                                              isinstance(i.target, str) and i.policy != 'subsidies_cap']
            list_conditions = self._list_condition_subsidies.copy()
            if not self._endogenous:
                list_conditions += ['mitigation_option']

            condition = defined_condition(index, certificate, certificate_before,
                                          certificate_before_heater,
                                          energy_saved_3uses, cost_insulation,
                                          consumption_saved, list_conditions
                                          )
        else:
            # TODO: do not work because _condition_store do not have _list_condition_subsidies
            condition = self._condition_store

        policies_incentive = ['subsidy_ad_valorem', 'subsidy_target', 'subsidy_proportional', 'subsidies_cap',
                              'subsidy_present_bias', 'subsidy_landlord', 'subsidy_multi_family']
        self.policies += [i.name for i in policies_insulation if i.policy in policies_incentive and i.name not in self.policies]

        sub_non_cumulative = {}
        for policy in [p for p in policies_insulation if p.policy in ['subsidy_ad_valorem', 'subsidy_target', 'subsidy_proportional',
                                                                      'subsidy_present_bias', 'subsidy_landlord', 'subsidy_multi_family']]:

            if isinstance(policy.value, dict):
                value = policy.value[self.year]
            else:
                value = policy.value

            if policy.policy == 'subsidy_target':
                if isinstance(value, Series):
                    idx = pd.Index(['Single-family', 'Multi-family'], name='Housing type')
                    value = concat([value] * len(idx), keys=idx, axis=1).T

                value = (reindex_mi(self.prepare_subsidy_insulation(value), index).T * surface).T
                if policy.target is not None:
                    if isinstance(condition[policy.target], Series):
                        value = (value.T * reindex_mi(condition[policy.target], value.index)).T
                    else:
                        raise NotImplemented('Need to implement target for subsidy_target')

            elif policy.policy == 'subsidy_ad_valorem':
                cost = policy.cost_targeted(reindex_mi(cost_insulation, index),
                                            target_subsidies=condition.get(policy.target))
                if isinstance(value, (Series, float, int)):
                    temp = reindex_mi(value, cost.index)
                    value = (temp * cost.T).T
                else:
                    temp = self.prepare_subsidy_insulation(value, policy=policy.policy)
                    value = reindex_mi(temp, cost.index) * cost
                    value = value.fillna(0)

            elif policy.policy == 'subsidy_proportional':
                consumption_saved_cumac = self.to_cumac(consumption_saved, discount=self.social_discount_rate,
                                                        duration=self.lifetime_insulation)
                consumption_saved_cumac[consumption_saved_cumac < 0] = 0
                # reindex_mi(cost_insulation, index) / consumption_saved_cumac
                if policy.proportional == 'tCO2_cumac':
                    temp = self.add_energy(consumption_saved_cumac)
                    emission_saved_cumac = (temp.T * reindex_mi(carbon_content.rename_axis(['Energy']),
                                                                temp.index)).T / 10 ** 3
                    value = value * emission_saved_cumac
                elif policy.proportional == 'MWh_cumac':
                    value = value * consumption_saved_cumac
                elif policy.proportional == 'health_cost':
                    value = self.to_cumac(health_cost_saved, discount=self.social_discount_rate,
                                          duration=self.lifetime_insulation) * 1e3
                    value = reindex_mi(value, index).fillna(0)
                else:
                    raise NotImplemented

            elif policy.policy == 'subsidy_present_bias':
                value = bill_saved.copy()
                value[value < 0] = 0
                value = (reindex_mi((self.preferences_insulation['discount_factor_perfect'] -
                                     self.preferences_insulation['discount_factor']), value.index) * value.T).T
            elif policy.policy == 'subsidy_landlord':
                value = self.landlord_dilemma.copy()
                value = reindex_mi(value, index).fillna(0)
                value = concat([value] * cost_insulation.shape[1], keys=cost_insulation.columns, axis=1)

            elif policy.policy == 'subsidy_multi_family':
                value = self.multifamily_friction.copy()
                value = reindex_mi(value, index).fillna(0)
                value = concat([value] * cost_insulation.shape[1], keys=cost_insulation.columns, axis=1)

            # cap for all policies
            value.fillna(0, inplace=True)
            cost = reindex_mi(cost_insulation, index)
            value = value.where(value <= cost, cost)

            if policy.cap is not None:
                if isinstance(policy.cap, dict):
                    cap = policy.cap[self.year]
                else:
                    cap = policy.cap

                cap = reindex_mi(cap, value.index).fillna(float('inf'))
                cap = concat([cap] * value.shape[1], keys=value.columns, axis=1)
                value = value.where(value < cap, cap)

            if not policy.social_housing:
                value.loc[value.index.get_level_values('Occupancy status') == 'Social-housing', :] = 0

            if policy.non_cumulative is None:
                if policy.name in subsidies_details.keys():
                    subsidies_details[policy.name] = subsidies_details[policy.name] + value
                else:
                    subsidies_details[policy.name] = value.copy()
            else:
                sub_non_cumulative.update({policy: value.copy()})

        # adding bonus subsidies (subsidies have been calculated before)
        subsidies_bonus = [p for p in policies_insulation if p.policy == 'bonus']
        for policy in subsidies_bonus:
            if isinstance(policy.value, dict):
                bonus_value = policy.value[self.year]
            else:
                bonus_value = policy.value

            if policy.target is not None:
                value = (reindex_mi(bonus_value, condition[policy.target].index) * condition[policy.target].T).T
            else:
                if policy.name not in subsidies_details.keys():
                    raise KeyError('Bonus subsidies should be coupled in other subsidy')
                value = reindex_mi(bonus_value, subsidies_details[policy.name].index)
            value.fillna(0, inplace=True)

            if not policy.social_housing:
                print('ok')
                value.loc[value.index.get_level_values('Occupancy status') == 'Social-housing', :] = 0

            # if policy defined by insulation
            if len(value.columns.names) == 1:
                value = (self.prepare_subsidy_insulation(value).T * surface).T

            if policy.name in subsidies_details.keys():
                if isinstance(value, (DataFrame, float, int)):
                    subsidies_details[policy.name] = subsidies_details[policy.name] + value
                elif isinstance(value, Series):
                    subsidies_details[policy.name] = (subsidies_details[policy.name].T + value).T
            else:
                subsidies_details[policy.name] = value.copy()

        # store eligible before cap and cumulative
        eligible = {}
        for policy in subsidies_details:
            eligible.update({policy: (subsidies_details[policy] > 0).any(axis=1).copy()})
        for policy in sub_non_cumulative:
            eligible.update({policy.name: (sub_non_cumulative[policy] > 0).any(axis=1).copy()})

        # for non-cumulative subsidies, we compare the value of the subsidies with the sum of non-cumulative subsidies
        for policy, value in sub_non_cumulative.items():
            compare = sum([subsidies_details[p] for p in policy.non_cumulative if p in subsidies_details.keys()])
            if isinstance(compare, DataFrame):
                value = reindex_mi(value, compare.index)
                subsidies_details[policy.name] = value.where(value > compare, 0)
                for p in policy.non_cumulative:
                    if p in subsidies_details.keys():
                        subsidies_details[p] = subsidies_details[p].where(compare > value, 0)
            else:
                subsidies_details[policy.name] = value

            """for policy_compare in policy.non_cumulative:
                if policy_compare in subsidies_details.keys():
                    comp = reindex_mi(subsidies_details[policy_compare], value.index)
                    subsidies_details[policy_compare] = comp.where(comp > value, 0)
                    subsidies_details[policy.name] = value.where(value > comp, 0)"""

        subsidies_total = [subsidies_details[k] for k in subsidies_details.keys() if k not in ['reduced_vat', 'over_cap']]
        if subsidies_total:
            subsidies_total = sum(subsidies_total)
        else:
            subsidies_total = pd.DataFrame(0, index=consumption_saved.index, columns=consumption_saved.columns)

        for k in subsidies_details.keys():
            subsidies_details[k].sort_index(inplace=True)

        # overall cap for cumulated amount of subsidies
        subsidies_cap = [p for p in policies_insulation if p.policy == 'subsidies_cap']
        if subsidies_cap:
            # only one subsidy cap
            subsidies_cap = subsidies_cap[0]
            policies_target = subsidies_cap.target
            subsidies_cap = reindex_mi(subsidies_cap.value, subsidies_total.index)
            cap = (reindex_mi(cost_insulation, index).T * subsidies_cap).T
            over_cap = subsidies_total > cap
            subsidies_details['over_cap'] = (subsidies_total - cap)[over_cap].fillna(0)
            subsidies_details['over_cap'].sort_index(inplace=True)
            remaining = subsidies_details['over_cap'].copy()

            for policy_name in policies_target:
                if policy_name in subsidies_details.keys():
                    self.logger.debug('Capping amount for {}'.format(policy_name))
                    temp = remaining.where(remaining <= subsidies_details[policy_name], subsidies_details[policy_name])
                    subsidies_details[policy_name] -= temp
                    assert (subsidies_details[policy_name].values >= 0).all(), '{} got negative values'.format(
                        policy_name)
                    remaining -= temp
                    if allclose(remaining, 0, rtol=10 ** -1):
                        break

            # it is possible to extend policies_target to other policies
            assert allclose(remaining, 0, rtol=10 ** -1), 'Over cap'
            subsidies_total -= subsidies_details['over_cap']
        else:
            cap = reindex_mi(cost_insulation, index)
            over_cap = subsidies_total > cap
            subsidies_details['over_cap'] = (subsidies_total - cap)[over_cap].fillna(0)
            subsidies_details['over_cap'].sort_index(inplace=True)
            remaining = subsidies_details['over_cap'].copy()

            for policy_name in [k for k in subsidies_details.keys() if k != 'over_cap']:
                self.logger.debug('Capping amount for {}'.format(policy_name))
                temp = remaining.where(remaining <= subsidies_details[policy_name], subsidies_details[policy_name])
                subsidies_details[policy_name] -= temp
                assert (subsidies_details[policy_name].values >= 0).all(), '{} got negative values'.format(
                    policy_name)
                remaining -= temp
                if allclose(remaining, 0, rtol=10 ** -1):
                    break

            # it is possible to extend policies_target to other policies
            assert allclose(remaining, 0, rtol=10 ** -1), 'Over cap'
            subsidies_total -= subsidies_details['over_cap']

        return cost_insulation, vat_insulation, vat, subsidies_details, subsidies_total, condition, eligible

    def calculate_distortion_insulation(self, index, stock, cost_insulation,
                             consumption_saved, health_cost_saved,
                             bill_saved, carbon_value, cap=True):
        """Calculate distortion amount for each possible insulation choice.
        """
        distortion_details = {}

        # 1. distortion landlord
        value = self.landlord_dilemma.copy()
        value = reindex_mi(value, index).fillna(0)
        value = concat([value] * cost_insulation.shape[1], keys=cost_insulation.columns, axis=1)
        value.fillna(0, inplace=True)
        distortion_details.update({'landlord_dilemma': value.copy()})

        # 2. distortion multi-family
        value = self.multifamily_friction.copy()
        value = reindex_mi(value, index).fillna(0)
        value = concat([value] * cost_insulation.shape[1], keys=cost_insulation.columns, axis=1)
        value.fillna(0, inplace=True)
        distortion_details.update({'multi_family': value.copy()})

        # 3. distortion present-bias
        value = bill_saved.copy()
        value[value < 0] = 0
        value = (reindex_mi((self.preferences_insulation['discount_factor_perfect'] -
                             self.preferences_insulation['discount_factor']), value.index) * value.T).T
        value.fillna(0, inplace=True)
        distortion_details.update({'bill_saved': value.copy()})

        # 4. distortion emission
        consumption_saved_cumac = self.to_cumac(consumption_saved, discount=self.social_discount_rate,
                                                duration=self.lifetime_insulation) * 1e3
        consumption_saved_cumac[consumption_saved_cumac < 0] = 0
        temp = self.add_energy(consumption_saved_cumac)
        value = (temp.T * reindex_mi(carbon_value.rename_axis(['Energy']), temp.index)).T
        value.fillna(0, inplace=True)
        value = value.droplevel('Energy')
        distortion_details.update({'emission_saved': value.copy()})

        # 5. distortion health cost
        value = self.to_cumac(health_cost_saved, discount=self.social_discount_rate,
                              duration=self.lifetime_insulation) * 1e3
        value = reindex_mi(value, index).fillna(0)
        distortion_details.update({'health_cost': value.copy()})

        distortion_total = sum([i for k, i in distortion_details.items()])

        if cap is True:
            cap = reindex_mi(cost_insulation, index)
            distortion_total = distortion_total.where(distortion_total <= cap, cap)

        levels = ['Occupancy status', 'Housing type', 'Performance', 'Energy', 'Income owner']
        distortion_grouped = self.weighted_average(distortion_total, stock.copy(), levels=levels, technology='insulation')

        return distortion_grouped

    def endogenous_renovation(self, stock, bill_saved, subsidies_total, cost_insulation, frequency_insulation,
                              policies_insulation, calib_renovation=None, min_performance=None, subsidies_details=None,
                              cost_financing=None, supply=None, credit_constraint=None, health_cost_saved=None):
        """Calculate endogenous retrofit based on discrete choice model.


        Utility variables are investment cost, energy bill saving, and subsidies.
        Preferences are object attributes defined initially.

        # bill saved calculated based on the new heating system
        # certificate before work and so subsidies before the new heating system

        Parameters
        ----------
        stock: Series
        bill_saved: DataFrame
        subsidies_total: DataFrame
        cost_insulation: DataFrame
        frequency_insulation: float or int
        stock: Series, default None
        calib_renovation: dict, optional
        min_performance: str, optional
        subsidies_details: dict, optional
        cost_financing: DataFrame, optional
        supply: DataFrame, optional
        credit_constraint: float, optional

        Returns
        -------
        Series
            Retrofit rate
        DataFrame
            Market-share insulation
        DataFrame
            Utility insulation
        """

        def market_share_func(u):
            return (exp(u).T / exp(u).sum(axis=1)).T

        def to_market_share(_bill_saved, _subsidies_total, _cost_total, _cost_financing=None, _credit_constraint=None,
                            full_output=False, _min_performance=None, _health_cost_saved=None):
            """Calculate market-share between insulation options.


            Parameters
            ----------
            _bill_saved: DataFrame
            _subsidies_total: DataFrame
            _cost_total: DataFrame
            _cost_financing: DataFrame, optional
            _credit_constraint: DataFrame, optional
            full_output: bool, optional

            Returns
            -------
            ms_intensive: DataFrame
            utility_intensive: DataFrame
                Include alternative specific constant between renovation options.
                But not between status quo and renovation.
            """

            pref_sub = reindex_mi(self.preferences_insulation['subsidy'], _subsidies_total.index).rename(None)
            utility_subsidies = (_subsidies_total.T * pref_sub).T / 1000

            pref_investment = reindex_mi(self.preferences_insulation['cost'], _cost_total.index).rename(None)
            utility_investment = (_cost_total.T * pref_investment).T / 1000

            utility_bill_saving = (_bill_saved.T * reindex_mi(self.preferences_insulation['bill_saved'],
                                                              _bill_saved.index)).T / 1000

            utility_intensive = utility_bill_saving + utility_investment + utility_subsidies

            if _cost_financing is not None:
                pref_financing = reindex_mi(self.preferences_insulation['cost'], _cost_financing.index).rename(None)
                utility_financing = (_cost_financing.T * pref_financing).T / 1000
                utility_intensive += utility_financing

            if self.constant_insulation_intensive is not None:
                utility_intensive += self.constant_insulation_intensive

            if _credit_constraint is not None:
                _credit_constraint = _credit_constraint.astype(float)
                _credit_constraint = _credit_constraint.replace(0, float('nan'))
                utility_intensive *= _credit_constraint
                utility_intensive.dropna(how='all', inplace=True)

            if _health_cost_saved is not None:
                _health_cost_saved = reindex_mi(_health_cost_saved, utility_intensive.index).fillna(0)
                utility_health = (_health_cost_saved.T * reindex_mi(self.preferences_insulation['bill_saved'],
                                                                  _health_cost_saved.index)).T / 1000
                utility_intensive += utility_health

            if _min_performance is not None:
                certificate = reindex_mi(self._consumption_store['certificate_renovation'], utility_intensive.index)
                _performance_constraint = certificate <= _min_performance
                _performance_constraint = _performance_constraint.astype(float)
                _performance_constraint = _performance_constraint.replace(0, float('nan'))
                utility_intensive *= _performance_constraint

            ms_intensive = market_share_func(utility_intensive)

            if self.constant_insulation_extensive is not None and False:
                # test if nested logit model is the same than conditional logit model
                expected_utility = log(exp(utility_intensive).sum(axis=1))
                probability_renovation = to_renovation_rate(expected_utility)
                ms_nested = (ms_intensive.T * probability_renovation).T
                ms_nested = concat((1 - probability_renovation, ms_nested), axis=1)

                u_logit = (utility_intensive.T + reindex_mi(self.constant_insulation_extensive, utility_intensive.index)).T
                u_logit = concat((Series(0, index=u_logit.index), u_logit), axis=1)
                ms_logit = market_share_func(u_logit)
                allclose(ms_nested, ms_logit)

            if full_output:
                return ms_intensive, utility_intensive, utility_bill_saving
            else:
                return ms_intensive, utility_intensive

        def renovation_func(u, proba=1):
            return proba * 1 / (1 + exp(- u))

        def to_renovation_rate(_expected_utility, _proba_replacement=1):
            """Calculate retrofit rate based on binomial logit model.

            Parameters
            ----------


            Returns
            -------
            Series
                Renovation rate for each household.
            """
            utility_renovate = _expected_utility.copy()
            if self.constant_insulation_extensive is not None:
                utility_constant = reindex_mi(self.constant_insulation_extensive, _expected_utility.index)
                utility_renovate = _expected_utility + utility_constant

            probability_renovation = renovation_func(utility_renovate, proba=_proba_replacement)

            return probability_renovation

        def nested_logit_formula(_bill_saved, _subsidies_total, _cost_total, _cost_financing=None):

            _bill_saved[_bill_saved == 0] = float('nan')
            _subsidies_total[_bill_saved == 0] = float('nan')
            _cost_total[_bill_saved == 0] = float('nan')

            pref_sub = reindex_mi(self.preferences_insulation['subsidy'], _subsidies_total.index).rename(None)
            utility_subsidies = (_subsidies_total.T * pref_sub).T / 1000

            pref_investment = reindex_mi(self.preferences_insulation['cost'], _cost_total.index).rename(None)
            utility_investment = (_cost_total.T * pref_investment).T / 1000

            utility_bill_saving = (_bill_saved.T * reindex_mi(self.preferences_insulation['bill_saved'],
                                                              _bill_saved.index)).T / 1000

            utility_intensive = utility_bill_saving + utility_investment + utility_subsidies

            if _cost_financing is not None:
                pref_financing = reindex_mi(self.preferences_insulation['cost'], _cost_financing.index).rename(None)
                utility_financing = (_cost_financing.T * pref_financing).T / 1000
                utility_intensive += utility_financing

            if self.constant_insulation_intensive is not None:
                utility_intensive += self.constant_insulation_intensive

            _market_share = market_share_func(utility_intensive)

            expected_utility = log(exp(utility_intensive).sum(axis=1))
            if self.constant_insulation_extensive is not None:
                utility_constant = reindex_mi(self.constant_insulation_extensive, expected_utility.index)
                expected_utility += utility_constant

            _renovation_rate = renovation_func(expected_utility)

            probability = (_renovation_rate * _market_share.T).T
            return probability

        def apply_endogenous_renovation(_bill_saved, _subsidies_total, _cost_total, _cost_financing=None, _stock=None,
                                        _supply=None, _cost_curve=True, _credit_constraint=None, _frequency_insulation=1,
                                        _min_performance=None, _health_cost_saved=None):
            if self.number_firms_insulation is None and _supply is not None:
                self._markup_insulation_store = _supply['markup_insulation']

            _market_share, _utility_intensive = to_market_share(_bill_saved, _subsidies_total,
                                                                _cost_total * self._markup_insulation_store,
                                                                _cost_financing=_cost_financing,
                                                                _credit_constraint=_credit_constraint,
                                                                _min_performance=_min_performance,
                                                                _health_cost_saved=health_cost_saved)
            # calculate conditional expectation of error term (unobserved cost or benefits)
            # formula is log(exp(total_utility).sum()) - utility_observed
            total_utility = (_utility_intensive.T + reindex_mi(self.constant_insulation_extensive, _utility_intensive.index)).T
            if _min_performance is None:
                total_utility = add_no_renovation(total_utility)

            constant_unobserved = self.constant_insulation.copy()
            if _min_performance is None:
                constant_unobserved = add_no_renovation(constant_unobserved)
            constant_unobserved = reindex_mi(constant_unobserved, total_utility.index)
            # constant_unobserved = constant_unobserved.reindex(total_utility.columns, axis=1)

            error = (log(exp(total_utility).sum(axis=1)) - total_utility.T).T
            # test
            # error = 0 - total_utility
            _unobserved_value = error + constant_unobserved

            expected_utility = log(exp(_utility_intensive).sum(axis=1))

            _renovation_rate = to_renovation_rate(expected_utility, _frequency_insulation)

            if _supply:
                _investment_insulation = (_cost_total.reindex(_market_share.index) * _market_share).sum(axis=1)

                pref_investment = reindex_mi(self.preferences_insulation['cost'], _investment_insulation.index).rename(
                    None)

                weight = stock / stock.sum()

                cost_renovation = (_renovation_rate * _investment_insulation).sum() / _renovation_rate.sum()

                def mark_up(price, u=expected_utility, n=2):
                    proba = 1 / (1 + exp(-u))
                    elasticity = pref_investment * price / 1000 * (1 - proba)
                    elasticity_global = (elasticity * weight).sum()
                    return 1 / (1 + 1 / (elasticity_global * n))

                def cournot_equilibrium(price, n=1, cost_renovation=cost_renovation, u=expected_utility):
                    return price / cost_renovation - mark_up(price, u=u, n=n)

                """temp_elasticity, temp_margin, temp_markup, temp_feedback = dict(), dict(), dict(), dict()
                for i in range(-5, 5, 1):
                    u = utility + abs(utility) * i / 10
                    renovation = (1 / (1 + exp(-u)) * _stock).sum() / 10**3
                    temp_elasticity.update({renovation: price_elasticity_demand(cost_renovation, u=u)})
                    mu = mark_up(cost_renovation, u=u, n=2)
                    temp_markup.update({renovation: mu})
                    feedback = (1 / (1 + exp(-(u+(mu-1)*utility_investment))) * _stock).sum() / 10**3

                    temp_feedback.update({mu: feedback})

                    # temp_margin.update({renovation: fsolve(cournot_equilibrium, cost_renovation, args=(1, cost_renovation, u))[0]})
                temp_markup = pd.Series(temp_markup)
                temp_feedback = pd.Series(temp_feedback)
                temp_elasticity = pd.Series(temp_elasticity)
                temp_margin = pd.Series(temp_margin)

                make_plot(pd.Series(temp_elasticity), 'Elasticity (Percent)', format_y=lambda y, _: '{:,.2f}'.format(y),
                          save=os.path.join(self.path_calibration, 'elasticity.png'), integer=False)
                make_plot(pd.Series(temp_margin), 'Price (Euro)', format_y=lambda y, _: '{:,.2f}'.format(y),
                          save=os.path.join(self.path_calibration, 'price.png'), integer=False)"""

                # self.number_firms_insulation = 2
                if self.number_firms_insulation is None:
                    if False and self.path_calibration is not None:
                        temp = {}
                        for n in range(1, 10, 2):
                            root, info_dict, ier, mess = fsolve(cournot_equilibrium, cost_renovation,
                                                                args=(n, cost_renovation),
                                                                full_output=True)
                            temp.update({n: root[0]})
                        make_plot(pd.Series(temp), 'Price (Euro)', format_y=lambda y, _: '{:,.2f}'.format(y),
                                  save=os.path.join(self.path_calibration, 'price_cournot'))

                    mark_up_target = _supply['markup_insulation']
                    number_firms = fsolve(
                        lambda x: mark_up(cost_renovation * self._markup_insulation_store, u=expected_utility,
                                          n=x) - mark_up_target, 3)[0]
                    self.number_firms_insulation = number_firms
                    self.logger.info('Number of firms insulation market: {:.1f}'.format(self.number_firms_insulation))

                root, info_dict, ier, mess = fsolve(cournot_equilibrium,
                                                    cost_renovation * self._markup_insulation_store,
                                                    args=(
                                                    self.number_firms_insulation, cost_renovation, expected_utility),
                                                    full_output=True)
                """flow_renovation_before = (_renovation_rate * _stock).sum() / 10**3
                cost_renovation_before = (_renovation_rate * _investment_insulation).sum() / _renovation_rate.sum()                
                print(flow_renovation_before)
                print(cost_renovation)
                print(self.number_firms_insulation)
                proba = 1 / (1 + exp(-expected_utility))
                elasticity = pref_investment * cost_renovation * _supply['markup_insulation'] / 1000 * (1 - proba)
                elasticity_global = (elasticity * weight).sum()
                print(elasticity_global)"""

                mu = root[0] / cost_renovation
                if mu != 0:
                    self._markup_insulation_store = mu
                    self.logger.info('Equilibrium insulation found. Markup: {:.2f}'.format(mu))

                    _market_share, _utility_intensive = to_market_share(_bill_saved, _subsidies_total,
                                                                        _cost_total * self._markup_insulation_store,
                                                                        _cost_financing=_cost_financing)
                    expected_utility = log(exp(_utility_intensive).sum(axis=1))
                    _renovation_rate = to_renovation_rate(expected_utility)

                else:
                    self.logger.info('ERROR No Equilibrium. Keeping the same markup')
                    """
                    for i in range(50):
                        print((1 + i / 10))
                        print(price_elasticity_demand_bis(cost_renovation * (1 + i / 10), u=utility))
                        print(mark_up(cost_renovation * (1 + i / 10), u=utility, n=self.number_firms_insulation))

                    """

                """flow_renovation_after = (_renovation_rate * _stock).sum() / 10**3
                cost_renovation_after = (_renovation_rate * _investment_insulation).sum() / _renovation_rate.sum()"""

            return _market_share, _renovation_rate, _unobserved_value

        def calibration_renovation(_stock, _cost_total, _bill_saved, _subsidies_total, _calib_renovation,
                                   _cost_financing=None, _credit_constraint=None, _frequency_insulation=1):

            def calibration(x, _stock, utilities, _ms_insulation_ini, _renovation_rate_ini, _target, _ref, proba,
                            _option='deviation'):

                _constant_insulation = Series(x[:len(_ms_insulation_ini) - 1], index=_ms_insulation_ini.index.drop(_ref))
                _constant_insulation = concat((_constant_insulation, Series(0, index=_ref)))
                _constant_renovation = pd.Series(
                    x[len(_ms_insulation_ini) - 1:len(_renovation_rate_ini) + len(_ms_insulation_ini) - 1],
                    index=_renovation_rate_ini.index)
                _scale = x[-1]

                _utility_intensive = (utilities + _constant_insulation) * _scale
                _ms = market_share_func(_utility_intensive)
                expected_utility = log(exp(_utility_intensive).sum(axis=1))
                _rate = renovation_func(expected_utility + reindex_mi(_constant_renovation, expected_utility.index), proba)

                flow_renovation = _rate * _stock
                f_replace = (_ms.T * flow_renovation).T

                market_share_agg = (f_replace.sum() / f_replace.sum().sum()).reindex(_ms_insulation_ini.index)
                rslt = market_share_agg - _ms_insulation_ini
                rslt.drop(_ref, inplace=True)

                flow_renovation_agg = flow_renovation.groupby(_renovation_rate_ini.index.names).sum()
                renovation_rate_agg = flow_renovation_agg / _stock.groupby(_renovation_rate_ini.index.names).sum()
                rslt = append(rslt, renovation_rate_agg - _renovation_rate_ini)

                if _option == 'deviation':
                    renovation_mean = (_rate * _stock).sum() / _stock.sum()
                    std_deviation_calc = (((_rate - renovation_mean) ** 2 * _stock).sum() / (
                            _stock.sum() - 1)) ** (1 / 2)
                    rslt = append(rslt, _target - std_deviation_calc)

                elif _option == 'ratio_min_max':
                    flow_renovation = self.add_certificate(flow_renovation).groupby('Performance').sum()
                    stock_performance = self.add_certificate(_stock).groupby('Performance').sum()
                    renovation_best = flow_renovation[[i for i in flow_renovation.index if i <= 'E']].sum() / stock_performance[[i for i in stock_performance.index if i <= 'E']].sum()
                    renovation_worst = flow_renovation[[i for i in flow_renovation.index if i >= 'F']].sum() / stock_performance[[i for i in stock_performance.index if i >= 'F']].sum()
                    factor = renovation_worst / renovation_best
                    rslt = append(rslt, _target - factor)

                elif _option == 'share_fg':
                    flow_renovation = self.add_certificate(flow_renovation).groupby('Performance').sum()
                    renovation_worst = flow_renovation[[i for i in flow_renovation.index if i >= 'F']].sum()
                    factor = renovation_worst / flow_renovation.sum()
                    rslt = append(rslt, _target - factor)

                elif _option == 'price_elasticity':
                    _ms_full = (_ms.T * _rate).T
                    _price_elasticity = self.preferences_insulation['cost'] * _scale * _cost_total / 1000 * (1 - _ms_full)
                    _price_elasticity = (_price_elasticity * _ms).sum(axis=1)
                    _price_elasticity_average = (_stock * _price_elasticity).sum() / _stock.sum()
                    rslt = append(rslt, _target - _price_elasticity_average)

                else:
                    raise NotImplemented

                return rslt

            self.logger.info('Calibration renovation nested-logit function')

            # target
            renovation_rate_ini = _calib_renovation['renovation_rate_ini']

            if self.no_friction is True:
                renovation_rate_agg  = (_stock * reindex_mi(renovation_rate_ini, _stock.index)).sum() / _stock.sum()
                renovation_rate_ini = pd.Series(renovation_rate_agg, index=pd.Index([True], name='Existing'))

            ms_insulation_ini = _calib_renovation['ms_insulation_ini']
            option = _calib_renovation['scale']['option']
            target = _calib_renovation['scale']['target']

            _, utility_intensive, utility_bill_saving = to_market_share(_bill_saved, _subsidies_total, _cost_total,
                                                                        _cost_financing=_cost_financing,
                                                                        full_output=True)

            if _credit_constraint is not None:
                _credit_constraint = _credit_constraint.astype(float)
                _credit_constraint = _credit_constraint.replace(0, float('nan'))
                utility_intensive *= _credit_constraint
                utility_intensive.dropna(how='all', inplace=True)

            # initialization
            ref = MultiIndex.from_tuples([(False, False, True, False)], names=['Wall', 'Floor', 'Roof', 'Windows'])

            constant_insulation_intensive = ms_insulation_ini.reindex(_subsidies_total.columns, axis=0).copy()
            constant_insulation_intensive[constant_insulation_intensive > 0] = 0
            constant_insulation_intensive.drop(ref, inplace=True)
            constant_insulation_intensive = constant_insulation_intensive.to_numpy()

            c = (_cost_total * ms_insulation_ini).sum(axis=1)
            constant_renovation = - abs(self.preferences_insulation['cost'] * c / 1000).mean()
            constant_renovation = Series(constant_renovation, index=renovation_rate_ini.index)
            constant_renovation = constant_renovation.to_numpy()

            x0 = append(constant_insulation_intensive, constant_renovation)
            x0 = append(x0, 1)
            root, info_dict, ier, mess = fsolve(calibration, x0, args=(
                _stock, utility_intensive, ms_insulation_ini, renovation_rate_ini,
                target, ref, _frequency_insulation, option), full_output=True)

            if ier != 1:
                raise ValueError('Renovation model did not converge')

            constant_insulation_intensive = Series(root[:len(ms_insulation_ini) - 1], index=ms_insulation_ini.index.drop(ref))
            constant_insulation_intensive = concat((constant_insulation_intensive, Series(0, index=ref)))
            constant_insulation_intensive = constant_insulation_intensive.loc[ms_insulation_ini.index]
            constant_renovation = pd.Series(
                root[len(ms_insulation_ini) - 1:len(renovation_rate_ini) + len(ms_insulation_ini) - 1],
                index=renovation_rate_ini.index)
            scale = root[-1]

            self.constant_insulation_intensive = constant_insulation_intensive * scale
            self.constant_insulation_extensive = constant_renovation
            self.apply_scale(scale, gest='insulation')

            # results
            _market_share, _renovation_rate, _ = apply_endogenous_renovation(_bill_saved, _subsidies_total, _cost_total,
                                                                             _stock=_stock,
                                                                             _cost_financing=_cost_financing,
                                                                             _frequency_insulation=_frequency_insulation,
                                                                             _credit_constraint=_credit_constraint)

            flow = _renovation_rate * _stock

            rate = flow.groupby(renovation_rate_ini.index.names).sum() / _stock.groupby(
                renovation_rate_ini.index.names).sum()
            compare_rate = concat((rate.rename('Calculated'), renovation_rate_ini.rename('Observed')),
                                  axis=1).round(3)
            agg = flow.groupby(renovation_rate_ini.index.names).sum()

            wtp = self.constant_insulation_extensive / self.preferences_insulation['cost']
            stock_ini = _stock.groupby(renovation_rate_ini.index.names).sum()

            details = concat((
                self.constant_insulation_extensive, rate, renovation_rate_ini, stock_ini / 10 ** 6,
                agg / 10 ** 3, wtp), axis=1, keys=['constant', 'calcul', 'observed', 'stock', 'flow', 'wtp']).round(decimals=3)
            if self.path_calibration is not None:
                details.to_csv(os.path.join(self.path_calibration, 'calibration_flow_renovation.csv'))

            flow_insulation = (flow * _market_share.T).T.sum()
            share = flow_insulation / flow_insulation.sum()
            compare_share = concat((share.rename('Calculated'), ms_insulation_ini.rename('Observed')),
                                   axis=1).round(3)

            wtp = self.constant_insulation_intensive / self.preferences_insulation['cost']
            details = concat(
                (self.constant_insulation_intensive, share, ms_insulation_ini, flow_insulation / 10 ** 3, wtp), axis=1,
                keys=['constant', 'calcul', 'observed', 'thousand', 'wtp']).round(decimals=3)
            if self.path_calibration is not None:
                details.to_csv(os.path.join(self.path_calibration, 'calibration_market_share_insulation.csv'))

            compare = concat((compare_rate, compare_share), ignore_index=True)
            if allclose(compare['Calculated'], compare['Observed'], rtol=10 ** -2):
                self.logger.debug('Calibration investment decision insulation worked')
            else:
                assert allclose(compare['Calculated'], compare['Observed'],
                                rtol=10 ** -2), 'Calibration insulation did not work'

            if self.path_calibration is not None and self.no_friction is False:
                # export utility coefficient
                cost = Series(self.preferences_insulation['cost'])
                cost = cost.reset_index().set_axis(['Attribute', 'Value'], axis=1)

                preference_benefits = self.preferences_insulation['bill_saved']
                preference_benefits = preference_benefits.reset_index().set_axis(['Attribute', 'Value'], axis=1)

                constant_insulation_extensive = self.constant_insulation_extensive.copy()
                constant_insulation_extensive.index = constant_insulation_extensive.index.map(lambda x: '{} | {}'.format(x[0], x[1]))
                constant_insulation_extensive = constant_insulation_extensive.reset_index().set_axis(['Attribute', 'Value'], axis=1)

                constant_insulation_intensive = self.constant_insulation_intensive.copy()
                names = constant_insulation_intensive.index.names
                constant_insulation_intensive.index = constant_insulation_intensive.index.map(lambda row: ", ".join([name for name, value in zip(names, row) if value]))
                constant_insulation_intensive = constant_insulation_intensive.reset_index().set_axis(['Attribute', 'Value'], axis=1)

                temp = {'Cost': cost,
                        'Benefits': preference_benefits,
                        'Constant insulation': constant_insulation_extensive,
                        'Constant measure': constant_insulation_intensive}
                temp = concat(temp, axis=0).droplevel(-1)
                temp.to_csv(os.path.join(self.path_calibration, 'coefficient_utility_insulation.csv'))

                # export hidden cost that result from calibration
                temp = concat((self.hidden_cost, self.landlord_dilemma, self.multifamily_friction), axis=1,
                              keys=['Hidden cost', 'Landlord-tenant dilemma', 'Multi-family friction']).round(0)
                temp.to_csv(os.path.join(self.path_calibration, 'parameters_friction_insulation.csv'))

                # export hidden cost for each renovation work type that result from calibration
                temp = self.hidden_cost_insulation.copy()
                temp = temp + self.hidden_cost.loc[('Single-family', 'Owner-occupied')]
                names = temp.index.names
                temp.index = temp.index.map(lambda row: ", ".join([name for name, value in zip(names, row) if value]))
                temp.to_csv(os.path.join(self.path_calibration, 'hidden_cost_insulation.csv'))

                # TODO: Clean this part
                ms_full = (_market_share.T * _renovation_rate).T
                price_elasticity = self.preferences_insulation['cost'] * _cost_total / 1000 * (1 - ms_full)
                price_elasticity = (price_elasticity * _market_share).sum(axis=1)
                df = price_elasticity.groupby(['Housing type', 'Occupancy status', 'Income owner']).describe()
                df.to_csv(os.path.join(self.path_calibration, 'price_elasticity_description.csv'))

                temp = concat((price_elasticity, stock), axis=1, keys=['Price elasticity', 'Stock'])
                make_hist(temp, 'Price elasticity', 'Housing type', 'Housing (Million)',
                          format_y=lambda y, _: '{:.0f}'.format(y / 10 ** 6), kde=True,
                          save=os.path.join(self.path_calibration, 'price_elasticity_insulation.png'))

                compare.to_csv(os.path.join(self.path_calibration, 'result_calibration.csv'))

        def test_sensitivity_ad_valorem(_stock, _cost_total, _bill_saved, _subsidies_total, _cost_financing,
                                          _credit_constraint=credit_constraint):

            # TODO: add average cost of renovation, consumption saved + add same graph for global renovation
            rslt = dict()
            for sub in range(0, 110, 10):
                sub /= 100
                _market_share, _renovation_rate, _ = apply_endogenous_renovation(_bill_saved,
                                                                              (_cost_total + _cost_financing) * sub,
                                                                              _cost_total,
                                                                              _cost_financing=_cost_financing,
                                                                              _credit_constraint=credit_constraint)
                rslt.update({sub: (_renovation_rate * _stock).sum() / 10 ** 3})

            if self.path_ini is not None:
                make_plot(Series(rslt), 'Renovation function of ad valorem subsidy (Thousand households)',
                          save=os.path.join(self.path_ini, 'sensi_ad_valorem.png'), integer=False, legend=False)

        def test_sensitivity(_stock, _cost_total, _bill_saved, _subsidies_total, _cost_financing,
                             _credit_constraint=credit_constraint):

            # bill saved
            result_bill_saved = dict()
            result_bill_saved['Average cost (Thousand euro)'] = dict()
            result_bill_saved['Flow renovation (Thousand)'] = dict()

            values = [-0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5]
            for v in values:
                v += 1
                _market_share, _renovation_rate, _ = apply_endogenous_renovation(_bill_saved * v, _subsidies_total,
                                                                                 _cost_total, _supply=None,
                                                                                 _cost_financing=_cost_financing,
                                                                                 _credit_constraint=credit_constraint)
                f_renovation = _renovation_rate * _stock
                f_replace = (f_renovation * _market_share.T).T
                result_bill_saved['Average cost (Thousand euro)'].update(
                    {v: (f_replace * _cost_total).sum().sum() / f_renovation.sum() / 10 ** 3})
                result_bill_saved['Flow renovation (Thousand)'].update({v: f_renovation.sum() / 10 ** 3})
            result_bill_saved['Average cost (Thousand euro)'] = Series(
                result_bill_saved['Average cost (Thousand euro)'])
            result_bill_saved['Flow renovation (Thousand)'] = Series(result_bill_saved['Flow renovation (Thousand)'])

            # subsidies
            result_subsidies = dict()
            result_subsidies['Average cost (Thousand euro)'] = dict()
            result_subsidies['Flow renovation (Thousand)'] = dict()

            values = [-0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5]
            for v in values:
                v += 1
                _market_share, _renovation_rate, _ = apply_endogenous_renovation(_bill_saved, _subsidies_total * v,
                                                                                 _cost_total, _supply=None,
                                                                                 _cost_financing=_cost_financing,
                                                                                 _credit_constraint=credit_constraint)
                f_renovation = _renovation_rate * _stock
                f_replace = (f_renovation * _market_share.T).T
                result_subsidies['Average cost (Thousand euro)'].update(
                    {v: (f_replace * _cost_total).sum().sum() / f_renovation.sum() / 10 ** 3})
                result_subsidies['Flow renovation (Thousand)'].update({v: f_renovation.sum() / 10 ** 3})
            result_subsidies['Average cost (Thousand euro)'] = Series(result_subsidies['Average cost (Thousand euro)'])
            result_subsidies['Flow renovation (Thousand)'] = Series(result_subsidies['Flow renovation (Thousand)'])

            # subsidies
            result_cost = dict()
            result_cost['Average cost (Thousand euro)'] = dict()
            result_cost['Flow renovation (Thousand)'] = dict()

            values = [-0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5]
            for v in values:
                v += 1
                _market_share, _renovation_rate, _ = apply_endogenous_renovation(_bill_saved, _subsidies_total,
                                                                                 _cost_total * v, _supply=None,
                                                                                 _cost_financing=_cost_financing,
                                                                                 _credit_constraint=credit_constraint)
                f_renovation = _renovation_rate * _stock
                f_replace = (f_renovation * _market_share.T).T
                result_cost['Average cost (Thousand euro)'].update(
                    {v: (f_replace * _cost_total).sum().sum() / f_renovation.sum() / 10 ** 3})
                result_cost['Flow renovation (Thousand)'].update({v: f_renovation.sum() / 10 ** 3})
            result_cost['Average cost (Thousand euro)'] = Series(result_cost['Average cost (Thousand euro)'])
            result_cost['Flow renovation (Thousand)'] = Series(result_cost['Flow renovation (Thousand)'])

            # result
            dict_df = {'Bill saving': result_bill_saved['Average cost (Thousand euro)'],
                       'Cost': result_cost['Average cost (Thousand euro)'],
                       'Subsidies': result_subsidies['Average cost (Thousand euro)']
                       }
            if self.path_ini is not None:
                make_plots(dict_df, 'Average cost (Thousand euro)',
                           save=os.path.join(self.path_ini, 'sensi_average_cost.png'))
            dict_df = {'Bill saving': result_bill_saved['Flow renovation (Thousand)'],
                       'Cost': result_cost['Flow renovation (Thousand)'],
                       'Subsidies': result_subsidies['Flow renovation (Thousand)']
                       }
            if self.path_ini is not None:
                make_plots(dict_df, 'Flow renovation (Thousand)',
                           save=os.path.join(self.path_ini, 'sensi_flow_renovation.png'))

        def test_policies(_stock, _subsidies_details, _cost_total, _bill_saved, _subsidies_total, _cost_financing,
                          _credit_constraint=None):
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

            _market_share, _renovation_rate, _ = apply_endogenous_renovation(_bill_saved, _subsidies_total, _cost_total,
                                                                          _supply=None, _cost_financing=_cost_financing,
                                                                          _credit_constraint=credit_constraint)

            f_retrofit = _renovation_rate * _stock
            f_replace = (f_retrofit * _market_share.T).T
            avg_cost_global = (f_replace * _cost_total).sum().sum() / f_retrofit.sum()

            _subsidies_details = {k: item for k, item in _subsidies_details.items() if k != 'over_cap'}
            rslt = dict()
            for key, _sub in _subsidies_details.items():
                _c = 0
                mask = _sub > 0
                eligible = (_sub > 0).any(axis=1)
                if key == 'reduced_vat':
                    _c = _sub
                    _sub = 0

                f_replace_eligible = f_replace.loc[eligible, :]

                _eligible = f_replace_eligible.sum().sum()
                _beneficiaries = f_replace[_sub > 0].sum().sum()

                avg_cost_eligible = (f_replace_eligible * _cost_total).sum().sum() / f_replace_eligible.sum().sum()
                if key == 'reduced_vat':
                    avg_sub = (f_replace_eligible * _c).sum().sum() / f_replace_eligible.sum().sum()
                else:
                    avg_sub = (f_replace_eligible * _sub).sum().sum() / f_replace_eligible.sum().sum()

                share_sub = avg_sub / avg_cost_eligible

                c_total = _cost_total + _c
                assert (c_total >= _cost_total).all().all(), 'Cost issue'
                sub_total = _subsidies_total - _sub
                assert (sub_total <= _subsidies_total).all().all(), 'Subsidies issue'
                ms_nosub, renovation_nosub , _ = apply_endogenous_renovation(_bill_saved, sub_total, c_total, _supply=None,
                                                                         _cost_financing=_cost_financing,
                                                                         _credit_constraint=credit_constraint)
                f_renovation_nosub = renovation_nosub * stock
                f_replace_nosub = (f_renovation_nosub * ms_nosub.T).T
                # avg_cost_global_sub = (f_replace_sub * _cost_total).sum().sum() / f_retrofit_sub.sum()

                f_replace_eligible_nosub = f_replace_nosub.loc[eligible, :]
                avg_cost_eligible_nosub = (f_replace_eligible_nosub * _cost_total).sum().sum() / f_replace_eligible_nosub.sum().sum()

                _additional_participant = f_replace_eligible.sum().sum() - f_replace_eligible_nosub.sum().sum()
                _additional_participant_share = _additional_participant / f_replace_eligible_nosub.sum().sum()
                _non_additional_participant = f_replace_eligible_nosub.sum().sum()
                _free_riders = _non_additional_participant / f_replace_eligible.sum().sum()
                # _free_riders = f_retrofit_sub.sum() / f_retrofit.sum()
                # _intensive_margin = (avg_cost_global - avg_cost_global_sub) / avg_cost_global_sub
                _intensive_margin = (avg_cost_eligible - avg_cost_eligible_nosub) / avg_cost_eligible_nosub

                rslt.update({key: Series(
                    [_eligible, _beneficiaries, avg_sub, share_sub,
                     _additional_participant, _non_additional_participant, _free_riders,
                     avg_cost_eligible, avg_cost_eligible_nosub, _intensive_margin],
                    index=['Eligible (hh)', 'Beneficiaries (hh)', 'Average sub (euro)', 'Share sub (%)',
                           'Additional participants (hh)', 'Non additional participants (hh)',
                           'Share of non additional participants (%)',
                           'Average cost eligible (EUR)', 'Average cost eligible wo sub (EUR)',
                           'Intensive margin benef (%)'])})

            rslt = DataFrame(rslt)
            if self.path_ini is not None:
                rslt.to_csv(os.path.join(self.path_ini, 'result_policies_assessment.csv'))
                make_sensitivity_tables(rslt, os.path.join(self.path_ini, 'result_policies_assessment.png'))

            return rslt

        def indicator_renovation_rate(_stock, _cost_insulation, _bill_saved, _subsidies_total, _cost_financing,
                                      _credit_constraint=credit_constraint):
            _market_share, _renovation_rate, _unobserved_value = apply_endogenous_renovation(_bill_saved, _subsidies_total,
                                                                          _cost_insulation,
                                                                          _stock=_stock,
                                                                          _cost_financing=_cost_financing,
                                                                          _credit_constraint=credit_constraint)

            discount_factor = 10

            npv = - _cost_insulation - _cost_financing + _subsidies_total + (
                    reindex_mi(discount_factor, bill_saved.index) * _bill_saved.T).T
            npv = (npv * _market_share).sum(axis=1)

            level = ['Existing', 'Occupancy status', 'Income owner', 'Housing type', 'Wall', 'Floor', 'Roof', 'Windows',
                     'Heating system']
            flow = (_renovation_rate * stock).groupby(level).sum()
            rate = flow / stock.groupby(level).sum()
            rate = self.add_certificate(rate)
            npv = npv.groupby(level).mean()

            npv = self.add_certificate(npv)

            df = pd.concat((rate, npv), axis=1, keys=['rate', 'npv'])

            temp = df.reset_index('Occupancy status').copy()
            temp['Occupancy status'] = temp['Occupancy status'].apply(lambda x: self._resources_data['colors'][x])
            make_scatter_plot(temp, 'npv', 'rate', 'Net present value (euro)', 'Renovation rate (%)',
                              col_colors='Occupancy status',
                              format_x=lambda y, _: '{:,.0f}'.format(y),
                              format_y=lambda y, _: '{:,.0%}'.format(y),
                              save=os.path.join(self.path_ini, 'renovation_rate_npv.png'), annotate=False)

            dict_df = {i: df for i in df.index.names if i not in ['Existing', 'Wall', 'Floor', 'Roof', 'Windows']}
            make_grouped_scatterplots(dict_df, 'npv', 'rate', colors=self._resources_data['colors'],
                                      save=os.path.join(self.path_ini, 'renovation_rate_ini.png'), n_columns=2,
                                      format_y=lambda y, _: '{:.0%}'.format(y))

            temp = DataFrame()
            for i in [i for i in rate.index.names if i not in ['Existing', 'Wall', 'Floor', 'Roof', 'Windows']]:
                temp = concat((temp, rate.groupby(i).describe()), axis=0)
            temp.round(4).to_csv(os.path.join(self.path_ini, 'renovation_rate_describe_ini.csv'))

            rate = flow / stock.groupby(level).sum()
            rate = self.add_certificate(rate)

            flow = (_renovation_rate * stock * _market_share.T).T.groupby(level).sum()
            for j in ['Wall', 'Floor', 'Roof', 'Windows']:
                t = flow.xs(True, level=j, axis=1).sum(axis=1)
                rate = t / stock.groupby(level).sum()
                temp = DataFrame()
                for i in [i for i in rate.index.names if i not in ['Existing', 'Wall', 'Floor', 'Roof', 'Windows']]:
                    temp = concat((temp, rate.groupby(i).describe()), axis=0)
                temp.round(4).to_csv(os.path.join(self.path_ini, 'renovation_rate_describe_ini_{}.csv'.format(j.lower())))

            flow = self.add_certificate(flow)
            temp = flow.groupby('Performance').sum() / flow.sum()

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
            _temp = concat((_temp, t)).loc[self.constant_insulation_extensive.index]
            self.constant_insulation_extensive = _temp.copy()


        index = stock.index
        cost_insulation = reindex_mi(cost_insulation, index)

        if self.constant_insulation_intensive is None and self.rational_behavior_insulation is None:
            calibration_renovation(stock, cost_insulation.copy(), bill_saved.copy(), subsidies_total.copy(),
                                   calib_renovation,
                                   _cost_financing=cost_financing.copy(),
                                   _credit_constraint=credit_constraint,
                                   _frequency_insulation=frequency_insulation)

            if self.path_ini is not None:
                indicator_renovation_rate(stock, cost_insulation, bill_saved, subsidies_total, cost_financing,
                                          _credit_constraint=credit_constraint)
                test_sensitivity_ad_valorem(stock, cost_insulation, bill_saved, subsidies_total, cost_financing,
                                            _credit_constraint=credit_constraint)
                test_sensitivity(stock, cost_insulation, bill_saved, subsidies_total, cost_financing,
                                 _credit_constraint=credit_constraint)
                test_policies(stock, subsidies_details, cost_insulation, bill_saved, subsidies_total, cost_financing,
                              _credit_constraint=credit_constraint)

        regulation = [p for p in policies_insulation if p.policy == 'regulation']
        if 'landlord_dilemma' in [p.name for p in regulation]:
            apply_regulation('Privately rented', 'Owner-occupied', 'Occupancy status')
        if 'multi_family' in [p.name for p in regulation]:
            apply_regulation('Multi-family', 'Single-family', 'Housing type')
        if 'present_bias' in [p.name for p in regulation]:
            v = [p for p in regulation if p.name == 'present_bias'][0].value
            discount_factor = (1 - (1 + v) ** -self.preferences_insulation['lifetime']) / v
            idx = self.preferences_insulation['bill_saved'].index
            self.preferences_insulation['bill_saved'] = Series(
                discount_factor * abs(self.preferences_insulation['cost']),
                index=idx)

        market_share, renovation_rate, unobserved_value = apply_endogenous_renovation(bill_saved, subsidies_total,
                                                                                       cost_insulation,
                                                                                       _stock=stock,
                                                                                       _cost_financing=cost_financing,
                                                                                       _supply=supply,
                                                                                       _credit_constraint=credit_constraint,
                                                                                       _frequency_insulation=frequency_insulation,
                                                                                       _min_performance=min_performance,
                                                                                      _health_cost_saved=health_cost_saved)
        # positive means benefits so negative means hidden cost
        hidden_cost = - (unobserved_value / abs(self.preferences_insulation['cost'])) * 1000

        return renovation_rate, market_share, hidden_cost

    def exogenous_renovation(self, stock, condition):
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
            retrofit_rate = Series(0, index=stock.index)
            retrofit_rate[retrofit_rate.index.get_level_values('Heater replacement')] = 1
        elif self.param_exogenous['target'] == 'worst':
            consumption = reindex_mi(self._consumption_store['consumption'], stock.index)
            temp = concat((consumption, stock), keys=['Consumption', 'Stock'], axis=1)
            temp = temp.sort_values('Consumption', ascending=False)
            temp['Stock'] = temp['Stock'].cumsum()
            idx = temp[temp['Stock'] < self.param_exogenous['number']].index
            retrofit_rate = Series(1, index=idx)
            # segment to complete
            to_complete = self.param_exogenous['number'] - stock[idx].sum()
            idx = temp[temp['Stock'] >= self.param_exogenous['number']].index
            if not idx.empty:
                idx = idx[0]
                to_complete = Series(to_complete / stock[idx],
                                     index=MultiIndex.from_tuples([idx], names=temp.index.names))
                retrofit_rate = concat((retrofit_rate, to_complete))
        else:
            raise NotImplemented

        """market_share = DataFrame(0, index=retrofit_rate.index, columns=choice_insulation)
        market_share.loc[:, (True, True, True, True)] = 1"""
        market_share = condition['mitigation_option'].astype(int)
        market_share = reindex_mi(market_share, retrofit_rate.index).dropna()

        assert market_share.loc[market_share.sum(axis=1) != 1].empty, 'Market share problem'

        return retrofit_rate, market_share

    def store_information_insulation(self, condition, cost_insulation_raw, tax, cost_insulation, cost_financing,
                                     vat_insulation, subsidies_details, subsidies_total, consumption_saved,
                                     consumption_saved_actual, consumption_saved_no_rebound, amount_debt, amount_saving,
                                     discount, subsidies_loan, eligible):
        """Store insulation information.


        Information are post-treated to weight by the number of replacement.

        Parameters
        ----------
        condition: dict
        cost_insulation_raw: Series
            Cost of insulation for each envelope component of losses surface (€/m2).
        tax: float
            vat to apply (%).
        cost_insulation: DataFrame
            Cost total for each dwelling and each insulation gesture (€). Financing cost included.
        cost_financing: DataFrame
            Financing cost  for each dwelling and each insulation gesture (€).
        vat_insulation: DataFrame
            vat applied to each insulation gesture cost (€).
        subsidies_details: dict
            Amount of subsidies for each dwelling and each insulation gesture (€).
        subsidies_total: DataFrame
            Total mount of subsidies for each dwelling and each insulation gesture (€).
        consumption_saved: DataFrame
        consumption_saved_actual: DataFrame
        consumption_saved_no_rebound: DataFrame
        amount_debt: Series
        amount_saving: Series
        discount:Series
        subsidies_loan
        """
        list_condition = [c for c in condition.keys() if c not in self._list_condition_subsidies]
        self._condition_store = {k: item for k, item in condition.items() if k in list_condition}

        self._renovation_store.update({
            'consumption_saved_households': consumption_saved,
            'consumption_saved_actual_households': consumption_saved_actual,
            'consumption_saved_no_rebound_households': consumption_saved_no_rebound,
            'cost_households': cost_insulation * self._markup_insulation_store,
            'cost_financing_households': cost_financing,
            'vat_households': vat_insulation,
            'subsidies_households': subsidies_total,
            'subsidies_loan_households': subsidies_loan,
            'subsidies_details_households': subsidies_details,
            'cost_component': cost_insulation_raw * self.surface_insulation * (1 + tax) * self._markup_insulation_store,
            'debt_households': amount_debt,
            'saving_households': amount_saving,
            'discount': discount,
            'eligible': eligible
        })

    def calculate_health_cost_saved(self, stock, health_cost):
        s = stock.rename_axis(index={'Heating system': 'Heating system before'}).rename_axis(
            index={'Heating system final': 'Heating system'})
        _, certificate, _ = self.consumption_heating(index=s.index, method='3uses', full_output=True)
        s = concat((s, reindex_mi(certificate, s.index).rename('Performance')), axis=1)
        s.set_index('Performance', append=True, inplace=True)
        s = s.squeeze()
        s = self.add_level(s, self.stock_mobile, 'Income tenant')
        health_cost_before = reindex_mi(health_cost, s.index)
        health_cost_before = (health_cost_before * s).groupby(
            [i for i in s.index.names if i != 'Income tenant']).sum() / s.groupby(
            [i for i in s.index.names if i != 'Income tenant']).sum()
        health_cost_before = health_cost_before.droplevel('Performance').rename_axis(
            index={'Heating system': 'Heating system final'}).rename_axis(
            index={'Heating system before': 'Heating system'})
        s = s.droplevel('Performance').rename_axis(index={'Heating system': 'Heating system final'}).rename_axis(
            index={'Heating system before': 'Heating system'})

        _, _, certificate_after_3uses = self.prepare_consumption(self._choice_insulation,
                                                                 index=s.index,
                                                                 level_heater='Heating system final',
                                                                 full_output=True,
                                                                 method_epc='3uses')

        c = reindex_mi(certificate_after_3uses, s.index)
        health_cost_after = {}
        for i, j in c.items():
            df = concat((j, s), keys=['Performance', 'Stock'], axis=1).dropna().set_index(
                'Performance', append=True).squeeze()
            health_cost_after.update({i: reindex_mi(health_cost, df.index).droplevel('Performance')})
        health_cost_after = DataFrame(health_cost_after).fillna(0)

        health_cost_after = (health_cost_after.T * s).T.groupby(
            [i for i in s.index.names if i != 'Income tenant']).sum()
        health_cost_after = (
                    health_cost_after.T / s.groupby([i for i in s.index.names if i != 'Income tenant']).sum()).T
        health_cost_saved = (health_cost_before - health_cost_after.T).T
        return health_cost_saved

    def insulation_replacement(self, stock_ini, prices, cost_insulation_raw, frequency_insulation,
                               policies_insulation=None, financing_cost=None,
                               calib_renovation=None, min_performance=None,
                               exogenous_social=None, prices_before=None, supply=None, carbon_value=None,
                               carbon_content=None, calculate_condition=True, bill_rebate=0,
                               credit_constraint=True, health_cost=None, default_quality=None):
        """Calculate insulation retrofit in the dwelling stock.

        1. Intensive margin
        2. Extensive margin
        Calibrate function first year.

        Consumption saving only depends on insulation work.
        However, certificate upgrade also consider if heating system have switched.

        To reduce calculation time, we reduce the number of combination by grouping.
        Cost, subsidies and constant depends on Housing type, Occupancy status, Housing type and Insulation performance.

        Parameters
        ----------
        stock_ini: Series
        prices: Series
        cost_insulation_raw: Series
            €/m2 of losses area by component.
        policies_insulation: list, optional
        frequency_insulation: float or int
        financing_cost: dict, optional
        calib_renovation: dict, optional
        min_performance: str, optional
        exogenous_social: DataFrame, optional
        prices_before: Series; optional
        supply: dict, optional
        carbon_value: float, optional
        carbon_content: float, optional
        calculate_condition: bool, optional
        bill_rebate: float, optional
        credit_constraint: bool, optional

        Returns
        -------
        Series
            Retrofit rate
        DataFrame
            Market-share insulation
        """

        stock = self.add_certificate(stock_ini)
        stock = stock[stock.index.get_level_values('Performance').astype(str) > 'B']
        stock = stock.droplevel('Performance')
        index = stock.index

        performance_insulation = self._performance_insulation_renovation.copy()
        if default_quality is not None:
            performance_insulation = {k: i * (1 + default_quality) for k, i in performance_insulation.items()}

        to_drop = select(stock, performance_insulation)
        if not to_drop.empty:
            index = index.drop(to_drop.index)
            stock = stock.loc[index]

        if not stock.empty:
            # select index that can undertake insulation replacement
            _, _, certificate_before_heater = self.consumption_heating_store(index, level_heater='Heating system')

            # before include the change of heating system
            consumption_std_before, consumption_3uses_before, certificate_before = self.consumption_heating_store(index, level_heater='Heating system final')
            _consumption_std_before = consumption_std_before.copy()

            # calculation of energy_saved_3uses after heating system final
            consumption_std_after, consumption_3uses, certificate_after = self.prepare_consumption(
                self._choice_insulation,
                index=index,
                level_heater='Heating system final',
                default_quality=default_quality)

            energy_saved_3uses = ((consumption_3uses_before - consumption_3uses.T) / consumption_3uses_before).T
            energy_saved_3uses.dropna(inplace=True)

            consumption_std_saved = (consumption_std_before - consumption_std_after.T).T
            _consumption_std_saved = (reindex_mi(consumption_std_saved, index).T * reindex_mi(self._surface, index)).T

            # energy bill before
            consumption_std_before = reindex_mi(consumption_std_before, index) * reindex_mi(self._surface, index)
            _consumption_std_before = consumption_std_before.copy()
            energy_bill_before = self.energy_bill(prices, consumption_std_before, level_heater='Heating system final',
                                                  bill_rebate=bill_rebate)
            consumption_std_before = self.add_attribute(consumption_std_before, 'Income tenant')
            consumption_std_before = consumption_std_before.unstack(['Heater replacement', 'Heating system final'])
            consumption_std_before = consumption_std_before.reorder_levels(self.stock.index.names)
            index_consumption = consumption_std_before.index.intersection(self.stock.index)
            consumption_std_before = consumption_std_before.loc[index_consumption]
            consumption_std_before = consumption_std_before.stack(['Heater replacement', 'Heating system final'])
            consumption_std_before = consumption_std_before.unstack('Income tenant')
            consumption_std_before = consumption_std_before.reorder_levels(index.names)
            consumption_std_before = consumption_std_before.stack('Income tenant')
            index_consumption = consumption_std_before.index
            heating_intensity_before = self.to_heating_intensity(consumption_std_before.index, prices,
                                                                 consumption=consumption_std_before,
                                                                 level_heater='Heating system final',
                                                                 bill_rebate=bill_rebate)

            consumption_std_after = self.prepare_consumption(self._choice_insulation, index=index,
                                                             level_heater='Heating system final', full_output=False,
                                                             default_quality=default_quality)
            consumption_std_after = reindex_mi(consumption_std_after, index).reindex(self._choice_insulation, axis=1)
            consumption_std_after = (consumption_std_after.T * reindex_mi(self._surface, index)).T
            _consumption_std_after = consumption_std_after.copy()
            energy_bill_after = self.energy_bill(prices, consumption_std_after, level_heater='Heating system final',
                                                 bill_rebate=bill_rebate)

            bill_saved = - energy_bill_after.sub(energy_bill_before, axis=0).dropna()
            bill_saved_std = bill_saved.copy()
            bill_saved = self.add_attribute(bill_saved, 'Income tenant')
            bill_saved = bill_saved.reorder_levels(index_consumption.names)
            bill_saved = bill_saved.loc[index_consumption]
            bill_saved = (bill_saved.T * heating_intensity_before).T
            weights = self.add_level(stock, self.stock_mobile, 'Income tenant')
            bill_saved = (bill_saved.T * weights).T
            bill_saved = (bill_saved.groupby(index.names).sum().T / weights.groupby(index.names).sum()).T

            consumption_saved = - _consumption_std_after.sub(_consumption_std_before, axis=0).dropna()
            consumption_saved = self.add_attribute(consumption_saved, 'Income tenant')
            consumption_saved = consumption_saved.reorder_levels(index_consumption.names)
            consumption_saved = consumption_saved.loc[index_consumption]
            consumption_saved = (consumption_saved.T * heating_intensity_before).T
            weights = self.add_level(stock, self.stock_mobile, 'Income tenant')
            consumption_saved = (consumption_saved.T * weights).T
            consumption_saved = (consumption_saved.groupby(index.names).sum().T / weights.groupby(index.names).sum()).T

            health_cost_saved = self.calculate_health_cost_saved(stock, health_cost)

            cost_insulation = self.prepare_cost_insulation(cost_insulation_raw * self.surface_insulation)
            cost_insulation = cost_insulation.T.multiply(self._surface, level='Housing type').T

            surface = reindex_mi(self._surface, index)
            cost_insulation, vat_insulation, tax, subsidies_details, subsidies_total, condition, eligible = self.apply_subsidies_insulation(
                index, policies_insulation, cost_insulation, surface, certificate_after, certificate_before,
                certificate_before_heater, energy_saved_3uses, consumption_saved, carbon_content,
                calculate_condition=calculate_condition, health_cost_saved=health_cost_saved, bill_saved=bill_saved)

            p = [p for p in policies_insulation if p.policy == 'zero_interest_loan']
            for pp in p:
                if pp.target is not None:
                    pp.target = condition[pp.target]
            cost_total, cost_financing, amount_debt, amount_saving, discount, subsidies = self.calculate_financing(
                reindex_mi(cost_insulation, index),
                subsidies_total,
                financing_cost,
                policies=p
            )
            subsidies_loan = DataFrame(0, index=cost_total.index, columns=cost_total.columns)
            for policy in p:
                subsidies_details.update({policy.name: subsidies[policy.name]})
                subsidies_loan += subsidies[policy.name]

            p = [p for p in policies_insulation if p.policy == 'credit_constraint']
            if credit_constraint and not p:
                credit_constraint = self.credit_constraint(amount_debt, financing_cost, bill_saved=None)
            else:
                credit_constraint = None

            cost_insulation_reindex = reindex_mi(cost_insulation, index)
            assert allclose(amount_debt + amount_saving + subsidies_total, cost_insulation_reindex), 'Sum problem'

            p = [p for p in policies_insulation if (p.policy == 'regulation') and (p.name == 'health_cost')]
            if p:
                health_cost_saved_subsidies = health_cost_saved.copy()
            else:
                health_cost_saved_subsidies = None

            if self._endogenous:

                bill_saved_expectation = bill_saved.copy()
                if self._belief_engineering_calculation is True:
                    bill_saved_expectation = bill_saved_std.copy()

                renovation_rate, market_share, hidden_cost = self.endogenous_renovation(
                    stock, bill_saved_expectation,
                    subsidies_total,
                    cost_insulation_reindex,
                    frequency_insulation,
                    policies_insulation,
                    calib_renovation=calib_renovation,
                    min_performance=min_performance,
                    subsidies_details=subsidies_details,
                    cost_financing=cost_financing,
                    supply=supply,
                    credit_constraint=credit_constraint,
                    health_cost_saved=health_cost_saved_subsidies)

                self._renovation_store.update({'hidden_cost': hidden_cost})

                if exogenous_social is not None:
                    index = renovation_rate[
                        renovation_rate.index.get_level_values('Occupancy status') == 'Social-housing'].index
                    s = self.add_certificate(stock[index])
                    renovation_rate_social = reindex_mi(exogenous_social.loc[:, self.year], s.index).droplevel(
                        'Performance')
                    renovation_rate.drop(index, inplace=True)
                    renovation_rate = concat((renovation_rate, renovation_rate_social), axis=0)

            else:
                renovation_rate, market_share = self.exogenous_renovation(stock, condition)

            if self.year == self.first_year + 2 and self.no_friction is False:
                distortion_grouped = self.calculate_distortion_insulation(index, stock, cost_insulation,
                                                                          consumption_saved, health_cost_saved,
                                                                          bill_saved, carbon_value, cap=True)

                levels = [k for k in distortion_grouped.index.names if k != 'Technology']
                subsidies_grouped = self.weighted_average(subsidies_total, stock.copy(), levels=levels,
                                                          technology='insulation')
                s = self.add_certificate(stock)
                s = self.add_energy(s)
                stock_grouped = s.groupby([i for i in subsidies_grouped.index.names if i in s.index.names]).sum()
                stock_grouped = stock_grouped[stock_grouped > 0]
                temp = concat((subsidies_grouped, distortion_grouped), axis=1, keys=['Subsidies', 'Distortion'])

                self._distortion_store.update({'insulation': temp.copy()})

                if self.path_calibration is not None:

                    from project.utils import make_scatter_plot
                    temp = self._distortion_store['insulation'].copy()
                    temp = temp.reset_index()
                    col_colors = 'Income owner'
                    temp[col_colors] = temp[col_colors].apply(lambda x: self._resources_data['colors'][x])
                    col_colors = None

                    make_scatter_plot(temp, 'Subsidies', 'Distortion', 'Subsidies (Thousand euro)',
                                      'Distortion (Thousand euro)',
                                      annotate=False, save=os.path.join(self.path_calibration, 'subsidies_distortion_insulation.png'),
                                      format_y=lambda y, _: '{:.0f}'.format(y/1e3),
                                      format_x=lambda x, _: '{:.0f}'.format(x/1e3),
                                      s=10, diagonal_line=True, col_colors=col_colors)

            if self._condition_store is None and calculate_condition:

                if prices_before is None:
                    prices_before = prices

                surface = self._surface.xs(True, level='Existing')
                consumption_before = self.add_attribute(_consumption_std_before, [i for i in surface.index.names if
                                                                             i not in _consumption_std_before.index.names])
                consumption_before = self.add_attribute(consumption_before, 'Income tenant')
                heating_intensity = self.to_heating_intensity(consumption_before.index, prices,
                                                              consumption=consumption_before,
                                                              level_heater='Heating system final',
                                                              bill_rebate=bill_rebate)

                consumption_before *= heating_intensity

                consumption_after = self.add_attribute(_consumption_std_after, [i for i in surface.index.names if
                                                                           i not in _consumption_std_after.index.names])
                consumption_after = self.add_attribute(consumption_after, 'Income tenant')
                heating_intensity_after = self.to_heating_intensity(consumption_after.index, prices_before,
                                                                    consumption=consumption_after,
                                                                    level_heater='Heating system final',
                                                                    bill_rebate=bill_rebate)

                consumption_saved_actual = (consumption_before - (consumption_after * heating_intensity_after).T).T
                consumption_saved_no_rebound = (consumption_before - consumption_after.T * heating_intensity).T

                self.store_information_insulation(condition, cost_insulation_raw, tax, cost_insulation, cost_financing,
                                                  vat_insulation, subsidies_details, subsidies_total, _consumption_std_saved,
                                                  consumption_saved_actual, consumption_saved_no_rebound,
                                                  amount_debt, amount_saving, discount, subsidies_loan, eligible)

            return renovation_rate, market_share
        else:
            renovation_rate = Series(0, index=stock_ini.index)
            market_share = DataFrame(0, index=stock_ini.index, columns=self._choice_insulation)
            market_share.iloc[:, -1] = 1

            return renovation_rate, market_share

    def flow_retrofit(self, prices, cost_heater, cost_insulation, frequency_insulation,
                      policies_heater=None, policies_insulation=None, calib_heater=None, district_heating=None,
                      financing_cost=None, calib_renovation=None,
                      step=1, exogenous_social=None, premature_replacement=None, prices_before=None, supply=None,
                      carbon_value_kwh=None, carbon_value=None, bill_rebate=0, carbon_content=None,
                      health_cost=None, default_quality=None, credit_constraint=True):
        """Compute heater replacement and insulation retrofit.


        1. Heater replacement based on current stock segment.
        2. Knowing heater replacement (and new heating system) calculating retrofit rate by segment and market
        share by segment.
        3. Then, managing inflow and outflow.

        Parameters
        ----------
        prices: Series
        cost_heater: Series
        calib_heater: DataFrame
        cost_insulation
        policies_heater: list
            Policies for heating system.
        policies_insulation: list
            Policies for insulation.
        financing_cost: optional, dict
        calib_renovation: dict, optional

        Returns
        -------
        Series
        """

        self.logger.info('Number of agents: {:,.0f}'.format(self.stock.shape[0]))

        stock_mobile = self.stock_mobile.groupby(
            [i for i in self.stock_mobile.index.names if i != 'Income tenant']).sum()
        stock_mobile = stock_mobile.xs(True, level='Existing', drop_level=False)

        # calculate average heating intensity
        temp = concat((self.stock_mobile,
                       self.to_heating_intensity(self.stock_mobile.index, prices, level_heater='Heating system')),
                      axis=1, keys=['Stock', 'Heating intensity'])
        self._heating_intensity_avg = temp['Stock'].mul(temp['Heating intensity']).sum() / temp['Stock'].sum()

        # accounts for heater replacement - depends on energy prices, cost and policies heater
        self.logger.info('Calculation heater replacement')
        stock = self.heater_replacement(stock_mobile, prices, cost_heater, policies_heater,
                                        calib_heater=calib_heater, step=1, financing_cost=financing_cost,
                                        district_heating=district_heating, premature_replacement=premature_replacement,
                                        prices_before=prices_before, bill_rebate=bill_rebate,
                                        carbon_content=carbon_content, carbon_value=carbon_value,
                                        credit_constraint=credit_constraint)

        if supply is not None:
            if supply['heater']:
                self.logger.info('Solving supply demand equilibrium on heater market')

                def marginal_cost_heater(cost_heater, demand, alpha, demand_ini):
                    return (cost_heater + alpha * (demand - demand_ini)).where(demand > demand_ini, cost_heater)

                if self.cost_curve_heater is None:

                    demand = stock.xs(True, level='Heater replacement').groupby('Heating system final').sum()

                    def func(x, index):
                        alpha = Series(x[:index.shape[0]], index=index)
                        demand_ini = Series(x[index.shape[0]:], index=index)
                        y0 = (marginal_cost_heater(cost_heater, demand * 2, alpha, demand_ini) - cost_heater * 2).to_numpy()
                        y1 = (marginal_cost_heater(cost_heater, demand, alpha, demand_ini) - cost_heater).to_numpy()
                        return append(y0, y1)

                    index = demand.index
                    x0 = append(array([10] * index.shape[0]), demand.to_numpy())
                    root = fsolve(lambda x: func(x, index), x0)
                    alpha = Series(root[:index.shape[0]], index=index)
                    demand_ini = Series(root[index.shape[0]:], index=index)
                    self.cost_curve_heater = (alpha, demand_ini)

                def supply_demand_heater_equilibrium(_prices_heater, _prices_index):
                    _prices_heater = pd.Series(_prices_heater, index=_prices_index)

                    _stock = self.heater_replacement(stock_mobile, prices, _prices_heater, policies_heater,
                                                     financing_cost=financing_cost,
                                                     district_heating=district_heating,
                                                     premature_replacement=premature_replacement,
                                                     )
                    demand = stock.xs(True, level='Heater replacement').groupby('Heating system final').sum()

                    price_supply = marginal_cost_heater(cost_heater, demand, self.cost_curve_heater[0], self.cost_curve_heater[1])
                    return price_supply - _prices_heater

                prices_heater = fsolve(supply_demand_heater_equilibrium, cost_heater.to_numpy(),
                                       args=(cost_heater.index), xtol=10**-1)
                prices_heater = Series(prices_heater, index=cost_heater.index)

                stock = self.heater_replacement(stock_mobile, prices, prices_heater, policies_heater,
                                                calib_heater=calib_heater, step=1, financing_cost=financing_cost,
                                                district_heating=district_heating, premature_replacement=premature_replacement,
                                                prices_before=prices_before)
                assert ~stock.index.duplicated().any(), 'Duplicated index after heater replacement'

        self.logger.info('Number of agents that can insulate: {:,.0f}'.format(stock.shape[0]))
        self.logger.info('Calculation insulation replacement')

        # initialize calculate_condition
        supply_insulation = None
        if supply is not None:
            supply_insulation = supply['insulation']

        renovation_rate, market_share = self.insulation_replacement(stock, prices, cost_insulation,
                                                                    frequency_insulation,
                                                                    calib_renovation=calib_renovation,
                                                                    policies_insulation=policies_insulation,
                                                                    financing_cost=financing_cost,
                                                                    exogenous_social=exogenous_social,
                                                                    prices_before=prices_before,
                                                                    supply=supply_insulation,
                                                                    carbon_value=carbon_value_kwh,
                                                                    carbon_content=carbon_content,
                                                                    bill_rebate=bill_rebate,
                                                                    health_cost=health_cost,
                                                                    default_quality=default_quality,
                                                                    credit_constraint=credit_constraint)

        cost_curve_insulation = False
        if cost_curve_insulation:
            self.logger.info('Solving supply demand equilibrium on insulation market')

            def marginal_cost_insulation(cost_insulation, demand, alpha):
                return cost_insulation + alpha * demand

            self.cost_curve_insulation = None
            if self.cost_curve_insulation is None:
                demand = (stock * renovation_rate * market_share.T).T.sum()
                demand = pd.Series({i: demand.xs(True, level=i).sum() / 10 ** 3 for i in demand.index.names})
                x0 = pd.Series([10, 10, 10, 10], index=['Wall', 'Floor', 'Roof', 'Windows'])
                self.cost_curve_insulation = fsolve(lambda x: marginal_cost_insulation(cost_insulation, demand * 2, x) - cost_insulation * 2, x0)

            def supply_demand_insulation_equilibrium(_prices_insulation):
                _renovation_rate, _market_share = self.insulation_replacement(stock, prices, _prices_insulation,
                                                                              policies_insulation=policies_insulation,
                                                                              financing_cost=financing_cost,
                                                                              exogenous_social=exogenous_social,
                                                                              calculate_condition=True)

                _demand = (stock * _renovation_rate * _market_share.T).T.sum()
                _demand = pd.Series({i: _demand.xs(True, level=i).sum() / 10 ** 3 for i in _demand.index.names})

                price_supply = marginal_cost_insulation(cost_insulation, _demand, self.cost_curve_insulation)
                return price_supply - _prices_insulation

            prices_insulation = fsolve(supply_demand_insulation_equilibrium, cost_insulation, xtol=10**-1)
            prices_insulation = pd.Series(prices_insulation, index=cost_insulation.index)
            renovation_rate, market_share = self.insulation_replacement(stock, prices, prices_insulation,
                                                                        calib_renovation=calib_renovation,
                                                                        policies_insulation=policies_insulation,
                                                                        financing_cost=financing_cost,
                                                                        exogenous_social=exogenous_social,
                                                                        prices_before=prices_before,
                                                                        supply=supply['insulation'],
                                                                        carbon_value=carbon_value)

        self.logger.info('Formatting and storing replacement')
        renovation_rate = renovation_rate.reindex(stock.index).fillna(0)
        factor = sum([(1 - renovation_rate) ** i for i in range(step)])
        flow_insulation = renovation_rate * stock * factor

        if step > 1:
            # approximation
            stock = self.heater_replacement(stock_mobile, prices, cost_heater, policies_heater,
                                            calib_heater=calib_heater, step=step, financing_cost=financing_cost,
                                            district_heating=district_heating, supply=supply)
            flow_insulation = flow_insulation.where(flow_insulation < stock, stock)

        flow_only_heater = stock - flow_insulation
        assert (flow_only_heater >= 0).all().all(), 'Remaining stock is not positive'

        flow_only_heater = flow_only_heater.xs(True, level='Heater replacement', drop_level=False).unstack(
            'Heating system final')
        flow_only_heater_sum = flow_only_heater.sum().sum()

        # insulation upgrade
        flow_insulation = flow_insulation[flow_insulation > 0]
        replacement_sum = flow_insulation.sum().sum()
        replaced_by = (flow_insulation * market_share.T).T

        # hidden cost must be calculated now to separate the mandatory renovation flows.
        if 'hidden_cost' in self._renovation_store.keys():
            temp = add_no_renovation(replaced_by)
            no_renovation = (stock * frequency_insulation - flow_insulation)[temp.index]
            temp.iloc[:, 0] = no_renovation.copy()
            self._renovation_store['hidden_cost_agg'] = self._renovation_store['hidden_cost'] * temp
            self._renovation_store['no_renovation'] = no_renovation.sum()

        assert round(replaced_by.sum().sum(), 0) == round(replacement_sum, 0), 'Sum problem'

        # energy performance certificate jump due to heater replacement without insulation upgrade
        only_heater = (stock - flow_insulation.reindex(stock.index, fill_value=0)).xs(True, level='Heater replacement')

        # storing information (flow, investment, subsidies)
        self.logger.debug('Store information retrofit')
        self._replaced_by = replaced_by.copy()
        self._only_heater = only_heater.copy()

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

        self.logger.debug('Formatting switch heater flow')
        flow_only_heater = flow_only_heater.stack('Heating system final')

        to_replace_heater = - flow_only_heater.droplevel('Heating system final')

        replaced_by_heater = flow_only_heater.droplevel('Heating system')
        replaced_by_heater.index = replaced_by_heater.index.rename('Heating system', level='Heating system final')
        replaced_by_heater = replaced_by_heater.reorder_levels(to_replace_heater.index.names)

        flow_only_heater = concat((to_replace_heater, replaced_by_heater), axis=0)
        flow_only_heater = flow_only_heater.groupby(flow_only_heater.index.names).sum()
        assert round(flow_only_heater.sum(), 0) == 0, 'Sum problem'

        self.logger.debug('Formatting renovation flow')
        to_replace = replaced_by.droplevel('Heating system final').sum(axis=1).copy()
        to_replace = to_replace.groupby(to_replace.index.names).sum()
        assert round(to_replace.sum(), 0) == round(replacement_sum, 0), 'Sum problem'

        self.logger.debug('Start frame to flow')

        replaced_by = self.frame_to_flow(replaced_by, default_quality=default_quality)
        self.logger.debug('End frame to flow')

        self.logger.debug('Concatenate all flows')
        to_replace = to_replace.reorder_levels(replaced_by.index.names)
        flow_only_heater = flow_only_heater.reorder_levels(replaced_by.index.names)
        flow_retrofit = concat((-to_replace, replaced_by, flow_only_heater), axis=0)
        flow_retrofit = flow_retrofit.groupby(flow_retrofit.index.names).sum()
        assert round(flow_retrofit.sum(), 0) == 0, 'Sum problem'

        return flow_retrofit

    def flow_obligation(self, policies_insulation, prices, cost_insulation, financing_cost=None,
                        frequency_insulation=1, health_cost=None, default_quality=None):
        """Account for flow obligation if defined in policies_insulation.

        Parameters
        ----------
        policies_insulation: list
            Check if obligation.
        prices: Series
        cost_insulation: Series
        financing_cost: dict, optional
        frequency_insulation: float or int, default 1
        health_cost: Series, optional

        Returns
        -------
        flow_obligation: Series
        """

        list_obligation = [p for p in policies_insulation if p.policy == 'obligation']
        if list_obligation == []:
            return None
        self.logger.info('Calculation flow obligation')

        stock = self.stock_mobile.copy()
        # only work if there is one obligation
        flows_obligation = []
        for obligation in list_obligation:
            self.logger.info('Flow obligation: {}'.format(obligation.name))
            banned_performance = obligation.value.loc[self.year]
            if not isinstance(banned_performance, str):
                return None

            performance_target = [i for i in self._resources_data['index']['Performance'] if i >= banned_performance]

            stock_certificate = self.add_certificate(stock)
            idx = stock.index[stock_certificate.index.get_level_values('Performance').isin(performance_target)]
            if idx.empty:
                return None

            proba = 1
            if obligation.frequency is not None:
                proba = obligation.frequency
                proba = reindex_mi(proba, idx)

            to_replace = stock.loc[idx] * proba
            to_replace = to_replace[to_replace > 0]

            if obligation.target is not None:
                target = reindex_mi(obligation.target, to_replace.index).fillna(False)
                to_replace = to_replace.loc[target]

            # formatting replace_by
            replaced_by = to_replace.copy()
            replaced_by = replaced_by.groupby([i for i in replaced_by.index.names if i != 'Income tenant']).sum()

            if 'Heater replacement' not in replaced_by:
                replaced_by = concat([replaced_by], keys=[False], names=['Heater replacement'])
            if 'Heating system final' not in replaced_by.index.names:
                temp = replaced_by.reset_index('Heating system')
                temp['Heating system final'] = temp['Heating system']
                replaced_by = temp.set_index(['Heating system', 'Heating system final'], append=True).squeeze()

            replaced_by.index = replaced_by.index.reorder_levels(
                ['Heater replacement', 'Existing', 'Occupancy status', 'Income owner', 'Housing type', 'Wall', 'Floor',
                 'Roof', 'Windows', 'Heating system', 'Heating system final'])

            # economic of switch to heat-pumps
            _, market_share = self.insulation_replacement(replaced_by, prices, cost_insulation,
                                                          frequency_insulation,
                                                          policies_insulation=policies_insulation,
                                                          financing_cost=financing_cost,
                                                          min_performance=obligation.min_performance,
                                                          credit_constraint=False,
                                                          health_cost=health_cost)

            if obligation.intensive == 'market_share':
                # market_share endogenously calculated by insulation_replacement
                pass

            elif obligation.intensive == 'global':
                market_share = DataFrame(0, index=replaced_by.index, columns=self._choice_insulation)
                market_share.loc[:, (True, True, True, True)] = 1
            elif obligation.intensive == 'cost':
                print('ok')
            else:
                raise NotImplemented

            assert ~market_share.isna().all(axis=1).any(), "Market-share issue"
            assert (market_share.sum(axis=1).round(5) == 1.0).all(), "Market-share sum issue"

            replaced_by = (replaced_by.rename(None) * market_share.T).T
            replaced_by = replaced_by.fillna(0)

            if 'hidden_cost' in self._renovation_store.keys():
                temp = (self._renovation_store['hidden_cost'] * replaced_by).loc[replaced_by.index, :]
                temp = temp.reindex(self._renovation_store['hidden_cost_agg'].index).fillna(0)
                temp = temp.reindex(self._renovation_store['hidden_cost_agg'].columns, axis=1).fillna(0)
                self._renovation_store['hidden_cost_agg'] += temp

            self._replaced_by = self._replaced_by.add(replaced_by.copy(), fill_value=0)
            replaced_by = self.frame_to_flow(replaced_by, default_quality=default_quality)

            assert to_replace.sum().round(0) == replaced_by.sum().round(0), 'Sum problem'

            self._flow_obligation.update({obligation.name.replace('_', ' ').capitalize(): to_replace.sum()})

            flow_obligation = concat((- to_replace, replaced_by), axis=0)
            flow_obligation = flow_obligation.groupby(flow_obligation.index.names).sum()
            flows_obligation.append(flow_obligation)
        return flows_obligation

    def parse_output_run(self, prices, inputs, climate=None, step=1, taxes=None,
                         lifetime_insulation=30, social_discount_rate=0.032, bill_rebate=0,
                         coefficient_public_funding=0.2, duration_financing=10, discount_financing=0.05):
        """Parse output.

        Renovation : envelope
        Retrofit : envelope and/or heating system

        Parameters
        ----------
        prices: Series
            Energy prices
        inputs: dict
            Exogenous data for post-treatment.
        climate: int, optional
        step: int, optional
        taxes: list, optional
        lifetime_insulation: int, optional
        social_discount_rate: float, default 0.032
        bill_rebate: float, optional
        coefficient_public_funding: float, default 0.2
        duration_financing: int, default 10
        discount_financing: float, default 0.05

        Returns
        -------
        dict
        """
        import gc

        carbon_value = inputs['carbon_value'].loc[self.year]

        prices_wt = prices.copy()
        if taxes is not None:
            tax = sum([tax.value.loc[self.year, :] for tax in taxes if self.year in tax.value.index]).fillna(0)
            prices_wt -= tax

        stock = self.simplified_stock()

        output = dict()
        output['Stock (Million)'] = stock.sum() / 10 ** 6
        output['Stock existing (Million)'] = self.stock.xs(True, level='Existing').sum() / 10 ** 6
        stock_new = 0
        if False in self.stock.index.get_level_values('Existing'):
            stock_new = self.stock.xs(False, level='Existing').sum() / 10 ** 6
        output['Stock new (Million)'] = stock_new

        surface = self.stock * self.surface
        output['Surface (Million m2)'] = surface.sum() / 10 ** 6
        output['Surface existing (Million m2)'] = surface.xs(True, level='Existing').sum() / 10 ** 6
        surface_new = 0
        if False in self.stock.index.get_level_values('Existing'):
            surface_new = surface.xs(False, level='Existing').sum() / 10 ** 6
        output['Surface new (Million m2)'] = surface_new

        surface = surface.groupby('Heating system').sum()
        output.update({'Surface {} (Million m2)'.format(i): surface.loc[i] / 10 ** 6 for i in surface.index})

        if 'population' in inputs.keys():
            output['Surface (m2/person)'] = (
                    output['Surface (Million m2)'] / (inputs['population'].loc[self.year] / 10 ** 6))

        temp = prices.copy().T
        temp.index = temp.index.map(lambda x: 'Prices {} (euro/kWh)'.format(x))
        output.update(temp.T)

        output['Consumption standard (TWh)'] = self.consumption_agg(prices=prices, freq='year', climate=climate,
                                                                    standard=True, agg='all')
        output['Consumption standard (kWh/m2)'] = (output['Consumption standard (TWh)'] * 10 ** 9) / (
                output['Surface (Million m2)'] * 10 ** 6)

        consumption_energy = self.consumption_agg(prices=prices, freq='year', climate=None, standard=False,
                                                  agg='energy', bill_rebate=bill_rebate)
        output['Consumption (TWh)'] = consumption_energy.sum()
        self.store_over_years[self.year].update({'Consumption (TWh)': output['Consumption (TWh)']})
        output['Consumption (kWh/m2)'] = (output['Consumption (TWh)'] * 10 ** 9) / (
                output['Surface (Million m2)'] * 10 ** 6)

        output['Consumption existing (TWh)'] = self.consumption_agg(prices=prices, freq='year', existing=True,
                                                                    agg='all', bill_rebate=bill_rebate)
        output['Consumption new (TWh)'] = output['Consumption (TWh)'] - output['Consumption existing (TWh)']
        output['Consumption existing (kWh/m2)'] = (output['Consumption existing (TWh)'] * 10 ** 9) / (
                output['Surface existing (Million m2)'] * 10 ** 6)

        output['Consumption PE (TWh)'] = thermal.final2primary(consumption_energy, Series(consumption_energy.index,
                                                                                          index=consumption_energy.index)).sum()

        if surface_new > 0:
            output['Consumption new (kWh/m2)'] = (output['Consumption new (TWh)'] * 10 ** 9) / (
                    output['Surface new (Million m2)'] * 10 ** 6)

        heating_intensity, budget_share = self.to_heating_intensity(self.stock.index, prices, bill_rebate=bill_rebate,
                                                                    full_output=True)

        condition_poverty = self.stock.index.get_level_values('Income tenant').isin(['D1', 'D2', 'D3', 'C1', 'C2']) & (
                    budget_share >= 0.08)
        energy_poverty = self.stock[condition_poverty].sum()

        output['Heating intensity (%)'] = (self.stock * heating_intensity).sum() / self.stock.sum()
        output['Energy poverty (Million)'] = energy_poverty / 10 ** 6

        if self.year == self.first_year:
            # marginal utility of money is not considered
            output['Space heating utility (Billion euro)'] = 0
        else:
            value = self.to_thermal_comfort(heating_intensity) / abs(self.preferences_insulation['cost']) * 1000
            output['Space heating utility (Billion euro)'] = (reindex_mi(value, self.stock.index) * self.stock).sum() / 1e9

            """
            x = self.to_thermal_comfort(heating_intensity) / abs(self.preferences_insulation['cost']) * 1000

            y = concat((heating_intensity, x), axis=1, keys=['Heating intensity', 'Thermal comfort'])
            # scatter plot heating intensity
            make_scatter_plot(y, 'Heating intensity', 'Thermal comfort', 'Heating intensity (%)',
                              'Thermal comfort (euro)',
                              annotate=False,
                              save=os.path.join(self.path_calibration, 'thermal_comfort.png'),
                              format_y=lambda y, _: '{:.0f}'.format(y),
                              format_x=lambda x, _: '{:.0%}'.format(x),
                              s=10)"""

        temp = consumption_energy.copy()
        temp.index = temp.index.map(lambda x: 'Consumption {} (TWh)'.format(x))
        output.update(temp.T)

        temp = self.consumption_agg(prices=prices, freq='year', climate=None, standard=False,
                                    agg='heater', bill_rebate=bill_rebate).dropna()
        consumption_hp = sum([temp.loc[i] for i in self._resources_data['index']['Heat pumps'] if i in temp.index])
        temp.index = temp.index.map(lambda x: 'Consumption {} (TWh)'.format(x))
        output.update(temp.T)

        output.update({'Consumption Heat pump (TWh)': consumption_hp})
        output.update({'Consumption Direct electric (TWh)': output['Consumption Electricity-Direct electric (TWh)']})
        output.update({'Consumption District heating (TWh)': output['Consumption Heating (TWh)']})

        consumption_energy_climate = None
        if climate is not None:
            consumption_energy_climate = self.consumption_agg(prices=prices, freq='year', climate=climate,
                                                              standard=False, agg='energy', bill_rebate=bill_rebate)
            output['Consumption climate (TWh)'] = consumption_energy_climate.sum()
            temp = consumption_energy_climate.copy()
            temp.index = temp.index.map(lambda x: 'Consumption {} climate (TWh)'.format(x))
            output.update(temp.T)
            output['Factor climate (%)'] = output['Consumption climate (TWh)'] / output['Consumption (TWh)']

        # hourly resolution energy consumption
        if False:
            consumption_hourly = self.consumption_agg(prices=prices, freq='hour', standard=False, climate=2006,
                                                      efficiency_hour=True, hourly_profile='power')

            # format_x datetime hourly
            temp = consumption_hourly.loc['Electricity']
            make_plot(temp, 'Consumption electricity (GWh/h)',
                      save=os.path.join(self.path, 'consumption_hour.png'),
                      format_y=lambda y, _: '{:.0f}'.format(y / 1e6), integer=False, legend=False)

            # only select a day in february

            temp = consumption_hourly.loc['Electricity'].loc['2006-02-01']
            make_plot(temp, 'Consumption electricity (GWh/h)',
                      save=os.path.join(self.path, 'consumption_day.png'),
                      format_y=lambda y, _: '{:.0f}'.format(y / 1e6), integer=False, legend=False)

        consumption = self.consumption_actual(prices, bill_rebate=bill_rebate) * self.stock
        coefficient_backup = reindex_mi(self.coefficient_backup, consumption.index)

        # correct that do consider secondary heating system
        temp = consumption.groupby('Existing').sum()
        temp.rename(index={True: 'Existing', False: 'New'}, inplace=True)
        temp.index = temp.index.map(lambda x: 'Consumption {} (TWh)'.format(x))
        output.update(temp.T / 10 ** 9)

        temp = consumption.groupby(self.certificate).sum()
        temp.index = temp.index.map(lambda x: 'Consumption {} (TWh)'.format(x))
        output.update(temp.T / 10 ** 9)

        temp = self.consumption_agg(agg='heater', standard=True, freq='year')
        temp.index = temp.index.map(lambda x: 'Consumption standard {} (TWh)'.format(x))
        output.update(temp.T)

        emission = inputs['carbon_emission'].loc[self.year:, :].copy()
        if inputs.get('renewable_gas') is not None:
            renewable_gas = inputs['renewable_gas'].loc[self.year]
            if renewable_gas < consumption_energy.loc['Natural gas']:
                temp = (consumption_energy.loc['Natural gas'] - renewable_gas) / consumption_energy.loc['Natural gas'] * emission.loc[self.year, 'Natural gas']
            else:
                temp = 0
            emission.loc[self.year, 'Natural gas'] = temp
        self.store_over_years[self.year].update({'Emission content (gCO2/kWh)': emission.loc[self.year, :]})

        temp = emission.loc[self.year, :]
        temp.index = temp.index.map(lambda x: 'Emission content {} (gCO2/kWh)'.format(x))
        output.update(temp.T)

        emission = emission.loc[self.year, :]
        carbon_value_kwh = carbon_value * emission / 10**6

        output['Emission mean (gCO2/kWh)'] = (consumption_energy * emission).sum() / consumption_energy.sum()
        temp = consumption_energy * emission
        output['Emission (MtCO2)'] = temp.sum() / 10 ** 3
        self.store_over_years[self.year].update({'Emission (MtCO2)': output['Emission (MtCO2)']})

        temp.index = temp.index.map(lambda x: 'Emission {} (MtCO2)'.format(x))
        output.update(temp.T / 10 ** 3)

        emission_start = inputs['carbon_emission'].loc[self.first_year, :]
        output['Emission mean first year (gCO2/kWh)'] = (consumption_energy * emission_start).sum() / consumption_energy.sum()
        output['Emission first year (MtCO2)'] = (consumption_energy * emission_start).sum() / 10 ** 3

        if consumption_energy_climate is not None:
            temp = consumption_energy_climate * emission
            output['Emission climate (MtCO2)'] = temp.sum() / 10 ** 3
            temp.index = temp.index.map(lambda x: 'Emission climate {} (MtCO2)'.format(x))
            output.update(temp.T / 10 ** 3)

        emission_reindex = emission.reindex(self.energy).set_axis(self.stock.index, axis=0)
        emission_stock = consumption * coefficient_backup * emission_reindex
        emission_stock += consumption * (1 - coefficient_backup) * emission.loc['Wood fuel']

        temp = emission_stock.groupby('Existing').sum()
        temp.rename(index={True: 'Existing', False: 'New'}, inplace=True)
        temp.index = temp.index.map(lambda x: 'Emission {} (MtCO2)'.format(x))
        output.update(temp.T / 10 ** 12)

        temp = self.stock.groupby(self.certificate).sum()
        temp.index = temp.index.map(lambda x: 'Stock {} (Million)'.format(x))
        output.update(temp.T / 10 ** 6)

        output['Stock efficient (Million)'] = output.get('Stock A (Million)', 0) + output.get('Stock B (Million)', 0)
        output['Stock low-efficient (Million)'] = output.get('Stock G (Million)', 0) + output.get('Stock F (Million)', 0)
        output['Stock to renovate (Million)'] = output['Stock low-efficient (Million)'] + output.get('Stock E (Million)', 0) + \
                                                output.get('Stock D (Million)', 0)

        temp = self.stock.groupby('Heating system').sum()

        output['Stock Direct electric (Million)'] = temp['Electricity-Direct electric'] / 10 ** 6
        output['Stock Heat pump (Million)'] = temp['Electricity-Heat pump water'] / 10 ** 6
        if 'Electricity-Heat pump air' in temp.keys():
            output['Stock Heat pump (Million)'] += temp['Electricity-Heat pump air'] / 10 ** 6

        heater = {
            'Oil fuel': ['Oil fuel-Performance boiler', 'Oil fuel-Standard boiler', 'Oil fuel-Collective boiler'],
            'Wood fuel': ['Wood fuel-Performance boiler', 'Wood fuel-Standard boiler'],
            'Natural gas': ['Natural gas-Performance boiler', 'Natural gas-Standard boiler',
                            'Natural gas-Collective boiler'],
            'District heating': ['Heating-District heating']
        }
        for key, item in heater.items():
            output['Stock {} (Million)'.format(key)] = temp[[i for i in item if i in temp.index]].sum() / 10 ** 6

        temp.index = temp.index.map(lambda x: 'Stock {} (Million)'.format(x))
        output.update(temp.T / 10 ** 6)

        # energy expenditures considering back-up cost
        prices_reindex = prices.reindex(self.energy).set_axis(self.stock.index, axis=0)
        energy_expenditure = consumption * coefficient_backup * prices_reindex
        energy_expenditure += consumption * (1 - coefficient_backup) * prices.loc['Wood fuel']
        temp = energy_expenditure.groupby('Income tenant').sum()
        temp.index = temp.index.map(lambda x: 'Energy expenditures {} (Billion euro)'.format(x))
        output.update(temp.T / 10 ** 9)
        output['Energy expenditures (Billion euro)'] = energy_expenditure.sum() / 10 ** 9

        prices_wt_reindex = prices_wt.reindex(self.energy).set_axis(self.stock.index, axis=0)
        energy_expenditure = consumption * coefficient_backup * prices_wt_reindex
        energy_expenditure += consumption * (1 - coefficient_backup) * prices.loc['Wood fuel']
        output['Energy expenditures wt (Billion euro)'] = energy_expenditure.sum() / 10 ** 9

        # taxes expenditures
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
            self.taxes_revenues.update({self.year: taxes_expenditures.copy()})
            taxes_expenditures.index = taxes_expenditures.index.map(
                lambda x: '{} (Billion euro)'.format(x.capitalize().replace('_', ' ').replace('Cee', 'Cee tax')))
            output.update((taxes_expenditures / step).T)
            output['Taxes expenditure (Billion euro)'] = taxes_expenditures.sum() / step

        output['Carbon value (Billion euro)'] = (consumption_energy * carbon_value_kwh).sum()
        output['Health cost (Billion euro)'] = self.health_cost(inputs['health_cost_dpe'], inputs['health_cost_income'], prices)
        output['Health expenditure (Billion euro)'] = 0 # temp['Health expenditure (Billion euro)']

        self.store_over_years[self.year].update({'Health cost (Billion euro)': output['Health cost (Billion euro)']})
        # output.update(o)

        if self.year > self.first_year:
            levels = [i for i in self._replaced_by.index.names if
                      i not in ['Heater replacement', 'Heating system final']]

            names = ['Heater replacement', 'Existing', 'Occupancy status', 'Income owner', 'Housing type', 'Wall',
                     'Floor', 'Roof', 'Windows', 'Heating system', 'Heating system final']
            self._replaced_by.index = self._replaced_by.index.reorder_levels(names)
            self._renovation_store['discount'] = self._renovation_store['discount'].groupby(levels).mean()

            levels_owner = ['Occupancy status', 'Income owner']

            replaced_by_grouped = self._replaced_by.groupby(levels).sum()

            # cost of epc transition
            if False:
                replacement = self._replaced_by.copy().groupby(['Housing type', 'Wall', 'Floor', 'Roof', 'Windows', 'Heating system']).sum()
                cost = (self._renovation_store['cost_households'].T / self._surface).T
                cost = cost.xs(False, level='Existing').xs('Owner-occupied', level='Occupancy status')
                cost = reindex_mi(cost, replacement.index)

                consumption_before, certificate_before, _ = self.consumption_heating(index=replacement.index, method='3uses',
                                                                    full_output=True)

                s = concat([Series(index=replacement.index, dtype=float)] * len(replacement.columns), axis=1).set_axis(replacement.columns, axis=1)
                # choice_insulation = choice_insulation.drop(no_insulation) # only for
                s.index.rename(
                    {'Wall': 'Wall before', 'Floor': 'Floor before', 'Roof': 'Roof before', 'Windows': 'Windows before'},
                    inplace=True)
                temp = s.fillna(0).stack(s.columns.names)
                temp = temp.reset_index().drop(0, axis=1)
                for i in ['Wall', 'Floor', 'Roof', 'Windows']:
                    # keep the info to unstack later
                    temp.loc[:, '{} bool'.format(i)] = temp.loc[:, i]
                    temp.loc[temp[i], i] = self._performance_insulation_renovation[i]
                    temp.loc[temp[i] == False, i] = temp.loc[temp[i] == False, '{} before'.format(i)]
                temp = temp.astype(
                    {'Housing type': 'string', 'Wall': 'float', 'Floor': 'float', 'Roof': 'float', 'Windows': 'float',
                     'Heating system': 'string'})
                index = MultiIndex.from_frame(temp)
                consumption_after, certificate_after, _ = self.consumption_heating(index=index, method='3uses', full_output=True)

                certificate_after = reindex_mi(certificate_after, index).droplevel(['Wall', 'Floor', 'Roof', 'Windows']).unstack(
                    ['{} bool'.format(i) for i in ['Wall', 'Floor', 'Roof', 'Windows']])
                certificate_after.index.rename({'Wall before': 'Wall', 'Floor before': 'Floor', 'Roof before': 'Roof',
                                                'Windows before': 'Windows'}, inplace=True)
                certificate_after.columns.rename(
                    {'Wall bool': 'Wall', 'Floor bool': 'Floor', 'Roof bool': 'Roof', 'Windows bool': 'Windows'},
                    inplace=True)
                consumption_after = reindex_mi(consumption_after, index).droplevel(['Wall', 'Floor', 'Roof', 'Windows']).unstack(
                    ['{} bool'.format(i) for i in ['Wall', 'Floor', 'Roof', 'Windows']])
                consumption_after.index.rename({'Wall before': 'Wall', 'Floor before': 'Floor', 'Roof before': 'Roof',
                                                'Windows before': 'Windows'}, inplace=True)
                consumption_after.columns.rename(
                    {'Wall bool': 'Wall', 'Floor bool': 'Floor', 'Roof bool': 'Roof', 'Windows bool': 'Windows'},
                    inplace=True)

                # calculate the cost of the transition
                rename_insulation = {'Wall': 'Wall insulation', 'Floor': 'Floor insulation', 'Roof': 'Roof insulation',
                                     'Windows': 'Windows insulation'}
                replacement.columns.rename(rename_insulation, inplace=True)
                replacement = replacement.stack(replacement.columns.names)

                replacement = concat((replacement, reindex_mi(certificate_before, replacement.index)), axis=1,
                                     keys=['replacement', 'certificate_before'])
                replacement = concat(
                    (replacement, reindex_mi(consumption_before, replacement.index).rename('consumption_before')), axis=1)

                cost.columns.rename(rename_insulation, inplace=True)
                cost = cost.stack(cost.columns.names)
                replacement = concat((replacement, cost.rename('Cost')), axis=1)

                consumption_after.columns.rename(rename_insulation, inplace=True)
                consumption_after = consumption_after.stack(consumption_after.columns.names)
                replacement = concat((replacement, consumption_after.rename('consumption_after')), axis=1)

                certificate_after.columns.rename(rename_insulation, inplace=True)
                certificate_after = certificate_after.stack(certificate_after.columns.names)
                replacement = concat((replacement, certificate_after.rename('certificate_after')), axis=1)

                replacement.set_index(['certificate_before', 'certificate_after'], append=True, inplace=True)
                replacement['consumption_saving'] = replacement['consumption_before'] - replacement['consumption_after']

                # weighted cost of the transition groupby certificate_before and certificate_after
                cost_transition = replacement.groupby(['certificate_before', 'certificate_after']).apply(
                    lambda x: (x['replacement'] * x['Cost']).sum() / x['replacement'].sum())
                cost_transition = cost_transition.unstack('certificate_after')
                replacement_transition = replacement.groupby(['certificate_before', 'certificate_after'])['replacement'].sum()
                replacement_transition = replacement_transition.unstack('certificate_after')

                if self.path_ini:
                    replacement.to_csv(os.path.join(self.path_ini, 'replacement.csv'))
                    cost_transition.to_csv(os.path.join(self.path_ini, 'cost_transition.csv'))
                    replacement_transition.to_csv(os.path.join(self.path_ini, 'replacement_transition.csv'))

            # consumption saving
            if self.consumption_before_retrofit is not None:
                consumption_before_retrofit = self.consumption_before_retrofit
                consumption_after_retrofit = self.store_consumption(prices, emission, bill_rebate=bill_rebate)
                temp = {'{} saving (TWh/year)'.format(k.split(' (TWh)')[0]): consumption_before_retrofit[k] -
                                                                             consumption_after_retrofit[k]
                        for k in consumption_before_retrofit.keys() if 'TWh' in k}
                output.update(temp)
                temp = {'{} saving (MtCO2/year)'.format(k.split(' (MtCO2)')[0]): consumption_before_retrofit[k] -
                                                                             consumption_after_retrofit[k]
                        for k in consumption_before_retrofit.keys() if 'MtCO2' in k}
                output.update(temp)

            consumption_saved_insulation = (
                        self._replaced_by * self._renovation_store['consumption_saved_households']).groupby(
                levels).sum()

            temp = self.add_level(self._replaced_by.fillna(0), self._stock_ref, 'Income tenant')
            consumption_saved_actual_insulation = (
                        temp * reindex_mi(self._renovation_store['consumption_saved_actual_households'],
                                          temp.index)).groupby(levels + ['Income tenant']).sum()
            consumption_saved_no_rebound_insulation = (
                        temp * reindex_mi(self._renovation_store['consumption_saved_no_rebound_households'],
                                          temp.index)).groupby(levels + ['Income tenant']).sum()
            rebound_insulation = (consumption_saved_no_rebound_insulation - consumption_saved_actual_insulation).sum(
                axis=1)
            rebound_insulation = rebound_insulation.groupby(self.to_energy(rebound_insulation)).sum()

            output.update({'Consumption standard saving insulation (TWh/year)': consumption_saved_insulation.sum().sum() / 10 ** 9})
            _consumption_saved_actual_insulation = self.apply_calibration(consumption_saved_actual_insulation.sum(axis=1))
            output.update({'Consumption saving insulation (TWh/year)': _consumption_saved_actual_insulation.sum() / 10 ** 9})
            _consumption_saved_no_rebound_insulation = self.apply_calibration(consumption_saved_no_rebound_insulation.sum(axis=1))
            output.update({'Consumption saving no rebound insulation (TWh/year)': _consumption_saved_no_rebound_insulation.sum() / 10 ** 9})
            output.update({'Rebound insulation (TWh/year)': rebound_insulation.sum() / 10 ** 9})

            output['Realization rate (% standard)'] = output['Consumption saving insulation (TWh/year)'] / output[
                'Consumption standard saving insulation (TWh/year)']
            output['Rebound insulation (% performance gap)'] =  output['Rebound insulation (TWh/year)'] /  (output[
                'Consumption standard saving insulation (TWh/year)'] - output['Consumption saving insulation (TWh/year)'])

            temp = consumption_saved_insulation.sum(axis=1)
            output.update(
                {'Emission standard saving insulation (MtCO2/year)': self.to_emission(temp, emission).sum() / 10 ** 12})

            temp = _consumption_saved_actual_insulation.copy()
            output.update(
                {'Emission saving insulation (MtCO2/year)': self.to_emission(temp, emission).sum() / 10 ** 12})

            consumption = self.consumption_heating_store(self._renovation_store['consumption_saved_households'].index,
                                                         full_output=False, level_heater='Heating system final')

            consumption = reindex_mi(consumption, self._renovation_store['consumption_saved_households'].index)
            consumption *= reindex_mi(self._surface, consumption.index)
            consumption_saved = (self._renovation_store['consumption_saved_households'].T / consumption).T
            assert (consumption_saved <= 1).all().all(), 'Percent issue'
            consumption_saved_mean = (consumption_saved * self._replaced_by).sum().sum() / self._replaced_by.sum().sum()
            output.update({'Consumption standard saving insulation (%)': consumption_saved_mean})

            consumption_saved_decision = (consumption_saved * self._replaced_by).sum(
                axis=1).groupby(['Housing type', 'Occupancy status']).sum() / self._replaced_by.sum(axis=1).groupby(
                ['Housing type', 'Occupancy status']).sum()
            temp = consumption_saved_decision
            temp.index = temp.index.map(lambda x: 'Consumption standard saving {} - {} (%)'.format(x[0], x[1]))
            output.update(temp.T)

            output.update(
                {'Consumption standard saving heater (TWh/year)': self._heater_store['consumption_saved'] / 10 ** 9})
            output.update({'Consumption saving heater (TWh/year)': self._heater_store[
                                                                       'consumption_saved_actual'].sum() / 10 ** 9})
            output.update({'Consumption saving no rebound heater (TWh/year)': self._heater_store[
                                                                                  'consumption_saved_no_rebound'].sum() / 10 ** 9})

            output.update(
                {'Emission saving heater (MtCO2/year)': self.to_emission(self._heater_store['consumption_saved_actual'], emission).sum() / 10 ** 12})

            rebound_heater = self._heater_store['rebound']
            output['Rebound heater (TWh/year)'] = rebound_heater.sum() / 10 ** 9

            rebound_ee = rebound_insulation + rebound_heater
            output['Rebound EE (TWh/year)'] = rebound_ee.sum() / 10 ** 9

            thermal_comfort = rebound_ee * prices
            rebound_insulation.index = rebound_insulation.index.map(
                lambda x: 'Rebound insulation {} (TWh/year)'.format(x))
            output.update(rebound_insulation.T / 10 ** 9)
            thermal_comfort.index = thermal_comfort.index.map(
                lambda x: 'Thermal comfort {} (Billion euro/year)'.format(x))
            output.update({'Thermal comfort EE (Billion euro)': thermal_comfort.sum() / 10 ** 9})

            output.update(thermal_comfort.T / 10 ** 9)

            rebound_heater.index = rebound_heater.index.map(lambda x: 'Rebound heater {} (TWh/year)'.format(x))
            output.update(rebound_heater.T / 10 ** 9)

            consumption_saved_price_constant = _consumption_saved_actual_insulation + self._heater_store['consumption_saved_actual']
            output['Consumption saving prices constant (TWh/year)'] = consumption_saved_price_constant.sum() / 10 ** 9
            output['Consumption saving prices effect (TWh/year)'] = round(
                output['Consumption saving (TWh/year)'] - output['Consumption saving prices constant (TWh/year)'], 3)

            if self.year - 1 in self.store_over_years.keys():
                consumption_saving = self.store_over_years[self.year - 1]['Consumption (TWh)'] - output['Consumption (TWh)']
                output['Consumption saving total (TWh/year)'] = consumption_saving
                output['Consumption saving total (%/year)'] = consumption_saving / self.store_over_years[self.year - 1]['Consumption (TWh)']
                output['Consumption saving natural replacement (TWh/year)'] = consumption_saving - output['Consumption saving (TWh/year)']

                emission_saving = self.store_over_years[self.year - 1]['Emission (MtCO2)'] - output['Emission (MtCO2)']
                output['Emission saving total (TWh/year)'] = emission_saving
                output['Emission saving total (%/year)'] = emission_saving / self.store_over_years[self.year - 1]['Emission (MtCO2)']
                output['Emission saving natural replacement (MtCO2/year)'] = emission_saving - output['Emission saving (MtCO2/year)']

            consumption_saving_prices = Series({i: (output['Consumption {} saving (TWh/year)'.format(i)] * 10**9 - consumption_saved_price_constant.loc[i]) for i in consumption_saved_price_constant.keys()})
            output['Emission saving prices (MtCO2/year)'] = (consumption_saving_prices * emission).sum() / 10**12
            output['Emission saving carbon content (MtCO2/year)'] = output['Emission saving (MtCO2/year)'] - output['Emission saving prices (MtCO2/year)'] - output['Emission saving heater (MtCO2/year)'] - output['Emission saving insulation (MtCO2/year)']

            prices_effect = pd.Series({i: output['Consumption {} saving (TWh/year)'.format(i)] -
                                          consumption_saved_price_constant.loc[i] / 10 ** 9 for i in
                                       consumption_saved_price_constant.index})
            thermal_loss = prices_effect * prices
            output.update({'Thermal loss prices (Billion euro)': round(thermal_loss.sum(), 3)})

            consumption_saving_prices = prices_effect * prices_wt
            output.update({'Consumption saving prices (Billion euro)': round(consumption_saving_prices.sum(), 3)})

            # retrofit and renovation
            renovation = replaced_by_grouped.sum().sum()
            output['Retrofit (Thousand households)'] = (renovation + self._only_heater.sum()) / 10 ** 3 / step
            output['Renovation (Thousand households)'] = renovation / 10 ** 3 / step

            output['Renovation obligation (Thousand households)'] = 0
            if self._flow_obligation:
                temp = pd.Series(self._flow_obligation)
                temp.index = temp.index.map(lambda x: 'Renovation {} (Thousand households)'.format(x))
                output.update(temp.T / 10 ** 3)
                output['Renovation obligation (Thousand households)'] = temp.sum() / 10**3

            output['Renovation endogenous (Thousand households)'] = output['Renovation (Thousand households)'] - output['Renovation obligation (Thousand households)']

            if True in self._replaced_by.index.get_level_values('Heater replacement'):
                temp = self._replaced_by.xs(True, level='Heater replacement').sum().sum()
            output['Renovation with heater replacement (Thousand households)'] = temp / 10 ** 3 / step
            output['Switch heater only (Thousand households)'] = self._only_heater.sum() / 10 ** 3 / step

            temp = {i: 0 for i in range(1, 6)}
            for n, g in NB_MEASURES.items():
                if False in self._replaced_by.index.get_level_values('Heater replacement'):
                    temp[n] += self._replaced_by.loc[:, g].xs(False, level='Heater replacement').sum().sum()
                temp[n + 1] += self._replaced_by.loc[:, g].xs(True, level='Heater replacement').sum().sum()
            if self._only_heater is not None:
                temp[1] += self._only_heater.sum()
            nb_measures = Series(temp)

            output['Replacement total (Thousand)'] = (nb_measures * nb_measures.index).sum() / 10 ** 3 / step
            output['Replacement total (Thousand renovating)'] = output['Replacement total (Thousand)'] - output[
                'Switch heater only (Thousand households)']
            nb_measures.index = nb_measures.index.map(lambda x: 'Retrofit measures {} (Thousand households)'.format(x))
            output.update(nb_measures.T / 10 ** 3)

            heater_no_carbon = ['Electricity-Heat pump water', 'Electricity-Heat pump air',
                                'Wood fuel-Performance boiler', 'Wood fuel-Standard boiler']

            def condition_decarbonizing(x):
                return x.index.get_level_values('Heating system final').isin(heater_no_carbon) & ~x.index.get_level_values('Heating system').isin(heater_no_carbon)

            output['Switch decarbonize (Thousand households)'] = self._only_heater[condition_decarbonizing(self._only_heater)].sum() / 10**3
            temp = self._replaced_by.sum(axis=1)
            condition = condition_decarbonizing(temp)
            output['Insulation (Thousand households)'] = temp[~condition].sum() / 10**3
            output['Insulation and switch decarbonize (Thousand households)'] = temp[condition].sum() / 10**3
            output['Decarbonize measures (Thousand households)'] = output['Switch decarbonize (Thousand households)'] + output['Insulation (Thousand households)'] + output['Insulation and switch decarbonize (Thousand households)']

            temp = self._replaced_by.groupby(['Housing type', 'Occupancy status']).sum()
            temp = (temp.sum(axis=1) / self._stock_ref.groupby(temp.index.names).sum()).dropna()
            temp.index = temp.index.map(lambda x: 'Rate {} - {} (%)'.format(x[0], x[1]))
            output.update(temp.T)

            temp = self._replaced_by.groupby(['Occupancy status', 'Housing type', 'Income owner']).sum()
            temp = (temp.sum(axis=1) / self._stock_ref.groupby(temp.index.names).sum()).dropna()
            temp = temp.xs('Owner-occupied', level='Occupancy status').xs('Single-family', level='Housing type')
            temp.index = temp.index.map(lambda x: 'Rate Single-family - Owner-occupied {} (%)'.format(x))
            output.update(temp.T)

            _, _, certificate = self.consumption_heating_store(self._stock_ref.index)
            temp = concat((self._replaced_by, reindex_mi(certificate.rename('Performance'), self._replaced_by.index)), axis=1)
            temp = temp.set_index('Performance', append=True).set_axis(self._replaced_by.columns, axis=1)
            s = concat((self._stock_ref, reindex_mi(certificate.rename('Performance'), self._stock_ref.index)), axis=1)
            s = s.set_index('Performance', append=True).squeeze()

            temp = temp.groupby(['Occupancy status', 'Housing type', 'Performance']).sum()
            temp = (temp.sum(axis=1) / s.groupby(temp.index.names).sum()).dropna()
            temp = temp.xs('Owner-occupied', level='Occupancy status').xs('Single-family', level='Housing type')
            temp.index = temp.index.map(lambda x: 'Rate Single-family - Owner-occupied {} (%)'.format(x))
            output.update(temp.T)

            epc_upgrade = self._heater_store['epc_upgrade'].stack()
            temp = {}
            for i in unique(epc_upgrade):
                temp.update({i: ((epc_upgrade == i) * self._only_heater).sum()})
            self._heater_store['epc_upgrade'] = Series(temp).sort_index()

            if self._condition_store is not None:
                epc_upgrade_all = self._condition_store['epc_upgrade_all']
                temp = {}
                for i in unique(epc_upgrade_all.values.ravel('K')):
                    temp.update({i: ((epc_upgrade_all == i) * self._replaced_by).sum(axis=1)})
                epc_upgrade_all = DataFrame(temp).groupby(levels).sum()

                temp = epc_upgrade_all.sum().squeeze().sort_index()
                temp = temp[temp.index.dropna()]
                o = {}
                for i in temp.index.union(self._heater_store['epc_upgrade'].index):
                    if i > 0:
                        t_renovation = 0
                        if i in temp.index:
                            t_renovation = temp.loc[i]
                            # o['Renovation {} EPC (Thousand households)'.format(i)] = t_renovation / 10 ** 3
                        t_heater = 0
                        if i in self._heater_store['epc_upgrade'].index:
                            t_heater = self._heater_store['epc_upgrade'].loc[i]
                        o['Retrofit {} EPC (Thousand households)'.format(i)] = (t_renovation + t_heater) / 10 ** 3 / step
                o = Series(o).sort_index(ascending=False)

                output['Retrofit at least 1 EPC (Thousand households)'] = sum(
                    [o['Retrofit {} EPC (Thousand households)'.format(i)] for i in temp.index.unique() if i >= 1])
                output['Retrofit at least 2 EPC (Thousand households)'] = sum(
                    [o['Retrofit {} EPC (Thousand households)'.format(i)] for i in temp.index.unique() if i >= 2])
                output.update(o.T)

                for condition in [c for c in self._condition_store.keys() if c not in self._list_condition_subsidies]:
                    temp = (self._replaced_by * self._condition_store[condition]).sum().sum()
                    output[
                        '{} (Thousand households)'.format(condition.capitalize().replace('_', ' '))] = temp / 10 ** 3 / step

            # switch heater
            temp = self._heater_store['replacement'].sum()
            output['Switch heater (Thousand households)'] = temp.sum() / 10 ** 3 / step
            t = self._heater_store['flow_premature_replacement']
            if isinstance(t, Series):
                t = t.sum()

            output['Switch premature heater (Thousand households)'] = t / 10 ** 3 / step

            output['Switch Heat pump (Thousand households)'] = temp[[i for i in self._resources_data['index'][
                'Heat pumps'] if i in temp.index]].sum() / 10 ** 3 / step
            temp.index = temp.index.map(lambda x: 'Switch {} (Thousand households)'.format(x))
            output.update((temp / 10 ** 3 / step).T)
            # capture heating system transition
            temp = self._heater_store['replacement'].stack('Heating system final').squeeze()
            temp = temp.reset_index(['Heating system', 'Heating system final'])
            temp['Heating system'] = temp['Heating system'].replace(self._resources_data['heating2heater'])
            temp['Heating system final'] = temp['Heating system final'].replace(self._resources_data['heating2heater'])
            temp = temp.set_index(['Heating system', 'Heating system final'], append=True).squeeze()
            temp = temp.groupby(['Heating system', 'Heating system final']).sum()
            temp = temp[temp > 0]
            temp.index = temp.index.map(lambda x: 'Switch from {} to {} (Thousand households)'.format(x[0], x[1]))
            output.update((temp / 10 ** 3 / step).T)

            # insulation - who is renovating ?
            temp = replaced_by_grouped.sum(axis=1)
            t = temp.groupby(['Housing type']).sum()
            t.index = t.index.map(lambda x: 'Renovation {} (Thousand households)'.format(x))
            output.update((t / 10 ** 3 / step).T)

            t = temp.groupby(['Housing type', 'Occupancy status']).sum()
            t.index = t.index.map(lambda x: 'Renovation {} - {} (Thousand households)'.format(x[0], x[1]))
            output.update((t / 10 ** 3 / step).T)

            # insulation - what is renovated ?
            o = {}
            for i in ['Wall', 'Floor', 'Roof', 'Windows']:
                temp = replaced_by_grouped.xs(True, level=i, axis=1).sum(axis=1)
                o['Replacement {} (Thousand households)'.format(i)] = temp.sum() / 10 ** 3 / step

                t = temp * reindex_mi(self._surface * self._ratio_surface.loc[:, i], temp.index)
                o['Replacement {} (Million m2)'.format(i)] = t.sum() / 10 ** 6 / step

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
            output['Replacement insulation (Thousand)'] = sum(
                [o['Replacement {} (Thousand households)'.format(i)] for i in ['Wall', 'Floor', 'Roof', 'Windows']])
            output['Replacement insulation average (/household)'] = output['Replacement insulation (Thousand)'] / \
                                                                    output['Renovation (Thousand households)']

            o = Series(o).sort_index(ascending=False)
            output.update(o.T)

            # economic output global
            temp = self._heater_store['cost'].sum()
            output['Investment heater (Billion euro)'] = temp.sum() / 10 ** 9 / step
            temp.index = temp.index.map(lambda x: 'Investment {} (Billion euro)'.format(x))
            output.update(temp.T / 10 ** 9 / step)
            investment_heater = self._heater_store['cost'].sum(axis=1)

            temp = self._heater_store['hidden_cost_agg'].sum() / self._heater_store['replacement'].sum()
            temp.index = temp.index.map(lambda x: 'Hidden cost average {} (euro)'.format(x))
            output.update(temp.T)

            output['Hidden cost heater (Billion euro)'] = self._heater_store['hidden_cost_agg'].sum().sum() / 10 ** 9 / step

            temp = self._heater_store['hidden_cost_agg'].groupby('Housing type').sum() / 10 ** 9 / step
            temp = temp.stack().squeeze()
            temp.index = temp.index.map(lambda x: 'Hidden cost {} - {} (Billion euro)'.format(x[0], x[1]))
            output.update(temp.T)

            output['Financing heater (Billion euro)'] = self._heater_store[
                                                            'cost_financing'].sum().sum() / 10 ** 9 / step

            investment_insulation_full = (self._replaced_by * self._renovation_store['cost_households']).groupby(levels).sum()
            investment_insulation = investment_insulation_full.sum(axis=1)
            output['Investment insulation (Billion euro)'] = investment_insulation.sum() / 10 ** 9 / step

            temp = (self._replaced_by * self._renovation_store['cost_financing_households']).sum().sum()
            output['Financing insulation (Billion euro)'] = temp.sum().sum() / 10 ** 9 / step

            annuities = calculate_annuities(investment_insulation_full, lifetime=lifetime_insulation, discount_rate=social_discount_rate)
            output['Annuities insulation (Billion euro/year)'] = annuities.sum().sum() / 10 ** 9 / step
            output['Efficiency insulation (euro/kWh standard)'] = output['Annuities insulation (Billion euro/year)'] / \
                                                                  output['Consumption standard saving insulation (TWh/year)']

            output['Efficiency insulation (euro/kWh)'] = output['Annuities insulation (Billion euro/year)'] / output[
                'Consumption saving insulation (TWh/year)']
            output['Efficiency insulation (euro/tCO2 standard)'] = output['Annuities insulation (Billion euro/year)'] * 10 ** 3 / \
                                                                   output['Emission standard saving insulation (MtCO2/year)']

            annuities_heater = calculate_annuities(investment_heater, lifetime=20, discount_rate=social_discount_rate)
            output['Annuities heater (Billion euro/year)'] = annuities_heater.sum().sum() / 10 ** 9 / step
            output['Efficiency heater (euro/kWh standard)'] = output['Annuities heater (Billion euro/year)'] / output[
                'Consumption standard saving heater (TWh/year)']
            output['Efficiency heater (euro/kWh)'] = output['Annuities heater (Billion euro/year)'] / output[
                'Consumption saving heater (TWh/year)']

            index = investment_heater.index.union(investment_insulation.index)
            investment_total = investment_heater.reindex(index, fill_value=0) + investment_insulation.reindex(index,
                                                                                                              fill_value=0)
            output['Investment total (Billion euro)'] = investment_total.sum() / 10 ** 9 / step
            output['Financing total (Billion euro)'] = output['Financing insulation (Billion euro)'] + output[
                'Financing heater (Billion euro)']

            # hidden cost
            names = self._renovation_store['hidden_cost_agg'].columns.names
            temp = self._replaced_by.sum()
            temp = temp.reindex(self._renovation_store['hidden_cost_agg'].columns)
            temp.iloc[0] = self._renovation_store['no_renovation']
            temp = self._renovation_store['hidden_cost_agg'].sum() / temp
            temp.index = temp.index.map(lambda row: ", ".join([name for name, value in zip(names, row) if value]))
            temp.index = temp.index.map(lambda x: 'Hidden cost average {} (euro)'.format(x))
            output.update(temp.T)

            """temp = self._renovation_store['hidden_cost_agg'] / self._replaced_by
            temp = temp / reindex_mi(self._renovation_store['cost_households'], self._replaced_by.index)"""

            hidden_cost = (self._renovation_store['hidden_cost_agg']).groupby(levels).sum()
            output['Hidden cost insulation (Billion euro)'] = hidden_cost.sum().sum() / 10 ** 9 / step

            # financing - how households finance renovation - state  / debt / saving ?
            subsidies_insulation = (self._replaced_by * self._renovation_store['subsidies_households']).groupby(levels).sum()
            to_pay = investment_insulation_full - subsidies_insulation

            # cost by income levels
            temp = self.add_level(to_pay, self._stock_ref, 'Income tenant').sum(axis=1)
            temp = temp.groupby('Income tenant').sum()
            temp.index = temp.index.map(lambda x: 'To pay insulation {} (Billion euro/year)'.format(x))
            output.update(temp.T / 10 ** 9 / step)

            annuities_insulation_hh = calculate_annuities(to_pay, lifetime=duration_financing, discount_rate=discount_financing)
            output['Annuities insulation households (Billion euro/year)'] = annuities_insulation_hh.sum().sum() / 10 ** 9 / step

            annuities_cost_insulation_hh = calculate_annuities(investment_insulation_full, lifetime=duration_financing,
                                                               discount_rate=discount_financing)
            del investment_insulation_full

            annuities_subsidies_insulation_hh = calculate_annuities(subsidies_insulation, lifetime=duration_financing,
                                                                    discount_rate=discount_financing)

            subsidies_insulation = subsidies_insulation.sum(axis=1)
            output['Subsidies insulation (Billion euro)'] = subsidies_insulation.sum() / 10 ** 9 / step
            if output['Subsidies insulation (Billion euro)'] != 0:
                output['Lever insulation (%)'] = output['Investment insulation (Billion euro)'] / output[
                    'Subsidies insulation (Billion euro)']
                annuities_sub = calculate_annuities(output['Subsidies insulation (Billion euro)'], lifetime=lifetime_insulation, discount_rate=social_discount_rate)
                output['Efficiency subsidies insulation (euro/kWh standard)'] = annuities_sub / output['Consumption standard saving insulation (TWh/year)']

            subsidies_loan_insulation = (self._replaced_by * self._renovation_store['subsidies_loan_households']).groupby(levels).sum()
            subsidies_loan_insulation = subsidies_loan_insulation.sum(axis=1)
            output['Subsidies loan insulation (Billion euro)'] = subsidies_loan_insulation.sum() / 10 ** 9 / step

            debt = (self._replaced_by * self._renovation_store['debt_households']).groupby(levels).sum().sum(
                axis=1).groupby(levels_owner).sum()
            del self._renovation_store['debt_households']

            output['Debt insulation (Billion euro)'] = debt.sum() / 10 ** 9 / step

            saving = (self._replaced_by * self._renovation_store['saving_households']).groupby(
                levels).sum().sum(axis=1).groupby(levels_owner).sum()
            del self._renovation_store['saving_households']
            output['Saving insulation (Billion euro)'] = saving.sum() / 10 ** 9 / step

            # financing - how households finance new heating system
            subsidies_heater = self._heater_store['subsidies'].sum(axis=1)
            to_pay = investment_heater - subsidies_heater

            temp = self.add_level(to_pay, self._stock_ref, 'Income tenant')
            temp = temp.groupby('Income tenant').sum()
            temp.index = temp.index.map(lambda x: 'To pay heater {} (Billion euro/year)'.format(x))
            output.update(temp.T / 10 ** 9 / step)

            temp = [output['To pay heater {} (Billion euro/year)'.format(i)] + output['To pay insulation {} (Billion euro/year)'.format(i)] +
                    output['Energy expenditures {} (Billion euro)'.format(i)] for i in self._resources_data['index']['Income tenant']]
            temp = Series(temp, index=self._resources_data['index']['Income tenant'])
            t = self.stock.groupby('Income tenant').sum() / 1e6
            t = temp / t
            temp.index = temp.index.map(lambda x: 'Total expenditure {} (Billion euro)'.format(x))
            t.index = t.index.map(lambda x: 'Total expenditure {} (Thousand euro/hh)'.format(x))
            output.update(temp.T / step)
            output.update(t.T)

            # subsidies_heater_hh = calculate_annuities(subsidies_heater, lifetime=10, discount_rate=0.05)
            annuities_heater_hh = calculate_annuities(to_pay, lifetime=duration_financing, discount_rate=discount_financing)
            output['Annuities heater households (Billion euro/year)'] = annuities_heater_hh.sum().sum() / 10 ** 9 / step

            annuities_cost_heater_hh = calculate_annuities(investment_heater, lifetime=duration_financing,
                                                           discount_rate=discount_financing)
            annuities_subsidies_heater_hh = calculate_annuities(subsidies_heater, lifetime=duration_financing,
                                                                discount_rate=discount_financing)

            output['Subsidies heater (Billion euro)'] = subsidies_heater.sum() / 10 ** 9 / step
            subsidies_loan_heater = self._heater_store['subsidies_loan'].sum(axis=1)
            output['Subsidies loan heater (Billion euro)'] = subsidies_loan_heater.sum() / 10 ** 9 / step

            temp = self._heater_store['debt'].loc[self._resources_data['index']['Income owner']]
            output['Debt heater (Billion euro)'] = temp.sum() / 10 ** 9 / step

            temp = self._heater_store['saving'].loc[self._resources_data['index']['Income owner']]
            output['Saving heater (Billion euro)'] = temp.sum() / 10 ** 9 / step

            index = subsidies_heater.index.union(subsidies_insulation.index)
            subsidies_total = subsidies_heater.reindex(index, fill_value=0) + subsidies_insulation.reindex(index, fill_value=0)
            output['Subsidies total (Billion euro)'] = subsidies_total.sum() / 10 ** 9 / step

            index = subsidies_loan_heater.index.union(subsidies_loan_insulation.index)
            subsidies_loan_total = subsidies_loan_heater.reindex(index, fill_value=0) + subsidies_loan_insulation.reindex(index, fill_value=0)
            output['Subsidies loan total (Billion euro)'] = subsidies_loan_total.sum() / 10 ** 9 / step

            output['Debt total (Billion euro)'] = output['Debt heater (Billion euro)'] + output[
                'Debt insulation (Billion euro)']
            output['Saving total (Billion euro)'] = output['Saving heater (Billion euro)'] + output[
                'Saving insulation (Billion euro)']

            # macro-average
            output['Investment total (Thousand euro/household)'] = output['Investment total (Billion euro)'] * 10 ** 3 / \
                                                                   output['Retrofit (Thousand households)']
            output['Investment insulation (Thousand euro/household)'] = 0
            if output['Renovation (Thousand households)'] != 0:
                output['Investment insulation (Thousand euro/household)'] = output[
                                                                                'Investment insulation (Billion euro)'] * 10 ** 3 / \
                                                                            output['Renovation (Thousand households)']
                output['Subsidies insulation (Thousand euro/household)'] = output[
                                                                               'Subsidies insulation (Billion euro)'] * 10 ** 3 / \
                                                                           output['Renovation (Thousand households)']
                output['Saving insulation (Thousand euro/household)'] = output[
                                                                            'Saving insulation (Billion euro)'] * 10 ** 3 / \
                                                                        output['Renovation (Thousand households)']
                output['Debt insulation (Thousand euro/household)'] = output[
                                                                          'Debt insulation (Billion euro)'] * 10 ** 3 / \
                                                                      output['Renovation (Thousand households)']

            # subsidies - description
            temp = subsidies_total.groupby('Income owner').sum() / investment_total.groupby('Income owner').sum()
            temp = temp.loc[self._resources_data['index']['Income owner']]
            temp.index = temp.index.map(lambda x: 'Share subsidies {} (%)'.format(x))
            output.update(temp.T)

            levels_owner = debt.index.names
            if self._replaced_by.sum().sum().round(0) > 0:
                lvls = ['Housing type', 'Occupancy status', 'Income tenant']

                # annuities include financing cost
                annuities_insulation_rented = annuities_insulation_hh[annuities_insulation_hh.index.get_level_values('Occupancy status').isin(['Privately rented', 'Social-housing'])]
                annuities_insulation_rented = annuities_insulation_rented.groupby(['Housing type', 'Income owner']).sum().sum(axis=1)

                annuities_insulation_year = self.add_level(annuities_insulation_hh, self._stock_ref, 'Income tenant')
                annuities_insulation_year = annuities_insulation_year.groupby(lvls).sum().sum(axis=1)
                temp = annuities_insulation_year.copy()
                temp.index = temp.index.map(lambda x: 'Annuities insulation {} - {} - {}'.format(x[0], x[1], x[2]))
                output.update(temp.T)

                temp = self.add_level(annuities_cost_insulation_hh, self._stock_ref, 'Income tenant')
                temp = temp.groupby(lvls).sum().sum(axis=1)
                temp.index = temp.index.map(lambda x: 'Annuities cost insulation {} - {} - {}'.format(x[0], x[1], x[2]))
                output.update(temp.T)

                temp = self.add_level(annuities_subsidies_insulation_hh, self._stock_ref, 'Income tenant')
                temp = temp.groupby(lvls).sum().sum(axis=1)
                temp.index = temp.index.map(lambda x: 'Annuities subsidies insulation {} - {} - {}'.format(x[0], x[1], x[2]))
                output.update(temp.T)

                annuities_heater_rented = annuities_heater_hh[annuities_heater_hh.index.get_level_values('Occupancy status').isin(['Privately rented', 'Social-housing'])]
                annuities_heater_rented = annuities_heater_rented.groupby(['Housing type', 'Income owner']).sum()

                annuities_heater_year = self.add_level(annuities_heater_hh, self._stock_ref, 'Income tenant')
                annuities_heater_year = annuities_heater_year.groupby(lvls).sum()
                temp = annuities_heater_year.copy()
                temp.index = temp.index.map(lambda x: 'Annuities heater {} - {} - {}'.format(x[0], x[1], x[2]))
                output.update(temp.T)

                temp = self.add_level(annuities_cost_heater_hh, self._stock_ref, 'Income tenant')
                temp = temp.groupby(lvls).sum()
                temp.index = temp.index.map(lambda x: 'Annuities cost heater {} - {} - {}'.format(x[0], x[1], x[2]))
                output.update(temp.T)

                temp = self.add_level(annuities_subsidies_heater_hh, self._stock_ref, 'Income tenant')
                temp = temp.groupby(lvls).sum()
                temp.index = temp.index.map(lambda x: 'Annuities subsidies heater {} - {} - {}'.format(x[0], x[1], x[2]))
                output.update(temp.T)

                annuities_year = annuities_insulation_year + annuities_heater_year
                annuities_rented_year = annuities_insulation_rented + annuities_heater_rented
                # annuities_rented_year add Landlord as occupancy status to account for cost
                index_arrays = [
                    annuities_rented_year.index.get_level_values(0),  # Housing type
                    annuities_rented_year.index.get_level_values(1),  # Income owner
                    ['Landlord'] * len(annuities_rented_year)  # Occupancy status
                ]
                new_index = pd.MultiIndex.from_arrays(index_arrays,
                                                      names=['Housing type', 'Income tenant', 'Occupancy status'])
                annuities_rented_year.index = new_index
                annuities_rented_year = annuities_rented_year.reorder_levels(['Housing type', 'Occupancy status', 'Income tenant'])
                annuities_year = concat((annuities_year, annuities_rented_year), axis=0)
                temp = annuities_year.copy()
                temp.index = temp.index.map(lambda x: 'Annuities {} - {} - {}'.format(x[0], x[1], x[2]))
                output.update(temp.T)

                """duration = 10
                years = [y for y in self.expenditure_store.keys() if y > self.year - duration]
                annuities_cumulated = sum([self.expenditure_store[y]['annuities'] for y in years])
                annuities_cumulated += annuities_year
                """

                consumption_std = reindex_mi(self.consumption_heating(full_output=False), self.stock.index)
                consumption_std *= reindex_mi(self._surface, self.stock.index)
                energy_exp_std = self.energy_bill(prices, consumption_std, bill_rebate=0)
                energy_exp_std *= self.stock
                energy_exp_std = energy_exp_std.groupby(lvls).sum()
                temp = energy_exp_std.copy()
                temp.index = temp.index.map(
                    lambda x: 'Energy expenditures standard {} - {} - {} (euro)'.format(x[0], x[1], x[2]))
                output.update(temp.T)

                consumption = self.consumption_actual(prices, bill_rebate=bill_rebate)
                energy_exp = self.energy_bill(prices, consumption, bill_rebate=bill_rebate)
                energy_exp *= self.stock
                energy_exp = energy_exp.groupby(lvls).sum()
                temp = energy_exp.copy()
                temp.index = temp.index.map(
                    lambda x: 'Energy expenditures {} - {} - {} (euro)'.format(x[0], x[1], x[2]))
                output.update(temp.T)

                consumption *= self.stock
                temp = consumption.groupby(lvls).sum().copy()
                temp.index = temp.index.map(
                    lambda x: 'Consumption {} - {} - {} (kWh)'.format(x[0], x[1], x[2]))
                output.update(temp.T)

                consumption = self.consumption_actual(prices, bill_rebate=bill_rebate)
                consumption *= self.stock
                emission = self.to_emission(consumption, inputs['carbon_emission'].loc[self.year, :])
                emission = emission.groupby(lvls).sum()
                temp = emission.copy()
                temp.index = temp.index.map(
                    lambda x: 'Emission {} - {} - {} (gCO2)'.format(x[0], x[1], x[2]))
                output.update(temp.T)

                s = self.stock.groupby(lvls).sum()
                temp = s.copy()
                temp.index = temp.index.map(lambda x: 'Stock {} - {} - {}'.format(x[0], x[1], x[2]))
                output.update(temp.T)

                i = (reindex_mi(self._income_tenant, s.index) * s)
                temp = i.copy()
                temp.index = temp.index.map(lambda x: 'Income {} - {} - {} (euro)'.format(x[0], x[1], x[2]))
                output.update(temp.T)

                # ------------------------------------------------------------------------------------------------------
                # following is not used most of it are just for testing
                owner = replaced_by_grouped.sum(axis=1).groupby(levels_owner).sum()
                temp = annuities_insulation_hh.sum(axis=1).groupby(levels_owner).sum() / owner
                temp.index = temp.index.map(
                    lambda x: 'Annuities insulation {} - {} (euro/year.household)'.format(x[0], x[1]))
                output.update(temp.T)

                temp = debt / owner
                temp.index = temp.index.map(lambda x: 'Debt insulation {} - {} (euro/household)'.format(x[0], x[1]))
                output.update(temp.T)

                temp = saving / owner
                temp.index = temp.index.map(lambda x: 'Saving insulation {} - {} (euro/household)'.format(x[0], x[1]))
                output.update(temp.T)

                # ------------------------------------------------------------------------------------------------------
                # economic private impact - distributive indicator - are investment profitable ?
                if False:
                    temp = annuities_insulation_hh.sum(axis=1).xs('Privately rented', level='Occupancy status', drop_level=False)
                    temp = self.add_level(temp, self._stock_ref, 'Income tenant')
                    owner_tenant = replaced_by_grouped.sum(axis=1)
                    owner_tenant = self.add_level(owner_tenant, self._stock_ref, 'Income tenant')
                    if 'Privately rented' in owner_tenant.index.get_level_values('Occupancy status'):
                        owner_private = owner_tenant.xs('Privately rented', level='Occupancy status', drop_level=False)
                        temp = temp.groupby('Income tenant').sum() / owner_private.groupby('Income tenant').sum()
                        temp = temp.loc[self._resources_data['index']['Income tenant']]
                        temp.index = temp.index.map(lambda x: 'Rent {} (euro/year.household)'.format(x))
                        output.update(temp.T)

                    temp = consumption_saved_insulation.sum(axis=1)
                    del consumption_saved_insulation

                    temp = self.add_level(temp, self._stock_ref, 'Income tenant')
                    bill_saving = self.to_emission(temp, prices)
                    temp = bill_saving.groupby(['Occupancy status', 'Income tenant']).sum() / owner_tenant.groupby(
                        ['Occupancy status', 'Income tenant']).sum()
                    temp.index = temp.index.map(
                        lambda x: 'Bill saving standard {} - {} (euro/year.household)'.format(x[0], x[1]))
                    output.update(temp.T)

                    temp = consumption_saved_actual_insulation.sum(axis=1)
                    del consumption_saved_actual_insulation
                    bill_saving = self.to_emission(temp, prices)
                    temp = bill_saving.groupby(['Occupancy status', 'Income tenant']).sum() / owner_tenant.groupby(
                        ['Occupancy status', 'Income tenant']).sum()
                    temp.index = temp.index.map(lambda x: 'Bill saving {} - {} (euro/year.household)'.format(x[0], x[1]))
                    output.update(temp.T)

                    # private balance
                    temp = [- output['Annuities insulation Owner-occupied - {} (euro/year.household)'.format(i)] + output[
                        'Bill saving Owner-occupied - {} (euro/year.household)'.format(i)] for i in
                            self._resources_data['index']['Income owner']]
                    temp = Series(temp, index=self._resources_data['index']['Income owner'])
                    temp.index = temp.index.map(lambda x: 'Balance Owner-occupied - {} (euro/year.household)'.format(x))
                    output.update(temp.T)

                    temp = [- output['Rent {} (euro/year.household)'.format(i)] + output[
                        'Bill saving Privately rented - {} (euro/year.household)'.format(i)] for i in
                            self._resources_data['index']['Income tenant']]
                    temp = Series(temp, index=self._resources_data['index']['Income tenant'])
                    temp.index = temp.index.map(lambda x: 'Balance Tenant private - {} (euro/year.household)'.format(x))
                    output.update(temp.T)

                    # private balance standard
                    temp = [- output['Annuities insulation Owner-occupied - {} (euro/year.household)'.format(i)] + output[
                        'Bill saving standard Owner-occupied - {} (euro/year.household)'.format(i)] for i in
                            self._resources_data['index']['Income owner']]
                    temp = Series(temp, index=self._resources_data['index']['Income owner'])
                    temp.index = temp.index.map(
                        lambda x: 'Balance standard Owner-occupied - {} (euro/year.household)'.format(x))
                    output.update(temp.T)

                    temp = [- output['Rent {} (euro/year.household)'.format(i)] + output[
                        'Bill saving standard Privately rented - {} (euro/year.household)'.format(i)] for i in
                            self._resources_data['index']['Income tenant']]
                    temp = Series(temp, index=self._resources_data['index']['Income tenant'])
                    temp.index = temp.index.map(
                        lambda x: 'Balance standard Tenant private - {} (euro/year.household)'.format(x))
                    output.update(temp.T)

            # economic state impact
            output['VAT heater (Billion euro)'] = self._heater_store['vat'] / 10 ** 9 / step

            temp = (self._replaced_by * self._renovation_store['vat_households']).sum().sum()
            output['VAT insulation (Billion euro)'] = temp / 10 ** 9 / step
            output['VAT (Billion euro)'] = output['VAT heater (Billion euro)'] + output['VAT insulation (Billion euro)']
            output['Investment heater WT (Billion euro)'] = output['Investment heater (Billion euro)'] - output[
                'VAT heater (Billion euro)']
            output['Investment insulation WT (Billion euro)'] = output['Investment insulation (Billion euro)'] - output[
                'VAT insulation (Billion euro)']
            output['Investment total WT (Billion euro)'] = output['Investment total (Billion euro)'] - output[
                'VAT (Billion euro)']
            output['Investment total WT / households (Thousand euro)'] = output['Investment total WT (Billion euro)'] * 10 ** 6 / (
                                                                                     output['Retrofit (Thousand households)'] * 10 ** 3)

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
                output['Carbon value indirect renovation (Billion euro)'] = output['Carbon footprint renovation (MtCO2)'] * carbon_value / 10 ** 3 / step
                output['Carbon value indirect (Billion euro)'] = output['Carbon footprint (MtCO2)'] * carbon_value / 10 ** 3 / step

            output['Bill rebate (euro/(household.year))'] = bill_rebate
            consumption = self.consumption_actual(prices, bill_rebate=bill_rebate)
            temp = (self.energy_bill(prices, consumption, bill_rebate=0) - self.energy_bill(prices, consumption, bill_rebate=bill_rebate))
            temp *= self.stock
            output['Bill rebate (Billion euro)'] = temp.sum() / 1e9
            output['Income state (Billion euro)'] = output['VAT (Billion euro)'] + output['Taxes expenditure (Billion euro)']
            output['Expenditure state (Billion euro)'] = output['Subsidies heater (Billion euro)'] + output[
                'Subsidies insulation (Billion euro)'] + output['Health expenditure (Billion euro)'] + output['Bill rebate (Billion euro)']
            output['Balance state (Billion euro)'] = output['Income state (Billion euro)'] - output['Expenditure state (Billion euro)']
            if self._balance_state_ini is None:
                self._balance_state_ini = output['Balance state (Billion euro)']

            # subsidies - details: policies amount and number of beneficiaries
            subsidies_details_renovation, replacement_eligible_renovation, subsidies_average_renovation, cost_average_renovation = {}, {}, {}, {}
            energy_saved_renovation = {}
            for key, sub in self._renovation_store['subsidies_details_households'].items():
                subsidies_details_renovation[key] = (
                        self._replaced_by * reindex_mi(sub, self._replaced_by.index)).groupby(levels).sum()

                if key in self._renovation_store['eligible'].keys():
                    eligible = self._renovation_store['eligible'][key]
                else:
                    eligible = sub.copy()
                    eligible[eligible > 0] = 1
                    eligible = eligible.any(axis=1)

                replacement_eligible = self._replaced_by.fillna(0).sum(axis=1) * eligible
                replacement_eligible_renovation[key] = replacement_eligible.groupby('Housing type').sum()

                if eligible.sum().sum() == 0:
                    subsidies_average_renovation[key] = 0
                    cost_average_renovation[key] = 0
                else:
                    subsidies_average_renovation[key] = sub.sum().sum() / replacement_eligible.sum()
                    cost = reindex_mi((self._renovation_store['cost_households'] - self._renovation_store['vat_households']), replacement_eligible.index)
                    cost = ((cost * self._replaced_by.fillna(0)).T * eligible).T
                    cost_average_renovation[key] = cost.sum().sum() / replacement_eligible.sum()

                    energy_saved = reindex_mi(self._renovation_store['consumption_saved_households'], replacement_eligible.index)
                    energy_saved = ((energy_saved * self._replaced_by.fillna(0)).T * eligible).T
                    energy_saved_renovation[key] = energy_saved.sum().sum() / replacement_eligible.sum()

            del self._renovation_store['subsidies_details_households']
            gc.collect()

            subsidies, replacement_eligible, sub_count, cost_average, energy_average = None, None, None, None, None
            for gest, subsidies_details in {'heater': self._heater_store['subsidies_details'],
                                            'insulation': subsidies_details_renovation}.items():
                if gest == 'heater':
                    sub_count = DataFrame(self._heater_store['replacement_eligible'], dtype=float)
                    cost_average = Series(self._heater_store['cost_average'], dtype=float)

                elif gest == 'insulation':
                    sub_count = DataFrame(replacement_eligible_renovation, dtype=float)
                    cost_average = Series(cost_average_renovation, dtype=float)
                    energy_average = Series(energy_saved_renovation, dtype=float)

                subsidies_details = Series({k: i.sum().sum() for k, i in subsidies_details.items()}, dtype='float64')

                for i in subsidies_details.index:
                    # use subsidies in post-treatment
                    if '{} {}'.format(i, gest) in inputs['use_subsidies'].index:
                        use_subsidies = inputs['use_subsidies'].loc['{} {}'.format(i, gest)]
                        subsidies_details[i] *= use_subsidies
                        sub_count[i] *= use_subsidies

                    temp = sub_count[i].copy()
                    temp.index = temp.index.map(
                        lambda x: '{} {} {} (Thousand households)'.format(i.capitalize().replace('_', ' '), gest, x))
                    output.update(temp.T / 10 ** 3 / step)

                    output.update({'{} {} (Thousand households)'.format(i.capitalize().replace('_', ' '), gest):
                                       sub_count[i].sum() / 1e3 / step})
                    output['Average cost {} {} (euro)'.format(i.capitalize().replace('_', ' '), gest)] = cost_average.loc[i]

                    if gest == 'insulation' and i in energy_average.keys():
                        output['Average energy std saved {} {} (MWh)'.format(i.capitalize().replace('_', ' '), gest)] = \
                        energy_average.loc[i] / 1e3

                    output['{} {} (Billion euro)'.format(i.capitalize().replace('_', ' '), gest)] = \
                        subsidies_details.loc[i] / 10 ** 9 / step

                if subsidies is None:
                    subsidies = subsidies_details.copy()
                    replacement_eligible = sub_count.copy()
                    cost = (cost_average * sub_count.sum()).copy()
                else:
                    subsidies = concat((subsidies, subsidies_details), axis=0)
                    replacement_eligible = concat((replacement_eligible, sub_count), axis=0)
                    cost = concat((cost, cost_average * sub_count.sum()), axis=0)

            if subsidies is not None:
                subsidies = subsidies.groupby(subsidies.index).sum()
                replacement_eligible = replacement_eligible.groupby(replacement_eligible.index).sum()
                cost = cost.groupby(cost.index).sum() / replacement_eligible.sum()

                for i in subsidies.index:
                    temp = replacement_eligible[i]
                    output['{} (Thousand households)'.format(
                        i.capitalize().replace('_', ' '))] = temp.sum() / 10 ** 3 / step
                    temp.index = temp.index.map(
                        lambda x: '{} {} (Thousand households)'.format(i.capitalize().replace('_', ' '), x))
                    output.update(temp.T / 10 ** 3)
                    output['{} (Billion euro)'.format(i.capitalize().replace('_', ' '))] = subsidies.loc[i] / 10 ** 9 / step

                    output['Average cost {} (euro)'.format(i.capitalize().replace('_', ' '))] = cost.loc[i]

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

            """
            self.store_over_years[self.year].update(
                {'Annuities heater (Billion euro/year)': output['Annuities heater (Billion euro/year)'],
                 'Annuities insulation (Billion euro/year)': output['Annuities insulation (Billion euro/year)'],
                 })
            years = [y for y in self.store_over_years.keys() if y > self.year - 20 and 'Annuities heater (Billion euro/year)' in self.store_over_years[y].keys()]
            annuities_heater_cumulated = sum([self.store_over_years[y]['Annuities heater (Billion euro/year)'] for y in years])

            # TODO: Cumulated 20 years ?
            years = [y for y in self.store_over_years.keys() if y > self.year - 20 and 'Annuities insulation (Billion euro/year)' in self.store_over_years[y].keys()]
            annuities_insulation_cumulated = sum(
                [self.store_over_years[y]['Annuities insulation (Billion euro/year)'] for y in years])
            """

            # Opportunity cost of public spending
            output['COFP (Billion euro)'] = 0
            balance = self._balance_state_ini - output['Balance state (Billion euro)']
            # if > 0 it's a cost because income net of expenditures decrease compared to initial state
            temp = balance * coefficient_public_funding
            output['COFP (Billion euro)'] = temp

            # running cost

            output['Cost energy (Billion euro)'] = output['Energy expenditures wt (Billion euro)']
            output['Cost emission (Billion euro)'] = (output['Emission (MtCO2)'] * carbon_value) / 10**3
            output['Cost heath (Billion euro)'] = output['Health cost (Billion euro)']
            output['Loss thermal comfort (Billion euro)'] = - output['Thermal comfort EE (Billion euro)'] + output['Thermal loss prices (Billion euro)']
            """
            output['Cost heater (Billion euro)'] = annuities_heater_cumulated
            output['Cost insulation (Billion euro)'] = annuities_insulation_cumulated
            variables = ['Cost energy (Billion euro)', 'Cost emission (Billion euro)', 'Cost heath (Billion euro)',
                         'Loss thermal comfort (Billion euro)', 'Cost heater (Billion euro)',
                         'Cost insulation (Billion euro)']
            output['Running cost (Billion euro)'] = sum([output[i] for i in variables])
            """

            consumption_saving_ee = (_consumption_saved_actual_insulation + self._heater_store['consumption_saved_actual'])

            output['CBA Consumption saving (TWh)'] = output['Consumption saving (TWh/year)']
            output['CBA Consumption saving insulation (TWh)'] = _consumption_saved_actual_insulation.sum() / 10**9
            output['CBA Consumption saving switch fuel (TWh)'] = self._heater_store['consumption_saved_actual'].sum() / 10**9
            output['CBA Consumption saving EE (TWh)'] = consumption_saving_ee.sum() / 10**9
            output['CBA Consumption saving prices (TWh)'] = output['Consumption saving prices effect (TWh/year)']
            output['CBA Rebound EE (TWh)'] = output['Rebound EE (TWh/year)']

            # cost-benefits analysis compared to previous year
            energy_wt_value = calculate_average(inputs['energy_prices_wt'].loc[self.year:], lifetime=lifetime_insulation,
                                                discount_rate=social_discount_rate)
            carbon_value_mean = calculate_average(inputs['carbon_value'].loc[self.year:], lifetime=lifetime_insulation,
                                                  discount_rate=social_discount_rate)

            consumption_saving_ee = (consumption_saving_ee * energy_wt_value).sum() / 10**9
            output['CBA Consumption saving EE (Billion euro)'] = consumption_saving_ee
            output['CBA Consumption saving prices (Billion euro)'] = output['Consumption saving prices (Billion euro)']
            output['CBA Thermal comfort EE (Billion euro)'] = output['Thermal comfort EE (Billion euro)']
            output['CBA Emission direct (Billion euro)'] = (output['Emission saving (MtCO2/year)'] * carbon_value_mean) / 10**3
            output['CBA Health cost (Billion euro)'] = self.store_over_years[self.year - 1]['Health cost (Billion euro)'] - self.store_over_years[self.year]['Health cost (Billion euro)']

            output['CBA benefits (Billion euro)'] = output['CBA Consumption saving EE (Billion euro)'] + \
                                                    output['CBA Consumption saving prices (Billion euro)'] + \
                                                    output['CBA Thermal comfort EE (Billion euro)'] + \
                                                    output['CBA Emission direct (Billion euro)'] + \
                                                    output['CBA Health cost (Billion euro)']

            # cost
            output['CBA Thermal loss prices (Billion euro)'] = - output['Thermal loss prices (Billion euro)']
            output['CBA Annuities insulation (Billion euro)'] = - output['Annuities insulation (Billion euro/year)']
            output['CBA Annuities heater (Billion euro)'] = - output['Annuities heater (Billion euro/year)']
            temp = calculate_annuities(output['Carbon value indirect renovation (Billion euro)'],
                                       lifetime=lifetime_insulation,
                                       discount_rate=social_discount_rate)
            output['CBA Carbon Emission indirect (Billion euro)'] = - temp
            output['CBA COFP (Billion euro)'] = - output['COFP (Billion euro)']
            output['CBA cost (Billion euro)'] = output['CBA Annuities heater (Billion euro)'] + \
                                                output['CBA Annuities insulation (Billion euro)'] + \
                                                output['CBA Carbon Emission indirect (Billion euro)'] + \
                                                output['CBA Thermal loss prices (Billion euro)'] + output['CBA COFP (Billion euro)']

            output['Cost-benefits analysis (Billion euro)'] = output['CBA benefits (Billion euro)'] + output['CBA cost (Billion euro)']

        output = Series(output).rename(self.year)
        stock = stock.rename(self.year)
        return stock, output

    def parse_output_run_cba(self, prices, inputs, step=1, taxes=None, bill_rebate=0):
        output = dict()

        # emission
        emission = inputs['carbon_emission'].loc[self.year, :]
        consumption_energy = self.consumption_agg(prices=prices, freq='year', climate=None, standard=False, agg='energy',
                                                  bill_rebate=bill_rebate)
        temp = consumption_energy * emission
        output['Emission (MtCO2)'] = temp.sum() / 10 ** 3

        # investment
        investment_cost = (self._replaced_by * self._renovation_store['cost_households']).sum().sum()
        investment_cost = investment_cost.sum() / 10 ** 9 / step
        vat = (self._replaced_by * self._renovation_store['vat_households']).sum().sum()
        output['VAT insulation (Billion euro)'] = vat / 10 ** 9 / step
        output['Investment insulation WT (Billion euro)'] = investment_cost - output['VAT insulation (Billion euro)']

        # vat
        investment_heater = self._heater_store['cost'].sum().sum() / 10 ** 9 / step
        output['VAT heater (Billion euro)'] = self._heater_store['vat'] / 10 ** 9 / step
        output['Investment heater WT (Billion euro)'] = investment_heater - output['VAT heater (Billion euro)']

        output['Health cost (Billion euro)'] = self.health_cost(inputs['health_cost_dpe'], inputs['health_cost_income'], prices)

        output['VAT (Billion euro)'] = output['VAT insulation (Billion euro)'] + output['VAT heater (Billion euro)']
        output['Health expenditure (Billion euro)'] = 0 # temp['Health expenditure (Billion euro)']
        output['Subsidies heater (Billion euro)'] = self._heater_store['subsidies'].sum().sum() / 10 ** 9 / step
        output['Subsidies insulation (Billion euro)'] = (self._replaced_by * self._renovation_store[
            'subsidies_households']).sum().sum() / 10 ** 9
        output['Subsidies (Billion euro)'] = output['Subsidies heater (Billion euro)'] + output['Subsidies insulation (Billion euro)']

        if taxes is not None:
            consumption_energy = self.consumption_agg(prices=prices, freq='year', climate=None, standard=False,
                                                      agg='energy', bill_rebate=bill_rebate)

            taxes_expenditures = dict()
            total_taxes = Series(0, index=prices.index)
            for tax in taxes:
                if self.year in tax.value.index:
                    if tax.name not in self.taxes_list:
                        self.taxes_list += [tax.name]
                    amount = tax.value.loc[self.year, :] * consumption_energy
                    taxes_expenditures[tax.name] = amount
                    total_taxes += amount

            output['Taxes expenditure (Billion euro)'] = DataFrame(taxes_expenditures).sum().sum() / step

        output['Income state (Billion euro)'] = output['VAT (Billion euro)'] + output['Taxes expenditure (Billion euro)']
        output['Expenditure state (Billion euro)'] = output['Subsidies (Billion euro)'] + output['Health expenditure (Billion euro)']

        output['Balance state (Billion euro)'] = output['Income state (Billion euro)'] - output['Expenditure state (Billion euro)']

        if False:
            levels = [i for i in self._replaced_by.index.names if
                      i not in ['Heater replacement', 'Heating system final']]
            temp = self.add_level(self._replaced_by.fillna(0), self._stock_ref, 'Income tenant')
            consumption_saved_actual_insulation = (
                        temp * reindex_mi(self._renovation_store['consumption_saved_actual_households'],
                                          temp.index)).groupby(levels + ['Income tenant']).sum()
            consumption_saved_no_rebound_insulation = (
                        temp * reindex_mi(self._renovation_store['consumption_saved_no_rebound_households'],
                                          temp.index)).groupby(levels + ['Income tenant']).sum()
            rebound_insulation = (consumption_saved_no_rebound_insulation - consumption_saved_actual_insulation).sum(
                axis=1)
            rebound_insulation = rebound_insulation.groupby(self.to_energy(rebound_insulation)).sum()

            rebound_heater = self._heater_store['rebound']

            thermal_comfort = (rebound_insulation + rebound_heater) * prices
            output.update({'Thermal comfort EE (Billion euro)': thermal_comfort.sum() / 10 ** 9})

            """_consumption_saved_actual_insulation = self.apply_calibration(
                consumption_saved_actual_insulation.sum(axis=1))

            consumption_saved_price_constant = _consumption_saved_actual_insulation + self._heater_store[
                'consumption_saved_actual']
            output['Consumption saving prices constant (TWh/year)'] = consumption_saved_price_constant.sum() / 10 ** 9
            output['Consumption saving prices effect (TWh/year)'] = round(
                output['Consumption saving (TWh/year)'] - output['Consumption saving prices constant (TWh/year)'], 3)

            prices_effect = pd.Series({i: output['Consumption {} saving (TWh/year)'.format(i)] -
                                          consumption_saved_price_constant.loc[i] / 10 ** 9 for i in
                                       consumption_saved_price_constant.index})
            thermal_loss = prices_effect * prices
            output.update({'Thermal loss prices (Billion euro)': round(thermal_loss.sum(), 3)})"""

        output = Series(output).rename(self.year)

        return output

    def parse_output_consumption(self, prices, bill_rebate=0):
        output = self.consumption_agg(prices=prices, freq='year', climate=None, standard=False, agg='energy',
                                      bill_rebate=bill_rebate)
        output.index = output.index.map(lambda x: 'Consumption {} (TWh)'.format(x))
        temp = prices.T
        temp.index = temp.index.map(lambda x: 'Prices {} (euro/kWh)'.format(x))
        output = pd.concat((output, temp), axis=0).rename(self.year)
        return output

    def apply_scale(self, scale, gest='insulation'):
        def calculate_indicators_insulation():

            self.hidden_cost_insulation = - self.constant_insulation_intensive / abs(
                self.preferences_insulation['cost']) * 1000

            hidden_cost_renovation = - self.constant_insulation_extensive / abs(
                self.preferences_insulation['cost']) * 1000

            # estimate of friction. First landlord-tenant dilemma, then multi-family friction.
            if self.no_friction is False:

                hidden_cost = hidden_cost_renovation.loc[('Single-family', 'Owner-occupied')]

                temp = hidden_cost_renovation.xs('Owner-occupied', level='Occupancy status', drop_level=True).copy()

                landlord_dilemma = hidden_cost_renovation - temp

                multi_family_friction = hidden_cost_renovation - (hidden_cost + landlord_dilemma)

                # test = hidden_cost + landlord_dilemma + multi_family_friction

                market_failures = Series(0, index=hidden_cost_renovation.index)
                market_failures += landlord_dilemma.reindex(market_failures.index).fillna(0)
                market_failures += multi_family_friction.reindex(market_failures.index).fillna(0)

                hidden_cost = hidden_cost_renovation - market_failures

                hidden_cost_renovation = concat([hidden_cost] * len(self.hidden_cost_insulation), axis=1, keys=self.hidden_cost_insulation.index)
                hidden_cost_renovation = hidden_cost_renovation + self.hidden_cost_insulation

                self.hidden_cost_renovation = hidden_cost_renovation
                self.hidden_cost = hidden_cost
                self.landlord_dilemma = landlord_dilemma
                self.multifamily_friction = multi_family_friction

                self.constant_insulation = - self.hidden_cost_renovation / 1000 * abs(self.preferences_insulation['cost'])
                self.utility_hidden_cost = - self.hidden_cost / 1000 * abs(self.preferences_insulation['cost'])
            else:
                constant_insulation = concat([self.constant_insulation_extensive] * len(self.constant_insulation_intensive), axis=1, keys=self.constant_insulation_intensive.index)
                constant_insulation = constant_insulation + self.constant_insulation_intensive
                self.constant_insulation = constant_insulation.copy()

        if gest == 'insulation':
            self.scale_insulation *= scale
            self.preferences_insulation['subsidy'] *= scale
            self.preferences_insulation['cost'] *= scale
            self.preferences_insulation['bill_saved'] *= scale

            calculate_indicators_insulation()
        elif gest == 'heater':
            self.scale_heater *= scale
            self.preferences_heater['subsidy'] *= scale
            self.preferences_heater['cost'] *= scale
            self.preferences_heater['bill_saved'] *= scale
            self.preferences_heater['status_quo_bias'] *= scale
            self.hidden_cost_heater = - self.constant_heater / abs(self.preferences_heater['cost']) * 1000

    def calibration_exogenous(self, coefficient_global=None, coefficient_backup=None, constant_heater=None,
                              scale_heater=None, constant_insulation_intensive=None, constant_insulation_extensive=None,
                              scale_insulation=None, energy_prices=None, rational_hidden_cost=None,
                              number_firms_insulation=None, number_firms_heater=None, hi_threshold=None):
        """Function calibrating buildings object with exogenous data.


        Parameters
        ----------
        coefficient_global: float
        coefficient_backup: Series
        constant_heater: Series
        scale_heater: float
        constant_insulation_intensive: Series
        constant_insulation_extensive: Series
        scale_insulation: float
        energy_prices: Series
            Energy prices for year y. Index are energy carriers {'Electricity', 'Natural gas', 'Oil fuel', 'Wood fuel'}.
        rational_hidden_cost:
        """

        # calibration energy consumption first year
        if (coefficient_global is None) and (energy_prices is not None):
            self.calibration_consumption(energy_prices.loc[self.first_year, :], None)
        else:
            self.coefficient_global = coefficient_global
            self.coefficient_backup = coefficient_backup

        if rational_hidden_cost is not None:
            self.rational_hidden_cost = rational_hidden_cost

        else:
            if constant_heater is not None:
                self.constant_heater = constant_heater

            if constant_insulation_intensive is not None:
                self.constant_insulation_intensive = constant_insulation_intensive.dropna()

            if constant_insulation_extensive is not None:
                self.constant_insulation_extensive = constant_insulation_extensive.dropna()

            if scale_heater is not None:
                self.apply_scale(scale_heater, gest='heater')

            if scale_insulation is not None:
                self.apply_scale(scale_insulation, gest='insulation')

        self.hi_threshold = hi_threshold
        self.number_firms_insulation, self.number_firms_heater = number_firms_insulation, number_firms_heater

    def remove_calibration(self):

        self.coefficient_global = None
        self.coefficient_backup = None

        self.preferences_insulation['subsidy'] /= self.scale_insulation
        self.preferences_insulation['cost'] /= self.scale_insulation
        self.preferences_insulation['bill_saved'] /= self.scale_insulation

        self.constant_heater = None
        self.constant_insulation_intensive = None
        self.constant_insulation_extensive = None
        self.scale_insulation = None

        self.rational_hidden_cost = None

    def flow_demolition(self, demolition_rate, step=1):
        """Demolition of E, F and G buildings based on their share in the mobile stock.

        Returns
        -------
        Series
        """
        self.logger.info('Demolition')
        if isinstance(demolition_rate, Series):
            demolition_rate = demolition_rate.loc[self.year]
        demolition_total = (demolition_rate * self.stock_mobile).sum() * step

        stock_demolition = self.stock_mobile[self.certificate.isin(self._target_demolition)]

        if stock_demolition.sum() < demolition_total:
            self._target_demolition = ['G', 'F', 'E', 'D']
            stock_demolition = self.stock_mobile[self.certificate.isin(self._target_demolition)]
            if stock_demolition.sum() < demolition_total:
                self._target_demolition = ['G', 'F', 'E', 'D', 'C']
                stock_demolition = self.stock_mobile[self.certificate.isin(self._target_demolition)]

        stock_demolition = stock_demolition / stock_demolition.sum()
        flow_demolition = (stock_demolition * demolition_total).dropna()
        return flow_demolition.reorder_levels(self.stock.index.names)

    def health_cost(self, health_cost_dpe, health_cost_income, prices, stock=None, method_health_cost=None):

        if method_health_cost is None:
            method_health_cost = self.method_health_cost

        if stock is None:
            stock = self.stock
        if method_health_cost == 'epc':

            _, certificate, _ = self.consumption_heating(method='3uses', full_output=True)
            temp = concat((stock, reindex_mi(certificate, stock.index).rename('Performance')), axis=1)
            temp.set_index('Performance', append=True, inplace=True)
            temp = temp.squeeze()
            return (temp * reindex_mi(health_cost_dpe, temp.index)).sum() / 10 ** 9

        elif method_health_cost == 'heating_intensity':

            heating_intensity = self.to_heating_intensity(stock.index, prices)
            stock = concat((stock, heating_intensity), axis=1, keys=['Stock', 'Heating intensity'])
            stock_health = stock.loc[stock['Heating intensity'] <= self.hi_threshold, 'Stock']

            return (health_cost_income * stock_health).sum() / 10 ** 9

    def calculate_renovation(self, prices, cost_insulation, cost_heater, carbon_content, health_cost=None,
                             carbon_value=50):
        """Calculate the economics renovation of the building stock.


        Parameters
        ----------
        prices: Series
            Energy prices.
        cost_insulation: Series
            Cost of insulation measures.
        cost_heater: Series
            Cost of heater measures.
        carbon_content: Series
            Carbon content of energy carriers.
        health_cost: Series
            Health cost of energy carriers.
        carbon_value: float
            Carbon value.
        """
        # select stock
        stock = self.add_certificate(self.stock)
        stock = stock[stock.index.get_level_values('Performance').astype(str) > 'B']
        stock = stock.droplevel('Performance')
        index = stock.index

        # calculate consumption before renovation and heater replacement
        consumption_before = self.consumption_heating_store(index, full_output=False)
        consumption_before = reindex_mi(consumption_before, index)
        temp = consumption_before * reindex_mi(self._surface, consumption_before.index)
        heating_intensity_before = self._to_heating_intensity(temp.index, prices,
                                                             consumption=temp,
                                                             level_heater='Heating system')
        consumption_before *= heating_intensity_before

        c_content = carbon_content.reindex(self.to_energy(consumption_before)).set_axis(consumption_before.index)
        emission_before = consumption_before * c_content / 1e6

        # calculate consumption after renovation and heater replacement
        s_no_switch = concat((stock, Series(stock.index.get_level_values('Heating system'), index=stock.index,
                              name='Heating system final')), axis=1).set_index('Heating system final', append=True).squeeze()
        idx_fossil = ['Natural gas', 'Oil fuel']
        s_switch = stock[self.to_energy(stock).isin(idx_fossil)]
        s_switch = concat((s_switch, Series('Electricity-Heat pump water', index=s_switch.index,
                                            name='Heating system final')), axis=1).set_index('Heating system final',
                                                                                             append=True).squeeze()
        s = concat((s_no_switch, s_switch), axis=0, keys=[False, True], names=['Heater replacement'])

        consumption_after, _, certificate_after = self.prepare_consumption(self._choice_insulation,
                                                                           index=s.index,
                                                                           level_heater='Heating system final',
                                                                           full_output=True)
        consumption_after = reindex_mi(consumption_after, s.index)

        _, _, certificate_after_3uses = self.prepare_consumption(self._choice_insulation,
                                                                 index=s.index,
                                                                 level_heater='Heating system final',
                                                                 full_output=True,
                                                                 method_epc='3uses')

        temp = (consumption_after.T * reindex_mi(self._surface, consumption_after.index)).T
        heating_intensity_after = self._to_heating_intensity(temp.index, prices,
                                                            consumption=temp,
                                                            level_heater='Heating system final')
        consumption_after *= heating_intensity_after

        c_content = carbon_content.reindex(self.to_energy(consumption_after, level_heater='Heating system final')).set_axis(consumption_after.index)
        emission_after = (consumption_after.T * c_content).T / 1e6
        consumption_after = consumption_after.droplevel('Heating system final').unstack('Heater replacement')
        emission_after = emission_after.droplevel('Heating system final').unstack('Heater replacement')

        # calculate consumptionn emission savingand cost due to renovation and heater replacement
        consumption_before = reindex_mi(consumption_before, consumption_after.index)
        consumption_saved = (consumption_before - consumption_after.T).T
        consumption_saved = (consumption_saved.T * reindex_mi(self._surface, consumption_saved.index)).T

        emission_before = reindex_mi(emission_before, consumption_after.index)
        emission_saved = (emission_before - emission_after.T).T
        emission_saved = (emission_saved.T * reindex_mi(self._surface, emission_saved.index)).T

        emission_saved_value = emission_saved * carbon_value
        emission_saved *= 1e6

        bill_before = self.energy_bill(prices, consumption_before * reindex_mi(self._surface, consumption_saved.index))
        bill_after = self.energy_bill(prices, (consumption_after.T * reindex_mi(self._surface, consumption_saved.index)).T)
        bill_saved = (bill_before - bill_after.T).T

        certificate_after = reindex_mi(certificate_after, s.index)
        certificate_after = certificate_after.droplevel('Heating system final').unstack('Heater replacement')

        certificate_after_3uses = reindex_mi(certificate_after_3uses, s.index)
        certificate_after_3uses = certificate_after_3uses.droplevel('Heating system final').unstack('Heater replacement')

        cost = self.prepare_cost_insulation(cost_insulation * self.surface_insulation)
        cost = cost.T.multiply(self._surface, level='Housing type').T

        cost = concat((cost, cost + cost_heater.loc['Electricity-Heat pump water']), axis=1, keys=[False, True],
                      names=['Heater replacement'])
        cost = cost.reorder_levels(consumption_saved.columns.names, axis=1)
        cost = reindex_mi(cost, consumption_saved.index)

        health_cost_saved = None
        if health_cost is not None:
            _, certificate_before, _ = self.consumption_heating(index=index, method='3uses', full_output=True)
            df = concat((certificate_before, stock.loc[index]), keys=['Performance', 'Stock'], axis=1).dropna().set_index(
                'Performance', append=True).squeeze()
            health_cost_before = reindex_mi(health_cost, df.index).droplevel('Performance').fillna(0)

            c = reindex_mi(certificate_after_3uses, index)
            health_cost_after = {}
            for i, j in c.items():
                df = concat((j, stock.loc[index]), keys=['Performance', 'Stock'], axis=1).dropna().set_index(
                    'Performance', append=True).squeeze()
                health_cost_after.update({i: reindex_mi(health_cost, df.index).droplevel('Performance')})
            health_cost_after = DataFrame(health_cost_after).fillna(0)

            health_cost_saved = (health_cost_before - health_cost_after.T).T
            health_cost_saved.columns.names = certificate_after_3uses.columns.names

        output = {
            'consumption_saved': consumption_saved,
            'emission_saved': emission_saved,
            'emission_saved_value': emission_saved_value,
            'cost': cost,
            'bill_saved': bill_saved,
            'certificate_after': certificate_after,
            'health_cost_saved': health_cost_saved,
            'stock': stock
        }
        return output

    def select_renovation_option(self, consumption_saved, emission_saved, cost_investment, stock,
                                 bill_saved, certificate_after, carbon_saved=None,
                                 health_cost_saved=None, lifetime=30, discount_rate=0.05,
                                 measures='deep_renovation',
                                 include_energy=True, include_carbon=False, include_health=False,
                                 include_failures=False, cost_failures=None,
                                 include_failures_landlords=False, cost_failures_landlords=None,
                                 include_failures_multi=False, cost_failures_multi=None,
                                 include_credit_constraint=None, income=None, ratio_threshold=0.05,
                                 duration_loan=10):
        """Select renovation option based on measures.

        Parameters
        ----------
        consumption_saved: Series
            Energy consumption saved by renovation measures.
        emission_saved: Series
            Emission saved by renovation measures.
        cost_investment: Series
            Cost of insulation measures.
        stock: Series
            Stock of buildings.
        certificate_after: Series
            Energy performance certificate after renovation measures.
        lifetime: int, default 30
            Lifetime of insulation measures.
        discount_rate: float, default 0.05
            Discount rate.
        measures: str, default 'deep_renovation'
            Type of insulation measures.
        carbon_saved: Series, default None
            Carbon saved by renovation measures.
        bill_saved: Series, default None
            Bill saved by renovation measures.
        health_cost_saved: Series, default None
            Health cost saved by renovation measures.
        include_energy: bool, default True
            Include energy in the calculation of indicators.
        include_carbon: bool, default False
            Include carbon in the calculation of indicators.
        include_health: bool, default False
            Include health in the calculation of indicators.
        """

        # calculate the discount factor
        discount_factor = (1 - (1 + discount_rate) ** -lifetime) / discount_rate
        if isinstance(discount_rate, Series):
            discount_rate = reindex_mi(discount_rate, cost_investment.index)
            discount_rate = concat([discount_rate] * cost_investment.shape[1], axis=1, keys=cost_investment.columns)

        # format the consumption saved
        cost_annualized = calculate_annuities(cost_investment, lifetime=lifetime, discount_rate=discount_rate)
        consumption_saved = reindex_mi(consumption_saved, stock.index).dropna(how='all')
        emission_saved = reindex_mi(emission_saved, stock.index).dropna(how='all')
        bill_saved = reindex_mi(bill_saved, stock.index).dropna(how='all')
        if include_health is True:
            health_cost_saved = reindex_mi(health_cost_saved, stock.index).dropna(how='all')

        index = consumption_saved.index
        cost_annualized = reindex_mi(cost_annualized, index)
        cost_investment = reindex_mi(cost_investment, index)
        discount_factor = reindex_mi(discount_factor, index)

        rslt_npv, rslt_roi = None, None
        if measures in ['global_insulation', 'global_renovation']:
            INSULATION = {'Wall': True, 'Floor': True, 'Roof': True, 'Windows': True}
            insulation = deepcopy(INSULATION)
            if measures == 'global_renovation':
                insulation.update({'Heater replacement': True})
            else:
                insulation.update({'Heater replacement': False})
            insulation = MultiIndex.from_frame(Series(insulation).to_frame().T)
            consumption_saved_efficiency = consumption_saved.loc[:, insulation]
            emission_saved_efficiency = emission_saved.loc[:, insulation]

            cost_annualized = cost_annualized.loc[:, insulation]

            if include_carbon is True:
                carbon_saved = reindex_mi(carbon_saved, index)
                carbon_saved = carbon_saved.loc[:, insulation]
                cost_annualized -= carbon_saved

            if include_health is True:
                health_cost_saved = health_cost_saved.loc[:, insulation]
                cost_annualized -= health_cost_saved

            cost_efficiency = cost_annualized / consumption_saved_efficiency
            cost_efficiency_carbon = cost_annualized / emission_saved_efficiency

            consumption_saved_efficiency = (stock * consumption_saved_efficiency.T).T
            consumption_saved_efficiency = consumption_saved_efficiency.stack(consumption_saved_efficiency.columns.names).squeeze()

            emission_saved_efficiency = (stock * emission_saved_efficiency.T).T
            emission_saved_efficiency = emission_saved_efficiency.stack(emission_saved_efficiency.columns.names).squeeze()

            cost_efficiency = cost_efficiency.stack(cost_efficiency.columns.names).squeeze()
            cost_efficiency_carbon = cost_efficiency_carbon.stack(cost_efficiency_carbon.columns.names).squeeze()

        elif measures in ['deep_renovation', 'deep_insulation']:

            # prepare information
            dict_df = {'consumption_saved': consumption_saved,
                       'emission_saved': emission_saved,
                       'cost_investment': cost_investment,
                       'bill_saved': bill_saved
                       }
            cash_flow = bill_saved.copy()

            if include_carbon is True:
                carbon_saved = reindex_mi(carbon_saved, index)
                cost_annualized = cost_annualized - carbon_saved
                cash_flow += carbon_saved
                dict_df.update({'carbon_saved': carbon_saved})

            if include_health is True:
                cost_annualized -= health_cost_saved
                cash_flow += health_cost_saved
                dict_df.update({'health_cost_saved': health_cost_saved})

            dict_df.update({'cash_flow': cash_flow})

            # filter renovation work
            c_after = certificate_after.copy()
            if measures == 'deep_insulation':
                c_after.loc[:, c_after.columns.get_level_values('Heater replacement')] = float('nan')
                c_after = c_after.dropna(how='all', axis=1)

            c_after = c_after.reindex(stock.index)
            c_after.dropna(how='all', inplace=True)
            condition = self.select_deep_renovation(c_after)
            condition = condition.replace(False, float('nan'))

            # credit constraint
            if include_credit_constraint is True and income is not None:
                cost_year = cost_investment / duration_loan
                income.rename_axis(['Income owner'], inplace=True)
                income = reindex_mi(income, cost_year.index)
                ratio = (cost_year.T / income).T
                condition_ratio = (ratio < ratio_threshold).replace(False, float('nan'))
                condition *= condition_ratio

            # calculate cost-efficiency by anualizing investment cost
            if include_failures is True and cost_failures is not None:
                if isinstance(cost_failures, Series):
                    cost_failures_annualized = reindex_mi(cost_failures, cost_annualized.index).fillna(0)
                    cost_failures_annualized = concat([cost_failures_annualized] * discount_rate.shape[1], axis=1, keys=discount_rate.columns)
                    cost_failures_annualized = calculate_annuities(cost_failures_annualized, lifetime=lifetime, discount_rate=discount_rate)
                    cost_annualized = cost_annualized + cost_failures_annualized

            if include_failures_landlords is True and cost_failures_landlords is not None:
                if isinstance(cost_failures_landlords, Series):
                    cost_failures_landlords_annualized = reindex_mi(cost_failures_landlords, cash_flow.index).fillna(0)
                    cost_failures_landlords_annualized = concat([cost_failures_landlords_annualized] * discount_rate.shape[1], axis=1, keys=discount_rate.columns)
                    cost_failures_landlords_annualized = calculate_annuities(cost_failures_landlords_annualized, lifetime=lifetime, discount_rate=discount_rate)
                    cost_annualized = cost_annualized + cost_failures_landlords_annualized

            if include_failures_multi is True and cost_failures_multi is not None:
                if isinstance(cost_failures_multi, Series):
                    cost_failures_multi_annualized = reindex_mi(cost_failures_multi, cash_flow.index).fillna(0)
                    cost_failures_multi_annualized = concat([cost_failures_multi_annualized] * discount_rate.shape[1], axis=1, keys=discount_rate.columns)
                    cost_failures_multi_annualized = calculate_annuities(cost_failures_multi_annualized, lifetime=lifetime, discount_rate=discount_rate)
                    cost_annualized = cost_annualized + cost_failures_multi_annualized

            if include_energy is True:
                cost_net = cost_annualized - cash_flow
            else:
                cost_net = cost_annualized

            # calculate cost-efficiency of renovation works
            cost_efficiency = cost_net / consumption_saved
            cost_efficiency *= reindex_mi(condition, cost_efficiency.index)
            cost_efficiency = cost_efficiency.apply(pd.to_numeric)
            cost_efficiency.dropna(how='all', inplace=True)
            rslt_efficiency = self.find_best_option(cost_efficiency, dict_df=dict_df, func='min')
            rslt_efficiency['stock'] = stock
            rslt_efficiency['consumption_saved_efficiency'] = rslt_efficiency['stock'] * rslt_efficiency['consumption_saved']

            # calculate cost-efficiency of avoiding emission of renovation works
            cost_efficiency_carbon = cost_net / emission_saved
            cost_efficiency_carbon *= reindex_mi(condition, cost_efficiency_carbon.index)
            cost_efficiency_carbon = cost_efficiency_carbon.apply(pd.to_numeric)
            cost_efficiency_carbon.dropna(how='all', inplace=True)
            rslt_efficiency_carbon = self.find_best_option(cost_efficiency_carbon, dict_df=dict_df, func='min')
            rslt_efficiency_carbon['stock'] = stock
            rslt_efficiency_carbon['emission_saved_efficiency'] = rslt_efficiency_carbon['stock'] * rslt_efficiency['emission_saved']

            # calculate net present value of renovation project
            temp = deepcopy(dict_df)

            if cost_failures is not None:
                if isinstance(cost_failures, Series):
                    cost_failures = reindex_mi(cost_failures, cash_flow.index).fillna(0)
                else:
                    raise NotImplemented

                if include_failures is True:
                    cost_investment = (cost_investment.T + cost_failures).T

            if cost_failures_landlords is not None:
                if isinstance(cost_failures_landlords, Series):
                    cost_failures_landlords = reindex_mi(cost_failures_landlords, cash_flow.index).fillna(0)
                else:
                    raise NotImplemented
                if include_failures_landlords is True:
                    cost_investment = (cost_investment.T + cost_failures_landlords).T

            if cost_failures_multi is not None:
                if isinstance(cost_failures_multi, Series):
                    cost_failures_multi = reindex_mi(cost_failures_multi, cash_flow.index).fillna(0)
                else:
                    raise NotImplemented
                if include_failures_multi is True:
                    cost_investment = (cost_investment.T + cost_failures_multi).T

            if include_energy is True:
                if isinstance(discount_factor, Series):
                    npv = - cost_investment + (discount_factor * cash_flow.T).T
                else:
                    npv = - cost_investment + discount_factor * cash_flow
            else:
                npv = - cost_investment

            npv *= reindex_mi(condition, npv.index)
            npv = npv.apply(pd.to_numeric)
            npv.dropna(how='all', inplace=True)
            rslt_npv = self.find_best_option(npv, dict_df=temp, func='max')

            # add information in dict result
            rslt_npv['cost_social'] = - rslt_npv['criteria']
            rslt_npv['stock'] = stock
            rslt_npv['consumption_saved_npv'] = stock * rslt_npv['consumption_saved']
            rslt_npv['emission_saved_npv'] = stock * rslt_npv['emission_saved']
            if cost_failures is not None:
                rslt_npv['cost_failures'] = cost_failures
            if cost_failures_landlords is not None:
                rslt_npv['cost_failures_landlords'] = cost_failures_landlords
            if cost_failures_multi is not None:
                rslt_npv['cost_failures_multi'] = cost_failures_multi

            # calculate roi
            if include_energy is True:
                roi = cost_investment / cash_flow
                roi *= reindex_mi(condition, roi.index)
                roi = roi.apply(pd.to_numeric)
                roi.dropna(how='all', inplace=True)
                rslt_roi = self.find_best_option(roi, dict_df=dict_df, func='min')
                rslt_roi['stock'] = stock
                rslt_roi['consumption_saved_roi'] = stock * rslt_roi['consumption_saved']

        else:
            raise NotImplemented

        output = {
            'rslt_efficiency': rslt_efficiency,
            'rslt_efficiency_carbon': rslt_efficiency_carbon,
            'rslt_npv': rslt_npv,
            'rslt_roi': rslt_roi
        }
        return output

    def make_static_analysis(self, cost_insulation, cost_heater, prices,
                             health_cost, carbon_content,
                             path_out=None, carbon_value=50, selected_options=None, sufix='',
                             social_discount_rate=0.05, private_discount_rate=0.05, lifetime=25,
                             make_plot=False, cost_failures=None, cost_failures_landlords=None, cost_failures_multi=None,
                             income=None, duration_loan=10):

        def closest(df, value):
            return (df - value).abs().sort_values().index[0]
        # select only stock mobile and existing before the first year
        if path_out is None:
            path_out = self.path_ini

        # get all economics information for all renovation options
        output = self.calculate_renovation(prices, cost_insulation, cost_heater, carbon_content, health_cost=health_cost,
                                           carbon_value=carbon_value)

        options = get_json('project/input/resources_dir/scenario_static.json')
        colors = get_json('project/input/resources_dir/colors_scenario_static.json')

        for k, i in options.items():
            if i['discount_rate'] == 'social_discount_rate':
                i['discount_rate'] = social_discount_rate
            if i['discount_rate'] == 'private_discount_rate':
                i['discount_rate'] = private_discount_rate

        if selected_options is None:
            selected_options = ['Deep insulation, private',
                                'Deep renovation, social',
                                'Deep renovation, private']

        options = {k: i for k, i in options.items() if
                   k in selected_options}

        dict_rslt, dict_subsidies = {}, {}
        for key, option in options.items():
            # select options for the analysis and calculate main indicators
            output_select = self.select_renovation_option(
                output['consumption_saved'],
                output['emission_saved'],
                output['cost'],
                output['stock'],
                output['bill_saved'],
                output['certificate_after'],
                carbon_saved=output['emission_saved_value'],
                health_cost_saved=output['health_cost_saved'],
                lifetime=lifetime,
                cost_failures=cost_failures,
                cost_failures_landlords=cost_failures_landlords,
                cost_failures_multi=cost_failures_multi,
                income=income,
                duration_loan=duration_loan,
                **option)

            # format the results to plot monotonic curve
            temp = self.prepare_abatement_plot(
                output_select['rslt_efficiency'],
                output_select['rslt_efficiency_carbon'],
                output_select['rslt_npv'],
                output_select['rslt_roi'])

            dict_rslt.update({key: temp})

            if 'I-E-C-S' in key:
                names = output['consumption_saved'].columns.names

                replace_columns = {
                    'consumption_saved': "Économie d'énergie (kWh/an.logement)",
                    'emission_saved': "Économie d'émission (gCO2/an.logement)",
                    'cost_investment': "Coût d'investissement (€/logement)",
                    'bill_saved': "Économie de facture (€/(an.logement))",
                    'carbon_saved': "Économie de dommage carbone (€/(an.logement))",
                    'health_cost_saved': "Économie de coût de santé (€/(an.logement))",
                    'cash_flow': "Cash flow champs social (€/(an.logement))",
                    'cost_social': "Coût actualisé social (€/logement)",
                    'stock': "Stock de résidences principales (unité)",
                    'consumption_saved_npv': "Économie d'énergie (kWh/an)",
                    'emission_saved_npv': "Économie d'émission (gCO2/an)",
                    'cost_failures': "Coût des frictions en forme réduite (€/logement)"
                }
                output_select['rslt_npv'].rename(columns=replace_columns, inplace=True)

                # Transforming the index levels into a single level
                output_select['rslt_npv']['Measures'] = output_select['rslt_npv']['columns'].apply(lambda row: ", ".join([name for name, value in zip(names, row) if value]))
                dict_subsidies.update({key: output_select['rslt_npv']})

        dict_rslt = reverse_dict(dict_rslt)

        # format the results based on initial value
        c_before = self.consumption_heating_store(self._stock_ref.index, full_output=False)
        c_before = reindex_mi(c_before, self._stock_ref.index) * self._stock_ref * reindex_mi(self._surface,
                                                                                              self._stock_ref.index)
        heating_intensity_before = self._to_heating_intensity(c_before.index, prices)
        c_before *= heating_intensity_before

        c_content = carbon_content.reindex(self.to_energy(c_before)).set_axis(c_before.index)
        e_before = (c_before * c_content).sum()

        stock_ini = self._stock_ref.sum() / 1e6

        # percentage of initial value
        for key in dict_rslt.keys():
            if key.split('_')[-1] == 'emission':
                for k in dict_rslt[key].keys():
                    dict_rslt[key][k].index = dict_rslt[key][k].index * 1e9 / e_before
                    # dict_rslt[key][k].name = 'Emission saving (% / 2018)'
                    dict_rslt[key][k].index.rename('Emission saved (% / 2018)', inplace=True)
                    dict_rslt[key][k].index.rename('Emission évitée (/2018)', inplace=True)
            elif key.split('_')[-1] == 'pop':
                for k in dict_rslt[key].keys():
                    dict_rslt[key][k].index = dict_rslt[key][k].index / stock_ini
                    # dict_rslt[key][k].name = 'Stock de résidences principales (% / 2018)'
                    dict_rslt[key][k].index.rename('Stock de résidences principales (% / 2018)', inplace=True)
                    dict_rslt[key][k].index.rename('Stock de résidences principales (/2018)', inplace=True)
            elif key.split('_')[-1] == 'energy':
                for k in dict_rslt[key].keys():
                    dict_rslt[key][k].index = dict_rslt[key][k].index / (c_before.sum() / 10 ** 9)
                    # dict_rslt[key][k].name = 'Consumption saving (% / 2018)'
                    dict_rslt[key][k].index.rename('Consumption saved (% / 2018)', inplace=True)
                    dict_rslt[key][k].index.rename("Économie d'énergie (/2018)", inplace=True)
            else:
                raise
        rslt_profitability = {}
        list_indicators = ['npv_energy', 'npv_emission', 'npv_pop']
        for indicator in list_indicators:
            rslt_profitability.update({indicator: {}})
            for k, i in dict_rslt[indicator].items():
                rslt_profitability[indicator].update({k: closest(i, 0)})

        # make plots
        if make_plot is True:
            temp = {k: i for k, i in dict_rslt['efficiency_energy'].items() if k.split(',')[0] in ['I', 'I-C', 'I-C-S']}
            if temp != {}:
                make_plots(temp, "Coût d'abattement énergie (€ par kWh cumac)",
                           save=os.path.join(path_out, 'cout_abattement_energie{}.png'.format(sufix)), ymax=0.3, ymin=0,
                           format_y=lambda y, _: '{:.2f}€/kWh'.format(y), loc='left', left=1.25, colors=colors,
                           format_x=lambda x, _: '{:.0%}'.format(x), order_legend=False, x_tick_interval=0.1,
                           xmin=0, xmax=1, ncol=1, export_csv=True)

            temp = {k: i for k, i in dict_rslt['efficiency_pop'].items() if k.split(',')[0] in ['I', 'I-C', 'I-C-S']}
            if temp != {}:
                make_plots(temp, "Coût d'abattement énergie (€ par kWh cumac)",
                           save=os.path.join(path_out, 'cout_abattement_logement{}.png'.format(sufix)), ymax=0.3, ymin=0,
                           format_y=lambda y, _: '{:.2f}€/kWh'.format(y), loc='left', left=1.25, colors=colors,
                           format_x=lambda x, _: '{:.0%}'.format(x), order_legend=False, x_tick_interval=0.1,
                           xmin=0, xmax=1, ncol=1, export_csv=True)

            efficiency_scenarios = ['I-E-S', 'I-E social discount', 'I-E', 'I-E-MF Bailleurs',
                                    'I-E-MF Bailleurs + Collectifs', 'I-E-MF Contrainte crédit']

            temp = {k: i for k, i in dict_rslt['efficiency_carbon_emission'].items() if k.split(',')[0] in efficiency_scenarios}
            if temp != {}:
                make_plots(temp, "Coût d'abattement émission (€ par tCO2 évitée)",
                           save=os.path.join(path_out, 'cout_abattement_emission{}.png'.format(sufix)), ymax=1000, ymin=0,
                           format_y=lambda y, _: '{:.0f}€/tCO2'.format(y), loc='left', left=1.25, colors=colors,
                           format_x=lambda x, _: '{:.0%}'.format(x), order_legend=efficiency_scenarios, x_tick_interval=0.1,
                           xmin=0, xmax=1, ncol=1, hlines=250, export_csv=True)

            temp = {k: i for k, i in dict_rslt['efficiency_carbon_pop'].items() if k.split(',')[0] in efficiency_scenarios}
            if temp != {}:
                make_plots(temp, "Coût d'abattement émission (€ par tCO2 évitée)",
                           save=os.path.join(path_out, 'cout_abattement_logement{}.png'.format(sufix)), ymax=1000, ymin=0,
                           format_y=lambda y, _: '{:.0f}€/tCO2'.format(y), loc='left', left=1.25, colors=colors,
                           format_x=lambda x, _: '{:.0%}'.format(x), order_legend=efficiency_scenarios, x_tick_interval=0.1,
                           xmin=0, xmax=1, ncol=1, hlines=250, export_csv=True)

            npv_scenarios = ['I-E-C-S', 'I-E-C', 'I-E social discount', 'I-E', 'I-E-MF Bailleurs',
                             'I-E-MF Bailleurs + Collectifs', 'I-E-MF Contrainte crédit']

            temp = {k.split(',')[0]: i for k, i in dict_rslt['npv_energy'].items() if k.split(',')[0] in npv_scenarios}

            if temp != {}:
                make_plots(temp, 'Coût actualisé (Milliers €)',
                           save=os.path.join(path_out, 'van_energie{}.png'.format(sufix)),
                           format_y=lambda y, _: '{:.0f}k€'.format(y / 10 ** 3), loc='left', left=1.25, colors=colors,
                           format_x=lambda x, _: '{:.0%}'.format(x), hlines=0, order_legend=npv_scenarios, x_tick_interval=0.1,
                           ymax=40*1e3, ymin=-20*1e3, xmin=0, xmax=1, ncol=1, export_csv=True)

            temp = {k: i for k, i in dict_rslt['npv_pop'].items() if k.split(',')[0] in npv_scenarios}
            if temp != {}:
                make_plots(temp, 'Coût actualisé (Milliers €)',
                           save=os.path.join(path_out, 'van_logement{}.png'.format(sufix)),
                           format_y=lambda y, _: '{:.0f}k€'.format(y / 10 ** 3), loc='left', left=1.25, colors=colors,
                           format_x=lambda x, _: '{:.0%}'.format(x), hlines=0, order_legend=False, x_tick_interval=0.1,
                           ymax=40*1e3, ymin=-20*1e3, xmin=0, xmax=1, ncol=1, export_csv=True)

            temp = {k: i for k, i in dict_rslt['npv_emission'].items() if k.split(',')[0] in npv_scenarios}
            if temp != {}:
                make_plots(temp, 'Coût actualisé (Milliers €)',
                           save=os.path.join(path_out, 'van_emission{}.png'.format(sufix)),
                           format_y=lambda y, _: '{:.0f}k€'.format(y / 10 ** 3), loc='left', left=1.25, colors=colors,
                           format_x=lambda x, _: '{:.0%}'.format(x), hlines=0, order_legend=False, x_tick_interval=0.1,
                           ymax=40*1e3, ymin=-20*1e3, xmin=0, xmax=1, ncol=1, export_csv=True)

        if False:
            make_plots(dict_rslt['efficiency'], 'Cost efficiency (euro per kWh saved)',
                       save=os.path.join(path_out, 'abatement_curve.png'), ymax=1, ymin=-1,
                       format_y=lambda y, _: '{:.2f}'.format(y), loc='left', left=1.25, colors=colors,
                       format_x=lambda x, _: '{:.0%}'.format(x), order_legend=False)

            make_plots(dict_rslt['efficiency'], 'Cost efficiency (euro per kWh saved)',
                       save=os.path.join(path_out, 'abatement_curve_zoom.png'), ymax=0.3, ymin=0,
                       format_y=lambda y, _: '{:.2f}'.format(y), loc='left', left=1.25, colors=colors,
                       format_x=lambda x, _: '{:.0%}'.format(x), order_legend=False)

            make_plots(dict_rslt['efficiency_carbon'], 'Cost efficiency (euro per tCO2 saved)',
                       save=os.path.join(path_out, 'abatement_curve_carbon.png'), ymax=5000, ymin=-1000,
                       format_y=lambda y, _: '{:.0f}'.format(y), loc='left', left=1.25, colors=colors,
                       format_x=lambda x, _: '{:.0%}'.format(x), order_legend=False)

            make_plots(dict_rslt['efficiency_carbon'], 'Cost efficiency (euro per tCO2 saved)',
                       save=os.path.join(path_out, 'abatement_curve_carbon_zoom.png'), ymax=500, ymin=0,
                       format_y=lambda y, _: '{:.0f}'.format(y), loc='left', left=1.25, colors=colors,
                       format_x=lambda x, _: '{:.0%}'.format(x), order_legend=False)

            make_plots(dict_rslt['npv'], 'Net present cost (Thousand euro)',
                       save=os.path.join(path_out, 'npv_insulation_curve.png'),
                       format_y=lambda y, _: '{:.0f}'.format(y / 10 ** 3), loc='left', left=1.25, colors=colors,
                       format_x=lambda x, _: '{:.0%}'.format(x), ymin=None, hlines=0, order_legend=False)

            make_plots(dict_rslt['roi'], 'Return Time on Investment (years)',
                       save=os.path.join(path_out, 'roi_insulation_curve.png'),
                       format_y=lambda y, _: '{:.0f}'.format(y), loc='left', left=1.25, colors=colors,
                       format_x=lambda x, _: '{:.0%}'.format(x), ymax=50, order_legend=False)

        return rslt_profitability, dict_subsidies

    @staticmethod
    def prepare_abatement_plot(rslt_efficiency, rslt_efficiency_carbon, rslt_npv=None, rslt_roi=None):

        def sort_cum(df, round=3):
            df.sort_values(y.name, inplace=True)
            df.dropna(inplace=True)
            df[y.name] = df[y.name].round(round)
            df = df.groupby([y.name]).agg({x.name: 'sum', y.name: 'first'})
            df[x.name] = df[x.name].cumsum() / 10 ** 9
            df = df.set_index(x.name)[y.name]
            return df


        output = {}
        x = rslt_efficiency['consumption_saved_efficiency'].rename('Consumption saved (TWh)')
        y = rslt_efficiency['criteria'].rename('Cost efficiency (euro/kWh)')
        df_efficiency = concat((x, y), axis=1)
        df_efficiency = sort_cum(df_efficiency, round=3)
        output.update({'efficiency_energy': df_efficiency})

        x = rslt_efficiency['stock'].rename('Stock (Million)')
        y = rslt_efficiency['criteria'].rename('Cost efficiency (euro/kWh)')
        df_efficiency_pop = concat((x, y), axis=1)
        df_efficiency_pop = sort_cum(df_efficiency_pop, round=3)
        df_efficiency_pop.index = df_efficiency_pop.index * 10 ** 3
        output.update({'efficiency_pop': df_efficiency_pop})

        x = rslt_efficiency_carbon['emission_saved_efficiency'].rename('Emission saved (MtCO2)')
        y = rslt_efficiency_carbon['criteria'].rename('Cost efficiency (euro/tCO2)') * 10 ** 6
        df_efficiency_carbon = concat((x, y), axis=1)
        df_efficiency_carbon = sort_cum(df_efficiency_carbon, round=0)
        output.update({'efficiency_carbon_emission': df_efficiency_carbon})

        x = rslt_efficiency_carbon['stock'].rename('Stock (Million)')
        y = rslt_efficiency_carbon['criteria'].rename('Cost efficiency (euro/tCO2)') * 10 ** 6
        df_efficiency_carbon_pop = concat((x, y), axis=1)
        df_efficiency_carbon_pop = sort_cum(df_efficiency_carbon_pop, round=0)
        df_efficiency_carbon_pop.index = df_efficiency_carbon_pop.index * 10 ** 3
        output.update({'efficiency_carbon_pop': df_efficiency_carbon_pop})

        # sort by marginal cost
        if rslt_npv is not None:
            x = rslt_npv['consumption_saved_npv'].rename('Consumption saved (TWh)')
            y = rslt_npv['cost_social'].rename('Net present cost (euro)')
            df_npv = concat((x, y), axis=1)
            df_npv.dropna(inplace=True)
            df_npv = df_npv[df_npv['Consumption saved (TWh)'] > 0]
            df_npv = sort_cum(df_npv, round=0)
            output.update({'npv_energy': df_npv})

            x = rslt_npv['emission_saved_npv'].rename('Emission saved (MtCO2)')
            y = rslt_npv['cost_social'].rename('Net present cost (euro)')
            df_npv = concat((x, y), axis=1)
            df_npv.dropna(inplace=True)
            df_npv = df_npv[df_npv['Emission saved (MtCO2)'] > 0]
            df_npv = sort_cum(df_npv, round=0)
            output.update({'npv_emission': df_npv})

            x = rslt_npv['stock'].rename('Stock (Million)')
            y = rslt_npv['cost_social'].rename('Net present cost(euro)')
            df_npv = concat((x, y), axis=1)
            df_npv = sort_cum(df_npv, round=0)
            df_npv.index = df_npv.index * 10 ** 3
            output.update({'npv_pop': df_npv})

        if rslt_roi is not None:
            x = rslt_roi['consumption_saved_roi'].rename('Consumption saved (TWh)')
            y = rslt_roi['criteria'].rename('Return on Investment (ROI) (year)')
            df_roi = concat((x, y), axis=1)
            df_roi = sort_cum(df_roi, round=1)
            output.update({'roi_energy': df_roi})

        return output

