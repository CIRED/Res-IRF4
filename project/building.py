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

import pandas as pd
from pandas import Series, DataFrame, MultiIndex, Index, IndexSlice, concat, to_numeric, unique, read_csv
from numpy import exp, log, zeros, ones, append, arange, array
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import logging

from project.utils import make_plot, format_ax, save_fig, format_legend, reindex_mi, timing, get_pandas
from project.input.resources import resources_data
import project.thermal as thermal

from itertools import product


class ThermalBuildings:
    """ThermalBuildings classes.

    Parameters:
    ----------
    stock: Series
        Building stock.
    surface: Series
        Surface by dwelling type.
    param: dict
        Generic input.
    efficiency: Series
        Heating system efficiency.
    income: Series
    consumption_ini: Series
    path: str
    year: int, default 2018

    Attributes:
    ----------

    heat_consumption_sd : Series
        kWh by segement
    heat_consumption : Series
        kWh by segement

    heat_consumption_calib : Series
    heat_consumption_energy : Series

    heating_intensity_tenant: dict
        Weighted average heating intensity (%) by decile.
    heating_intensity_avg: dict
        Weighted average heating intensity (%).
    energy_poverty: dict
        Number of energy poverty dwelling.

    taxes_expenditure: dict
    energy_expenditure: dict
    taxes_expenditure_details: dict

    """
    def __init__(self, stock, surface, ratio_surface, efficiency, income, consumption_ini, path=None, year=2018,
                 debug_mode=False):

        self.energy_poverty = None
        self.consumption_3uses_building, self.consumption_sd_building, self.certificate_building = Series(
            dtype='float'), Series(dtype='float'), Series(dtype='float')

        self.consumption_sd_building_choice, self.consumption_3uses_building_choice, self.certificate_building_choice = Series(
            dtype='float'), Series(dtype='float'), Series(dtype='float')

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
            self.path_static = os.path.join(path, 'static')
            if not os.path.isdir(self.path_static):
                os.mkdir(self.path_static)

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
        # self.certificate_nb = None

        # TODO only heating_intensity and calculate average in parse_output
        self.heating_intensity_avg = None
        # self.energy_poverty = None
        self.heat_consumption_sd = None
        self.heat_consumption = None
        self.heat_consumption_calib = None
        self.heat_consumption_energy = None
        self.taxes_expenditure = None
        self.energy_expenditure = None
        self.taxes_list = []
        self.taxes_expenditure_details = {}
        self.stock_yrs = {}

        self.stock = stock

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

        Parameters
        ----------
        stock: Series

        Returns
        -------

        """

        self._stock = stock
        self.stock_mobile = stock - self._stock_residual.reindex(stock.index, fill_value=0)
        self.surface = reindex_mi(self._surface, stock.index)
        # self.housing_type = Series(stock.index.get_level_values('Housing type'), index=stock.index)

        heating_system = Series(stock.index.get_level_values('Heating system'), index=stock.index)
        self.energy = heating_system.str.split('-').str[0].rename('Energy')
        self.efficiency = to_numeric(heating_system.replace(self._efficiency))

        self.stock_yrs.update({self.year: self.stock})

        consumption_sd, _, certificate = self.consumption_standard(stock.index)
        self.heat_consumption_sd = self.surface * reindex_mi(consumption_sd, stock.index)
        self.certificate = reindex_mi(certificate, stock.index)

    def simplified_stock(self, energy_level=False):
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
        heating_system

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

    def heating_need(self, hourly=True, climate=None, smooth=False):
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
                                                         infiltration='Medium', climate=climate, hourly=hourly,
                                                         smooth=smooth)

        heating_need = (heating_need.T * self.stock * self.surface).T
        return heating_need

    def heating_consumption(self, hourly=True, climate=None, smooth=False):
        """Calculation consumption standard of the current building stock.

        Parameters
        ----------
        hourly
        climate
        smooth

        Returns
        -------

        """

        idx = self.stock.index
        wall = Series(idx.get_level_values('Wall'), index=idx)
        floor = Series(idx.get_level_values('Floor'), index=idx)
        roof = Series(idx.get_level_values('Roof'), index=idx)
        windows = Series(idx.get_level_values('Windows'), index=idx)
        heating_system = Series(idx.get_level_values('Heating system'), index=idx).astype('object')
        efficiency = to_numeric(heating_system.replace(self._efficiency))
        consumption = thermal.conventional_heating_final(wall, floor, roof, windows, self._ratio_surface.copy(),
                                                         efficiency, climate=climate, hourly=hourly,
                                                         smooth=smooth)
        return consumption


    def consumption_standard(self, indexes, level_heater='Heating system'):
        """Pre-calculate space energy consumption based only on relevant levels.

        Parameters
        ----------
        indexes: MultiIndex, Index
            Index used to estimate consumption standard.
        level_heater: {'Heating system', 'Heating system final'}, default 'Heating system'

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
            wall = Series(idx.get_level_values('Wall'), index=idx)
            floor = Series(idx.get_level_values('Floor'), index=idx)
            roof = Series(idx.get_level_values('Roof'), index=idx)
            windows = Series(idx.get_level_values('Windows'), index=idx)
            heating_system = Series(idx.get_level_values('Heating system'), index=idx).astype('object')
            efficiency = to_numeric(heating_system.replace(self._efficiency))

            consumption = thermal.conventional_heating_final(wall, floor, roof, windows, self._ratio_surface.copy(),
                                                             efficiency)

            certificate, consumption_3uses = thermal.conventional_energy_3uses(wall, floor, roof, windows,
                                                                               self._ratio_surface.copy(),
                                                                               efficiency, idx)

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

    def consumption_actual(self, prices, consumption=None):
        """Space heating consumption based on standard space heating consumption and heating intensity (kWh/a).

        Space heating consumption is in kWh/building.a
        Equation is based on Allibe (2012).

        Parameters
        ----------
        prices: Series

        Returns
        -------
        Series
        """

        if consumption is None:
            consumption = self.heat_consumption_sd.copy()
        else:
            consumption = consumption.copy()

        energy_bill = AgentBuildings.energy_bill(prices, consumption)
        if isinstance(energy_bill, Series):
            budget_share = energy_bill / reindex_mi(self._income_tenant, self.stock.index)
            heating_intensity = -0.191 * budget_share.apply(log) + 0.1105
            consumption *= heating_intensity
            self.heating_intensity_avg = (self.stock * heating_intensity).sum() / self.stock.sum()
            self.energy_poverty = (self.stock[self.stock.index.get_level_values(
                'Income owner') == ('D1' or 'D2' or 'D3')])[budget_share >= 0.08].sum()
        elif isinstance(energy_bill, DataFrame):
            budget_share = (energy_bill.T / reindex_mi(self._income_tenant, self.stock.index)).T
            heating_intensity = -0.191 * budget_share.apply(log) + 0.1105
            consumption = heating_intensity * consumption

        return consumption

    def calculate_consumption(self, prices, taxes):
        """Calculate energy indicators.

        Parameters
        ----------
        prices: Series
        taxes: Series

        Returns
        -------

        """

        self.heat_consumption = self.consumption_actual(prices) * self.stock

        heat_consumption_energy = self.heat_consumption.groupby(self.energy).sum()
        if self.coefficient_consumption is None:

            consumption = concat((self.heat_consumption, self.energy), axis=1).groupby(
                ['Housing type', 'Energy']).sum().iloc[:, 0] / 10**9

            # considering 20% of electricity got wood stove - 50% electricity
            electricity_wood = 0.2 * consumption[('Single-family', 'Electricity')] * 1
            consumption[('Single-family', 'Wood fuel')] += electricity_wood
            consumption[('Single-family', 'Electricity')] -= electricity_wood
            consumption.groupby('Energy').sum()

            self.heat_consumption.groupby('Housing type').sum() / 10**9

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
            temp = concat((self.heat_consumption, self.energy), axis=1).groupby(
                ['Housing type', 'Energy']).sum().iloc[:, 0] / 10**9
            temp.index = temp.index.map(lambda x: 'Consumption {} {} (TWh)'.format(x[0], x[1]))
            validation.update(temp)
            temp = self.heat_consumption.groupby('Housing type').sum() / 10**9
            temp.index = temp.index.map(lambda x: 'Consumption {} (TWh)'.format(x))
            validation.update(temp)
            validation.update({'Consumption (TWh)': self.heat_consumption.sum() / 10**9})

            self.coefficient_consumption = self._consumption_ini * 10**9 / heat_consumption_energy

            temp = self.coefficient_consumption.copy()
            temp.index = temp.index.map(lambda x: 'Coefficient calibration {} (%)'.format(x))
            validation.update(temp)

            temp = heat_consumption_energy / 10**9
            temp.index = temp.index.map(lambda x: 'Consumption {} (TWh)'.format(x))
            validation.update(temp)

            validation = Series(validation)
            if resources_data['data_calibration'] is not None:
                validation = concat((validation, resources_data['data_calibration']), keys=['Calcul', 'Data'], axis=1)
                validation['Error'] = (validation['Calcul'] - validation['Data']) / validation['Data']

            if self.path is not None:
                validation.round(2).to_csv(os.path.join(self.path_calibration, 'validation_stock.csv'))

        coefficient = self.coefficient_consumption.reindex(self.energy).set_axis(self.stock.index, axis=0)
        self.heat_consumption_calib = (coefficient * self.heat_consumption).copy()

        self.heat_consumption_energy = self.heat_consumption_calib.groupby(self.energy).sum()

        prices_reindex = prices.reindex(self.energy).set_axis(self.stock.index, axis=0)
        self.energy_expenditure = prices_reindex * self.heat_consumption_calib

        total_taxes = Series(0, index=prices.index)
        for tax in taxes:
            if self.year in tax.value.index:
                if tax.name not in self.taxes_list:
                    self.taxes_list += [tax.name]
                amount = tax.value.loc[self.year, :] * heat_consumption_energy
                self.taxes_expenditure_details[tax.name] = amount
                total_taxes += amount

        self.taxes_expenditure = total_taxes

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


class AgentBuildings(ThermalBuildings):

    """Class AgentBuildings represents thermal dynamic building stock.


    Attributes
    ----------
    pref_inertia:  float or Series

    pref_investment_insulation: float or Series
    pref_bill_insulation: float or Series
    pref_subsidy_insulation: float or Series


    cost_insulation: DataFrame
        Cost by segment and by insulation choice (€).
    investment_insulation: DataFrame
        Investment realized by segment and by insulation choice (€).
    tax_insulation: DataFrame
        Tax by segment and by insulation choice (€).
    certificate_jump: DataFrame
        Number of jump of energy performance certificate.
    retrofit_rate: dict


    """

    def __init__(self, stock, surface, ratio_surface, efficiency, income, consumption_ini, preferences,
                 performance_insulation, path=None, demolition_rate=0.0, year=2018,
                 endogenous=True, number_exogenous=300000, utility_extensive='market_share',
                 logger=None, debug_mode=False, preferences_zeros=False, calib_scale=True, detailed_mode=None,
                 remove_market_failures=None, quintiles=None, financing_cost=True,
                 ):
        super().__init__(stock, surface, ratio_surface, efficiency, income, consumption_ini, path=path, year=year,
                         debug_mode=debug_mode)

        self.certificate_jump_heater = None
        self.global_renovation = None
        self.financing_cost = financing_cost
        self.subsidies_count_insulation, self.subsidies_average_insulation = dict(), dict()
        self.subsidies_count_heater, self.subsidies_average_heater = dict(), dict()

        self.prepared_cost_insulation = None
        self.certificate_jump_all = None
        self.retrofit_with_heater = None
        self._calib_scale = calib_scale
        self.vta = 0.1
        self.factor_etp = 7.44 / 10**6 # ETP/€
        self.lifetime_insulation = 30
        self._epc2int = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}

        self.quintiles = quintiles
        if detailed_mode is None:
            detailed_mode = True
        self.detailed_mode = detailed_mode

        if logger is None:
            logger = logging.getLogger()
        self.logger = logger
        self.policies = []

        # {'max', 'market_share'} define how to calculate utility_extensive
        self._utility_extensive = utility_extensive

        if preferences_zeros:
            preferences['heater'] = {k: 0 for k in preferences['heater'].keys()}
            preferences['insulation'] = {k: 0 for k in preferences['insulation'].keys()}

        if isinstance(preferences['heater']['investment'], Series):
            self.pref_investment_heater = preferences['heater']['investment'].copy()
        else:
            self.pref_investment_heater = preferences['heater']['investment']

        if isinstance(preferences['insulation']['investment'], Series):
            self.pref_investment_insulation_int = preferences['insulation']['investment'].copy()
            self.pref_investment_insulation_ext = preferences['insulation']['investment'].copy()
        else:
            self.pref_investment_insulation_int = preferences['insulation']['investment']
            self.pref_investment_insulation_ext = preferences['insulation']['investment']

        self.pref_subsidy_heater = preferences['heater']['subsidy']
        if isinstance(preferences['insulation']['subsidy'], Series):
            self.pref_subsidy_insulation_int = preferences['insulation']['subsidy'].copy()
            self.pref_subsidy_insulation_ext = preferences['insulation']['subsidy'].copy()
        else:
            self.pref_subsidy_insulation_int = preferences['insulation']['subsidy']
            self.pref_subsidy_insulation_ext = preferences['insulation']['subsidy']

        if isinstance(preferences['insulation']['bill_saved'], Series):
            self.pref_bill_heater = preferences['heater']['bill_saved'].copy()
        else:
            self.pref_bill_heater = preferences['heater']['bill_saved']

        if isinstance(preferences['insulation']['bill_saved'], Series):
            self.pref_bill_insulation_int = preferences['insulation']['bill_saved'].copy()
            self.pref_bill_insulation_ext = preferences['insulation']['bill_saved'].copy()
        else:
            self.pref_bill_insulation_int = preferences['insulation']['bill_saved']
            self.pref_bill_insulation_ext = preferences['insulation']['bill_saved']

        self.pref_inertia = preferences['heater']['inertia']
        self.pref_zil_int = preferences['insulation']['zero_interest_loan']
        self.pref_zil_ext = preferences['insulation']['zero_interest_loan']

        # self.discount_rate = - self.pref_investment_insulation_ext / self.pref_bill_insulation_ext
        # self.discount_factor = (1 - (1 + self.discount_rate) ** -self.lifetime_insulation) / self.discount_rate

        self.scale_int = None
        self.scale_ext = None
        self.calibration_scale = 'cite'
        self.param_supply = None
        self.capacity_utilization = None
        self.factor_yrs = {}

        self._demolition_rate = demolition_rate
        self._demolition_total = (stock * self._demolition_rate).sum()
        self._target_demolition = ['E', 'F', 'G']

        self._choice_heater = None
        self._probability_replacement = None

        self._endogenous = endogenous

        self._target_exogenous = ['F', 'G']
        self._market_share_exogenous = None
        self._number_exogenous = number_exogenous

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
        self.subsidies_details_insulation, self.subsidies_insulation = None, None

        self.zil_count, self.zil_loaned_avg, self.zil_loaned = None, None, None

        self._share_decision_maker = stock.groupby(
            ['Occupancy status', 'Housing type', 'Income owner', 'Income tenant']).sum().unstack(
            ['Occupancy status', 'Income owner', 'Income tenant'])
        self._share_decision_maker = (self._share_decision_maker.T / self._share_decision_maker.sum(axis=1)).T

        self._remove_market_failures = remove_market_failures

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
        assert (stock >= 0).all(), 'Stock Error: Building stock cannot be negative'
        stock[stock < 0] = 0
        stock = stock[stock > 0]
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

        # no_insulation used for specific reason
        # no_insulation = MultiIndex.from_tuples([(False, False, False, False)], names=choice_insulation.names)
        # choice_insulation = choice_insulation.append(no_insulation)
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

    def heater_replacement(self, prices, cost_heater, policies_heater, ms_heater=None, probability_replacement=1/20,
                           index=None):
        """Function returns new building stock after heater replacement.

        Parameters
        ----------
        prices: Series
        cost_heater: Series
        ms_heater: DataFrame, optional
        policies_heater: list
        probability_replacement: float or Series, default 1/17
        index: MultiIndex optional, default None

        Returns
        -------
        Series
        """
        if ms_heater is not None:
            self._choice_heater = list(ms_heater.columns)

        if isinstance(probability_replacement, float):
            probability_replacement = Series(len(self._choice_heater) * [probability_replacement],
                                                Index(self._choice_heater, name='Heating system final'))

        if index is None:
            index = self.stock.index
            index = index.droplevel('Income tenant')
            index = index[~index.duplicated()]

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
            market_share, probability_replacement = self.exogenous_market_share_heater(index, choice_heater_idx)

        replacement = ((market_share * probability_replacement).T * self.stock_mobile.groupby(
            market_share.index.names).sum()).T

        stock_replacement = replacement.stack('Heating system final')
        to_replace = replacement.sum(axis=1)
        stock = self.stock_mobile.groupby(to_replace.index.names).sum() - to_replace

        # adding heating system final equal to heating system because no switch
        stock = concat((stock, Series(stock.index.get_level_values('Heating system'), index=stock.index,
                                            name='Heating system final')), axis=1).set_index('Heating system final', append=True).squeeze()
        stock = concat((stock.reorder_levels(stock_replacement.index.names), stock_replacement),
                       axis=0, keys=[False, True], names=['Heater replacement'])
        assert round(stock.sum() - self.stock_mobile.sum(), 0) == 0, 'Sum problem'

        replaced_by = stock.droplevel('Heating system').rename_axis(index={'Heating system final': 'Heating system'})

        if self.detailed_mode:
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
            if policy.name not in self.policies:
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
            self.subsidies_count_heater.update({key: (replacement.fillna(0) * mask).sum().sum()})
            self.subsidies_average_heater.update({key: sub.sum().sum() / replacement.fillna(0).sum().sum()})

    def calibration_constant_heater(self, utility, ms_heater):
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

    def endogenous_market_share_heater(self, index, prices, subsidies_total, cost_heater, ms_heater=None):

        choice_heater = self._choice_heater
        choice_heater_idx = Index(choice_heater, name='Heating system final')
        energy = Series(choice_heater).str.split('-').str[0].set_axis(choice_heater_idx)

        temp = pd.Series(0, index=index, dtype='float').to_frame().dot(pd.Series(0, index=choice_heater_idx, dtype='float').to_frame().T)
        index_final = temp.stack().index
        heat_consumption_sd, _, certificate = self.consumption_standard(index_final, level_heater='Heating system final')
        heat_consumption_sd = reindex_mi(heat_consumption_sd.unstack('Heating system final'), index)
        prices_re = prices.reindex(energy).set_axis(heat_consumption_sd.columns)
        energy_bill_sd = ((heat_consumption_sd * prices_re).T * reindex_mi(self._surface, index)).T

        consumption_before = self.consumption_standard(index, level_heater='Heating system')[0]
        consumption_before = reindex_mi(consumption_before, index) * reindex_mi(self._surface, index)
        energy_bill_before = AgentBuildings.energy_bill(prices, consumption_before)

        bill_saved = - energy_bill_sd.sub(energy_bill_before, axis=0)
        utility_bill_saving = (bill_saved.T * reindex_mi(self.pref_bill_heater, bill_saved.index)).T / 1000
        utility_bill_saving = utility_bill_saving.loc[:, choice_heater]

        certificate = reindex_mi(certificate.unstack('Heating system final'), index)
        certificate_before = self.consumption_standard(index)[2]
        certificate_before = reindex_mi(certificate_before, index)

        self.certificate_jump_heater = - certificate.replace(self._epc2int).sub(
            certificate_before.replace(self._epc2int), axis=0)

        utility_subsidies = subsidies_total * self.pref_subsidy_heater / 1000

        cost_heater = cost_heater.reindex(utility_bill_saving.columns)
        pref_investment = reindex_mi(self.pref_investment_heater, utility_bill_saving.index).rename(None)
        utility_investment = pref_investment.to_frame().dot(cost_heater.to_frame().T) / 1000

        utility_inertia = DataFrame(0, index=utility_bill_saving.index, columns=utility_bill_saving.columns)
        for hs in choice_heater:
            utility_inertia.loc[
                utility_inertia.index.get_level_values('Heating system') == hs, hs] = self.pref_inertia

        utility = utility_inertia + utility_investment + utility_bill_saving + utility_subsidies

        if (self.constant_heater is None) and (ms_heater is not None):
            ms_heater.dropna(how='all', inplace=True)
            self.constant_heater = self.calibration_constant_heater(utility, ms_heater)
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
        index: MultiIndex
        choice_heater_idx: Index

        Returns
        -------
        DataFrame
            Market-share by segment and possible heater choice.
        Series
            Probability replacement.
        """
        self._market_share_exogenous = {'Wood fuel-Standard boiler': 'Wood fuel-Performance boiler',
                                        'Wood fuel-Performance boiler': 'Wood fuel-Performance boiler',
                                        'Oil fuel-Standard boiler': 'Oil fuel-Performance boiler',
                                        'Oil fuel-Performance boiler': 'Oil fuel-Performance boiler',
                                        'Natural gas-Standard boiler': 'Natural gas-Performance boiler',
                                        'Natural gas-Performance boiler': 'Natural gas-Performance boiler',
                                        'Electricity-Performance boiler': 'Electricity-Heat pump',
                                        'Electricity-Heat pump': 'Electricity-Heat pump'}

        market_share = Series(index=index, dtype=float).to_frame().dot(
            Series(index=choice_heater_idx, dtype=float).to_frame().T)

        for initial, final in self._market_share_exogenous.items():
            market_share.loc[market_share.index.get_level_values('Heating system') == initial, final] = 1

        to_replace = self.stock_mobile[self.certificate.isin(self._target_exogenous)]

        if to_replace.sum() < self._number_exogenous:
            self._target_exogenous = ['E', 'F', 'G']
            to_replace = self.stock_mobile[self.certificate.isin(self._target_exogenous)]
            if to_replace.sum() < self._number_exogenous:
                self._target_exogenous = ['D', 'E', 'F', 'G']
                to_replace = self.stock_mobile[self.certificate.isin(self._target_exogenous)]
                if to_replace.sum() < self._number_exogenous:
                    self._target_exogenous = ['C', 'D', 'E', 'F', 'G']
                    to_replace = self.stock_mobile[self.certificate.isin(self._target_exogenous)]
                    if to_replace.sum() < self._number_exogenous:
                        self._number_exogenous = 0

        to_replace = to_replace / to_replace.sum() * self._number_exogenous

        to_replace = to_replace.groupby(index.names).sum()
        probability_replacement = (to_replace / self.stock_mobile.groupby(index.names).sum()).fillna(0)
        probability_replacement = probability_replacement.reindex(market_share.index)
        return market_share, probability_replacement

    def insulation_replacement(self, prices, cost_insulation_raw, ms_insulation=None, renovation_rate_ini=None,
                               policies_insulation=None, target_freeriders=None, index=None, stock=None,
                               supply_constraint=False, financing_cost=None):
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
        prices: Series
        cost_insulation_raw: Series
            €/m2 of losses area by component.
        ms_insulation: Series
        renovation_rate_ini: Series
        policies_insulation: list
        target_freeriders: float
        index: MultiIndex or Index, default None
            Add heater replacement information compare to self.stock.
        stock: Series, default None

        Returns
        -------
        Series
            Retrofit rate
        DataFrame
            Market-share insulation
        """
        if index is None:
            index = self.stock.index

        _, _, certificate_before_heater = self.consumption_standard(index, level_heater='Heating system')
        # index only contains building with energy performance > B
        c_before = reindex_mi(certificate_before_heater, index)
        index = c_before[c_before > 'B'].index

        # before include the change of heating system
        _, consumption_3uses_before, certificate_before = self.consumption_standard(index,
                                                                                    level_heater='Heating system final')
        certificate_before = certificate_before[certificate_before > 'B']
        consumption_3uses_before = consumption_3uses_before.loc[certificate_before.index]

        surface = reindex_mi(self._surface, index)

        _, consumption_3uses, certificate = self.prepare_consumption(self._choice_insulation, index=index,
                                                                     level_heater='Heating system final')
        energy_saved_3uses = ((consumption_3uses_before - consumption_3uses.T) / consumption_3uses_before).T
        energy_saved_3uses.dropna(inplace=True)

        cost_insulation = self.prepare_cost_insulation(cost_insulation_raw * self.surface_insulation)
        cost_insulation = cost_insulation.T.multiply(self._surface, level='Housing type').T

        cost_insulation, tax_insulation, tax, subsidies_details, subsidies_total, condition, certificate_jump_all = self.apply_subsidies_insulation(
            index, policies_insulation, cost_insulation, surface, certificate, certificate_before, certificate_before_heater, energy_saved_3uses)

        if self._endogenous:

            utility_subsidies = subsidies_total.copy()
            zil = [p for p in policies_insulation if (p.name == 'zero_interest_loan')]
            l = ['reduced_tax'] + [z.name for z in zil if z.policy != 'subsidy_ad_volarem']
            for sub in l:
                if sub in subsidies_details.keys():
                    utility_subsidies -= subsidies_details[sub]

            utility_zil = None
            if 'zero_interest_loan' in subsidies_details:
                if zil[0].policy != 'subsidy_ad_volarem':
                    utility_zil = subsidies_details['zero_interest_loan'].copy()

            delta_subsidies = None
            if (self.year in [self.first_year + 1]) and (self.scale_ext is None):
                delta_subsidies = subsidies_details['cite'].copy()

            retrofit_rate, market_share = self.endogenous_retrofit(index, prices, utility_subsidies,
                                                                   cost_insulation,
                                                                   ms_insulation=ms_insulation,
                                                                   renovation_rate_ini=renovation_rate_ini,
                                                                   utility_zil=utility_zil,
                                                                   stock=stock,
                                                                   delta_subsidies=delta_subsidies,
                                                                   target_freeriders=target_freeriders,
                                                                   supply_constraint=supply_constraint,
                                                                   financing_cost=financing_cost
                                                                   )

        else:
            retrofit_rate, market_share = self.exogenous_retrofit(index, self._choice_insulation)

        if self.detailed_mode:
            self.store_information_insulation(certificate_jump_all, condition, cost_insulation_raw, tax, cost_insulation,
                                              tax_insulation, subsidies_details, subsidies_total, retrofit_rate,
                                              )
        else:
            self.subsidies_details_insulation = subsidies_details

        return retrofit_rate, market_share

    def apply_subsidies_insulation(self, index, policies_insulation, cost_insulation, surface, certificate, certificate_before,
                                   certificate_before_heater, energy_saved_3uses):
        """Calculate subsidies amount for each possible insulation choice.

        Parameters
        ----------
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

        def define_zil_target(certificate, certificate_before, energy_saved_3uses):
            """Define target.

            zero_interest_loan_old is the target in terms of EPC jump.
            zero_interest_loan_new is the requirement to be eligible to a 'global renovation' program,
            the renovation must reduce of 35% the conventional primary energy need
            and the resulting building must not be of G or F epc level.

            Parameters
            ----------
            certificate
            certificate_before
            energy_saved_3uses

            Returns
            -------
            target_subsidies: pd.DataFrame
                Each cell, a gesture and a segment, is a boolean which is True if it is targeted by the policy

            """
            energy_saved_min = 0.35

            target_subsidies = {}
            target_0 = certificate.isin(['D', 'C', 'B', 'A']).astype(int).mul(
                certificate_before.isin(['G', 'F', 'E']).astype(int), axis=0).astype(bool)
            target_1 = certificate.isin(['B', 'A']).astype(int).mul(certificate_before.isin(['D', 'C']).astype(int),
                                                                    axis=0).astype(bool)
            target_subsidies['zero_interest_loan_old'] = target_0 | target_1

            target_0 = certificate.isin(['E', 'D', 'C', 'B', 'A']).astype(bool)
            target_1 = energy_saved_3uses[energy_saved_3uses >= energy_saved_min].fillna(0).astype(bool)
            target_subsidies['zero_interest_loan_new'] = target_0 & target_1

            return target_subsidies

        def defined_condition(index, certificate, certificate_before, certificate_before_heater, energy_saved_3uses):
            """Define condition to get subsidies or loan.

            Depends on income (index) and energy performance of renovationd defined by certificate jump or
            energy_saved_3uses.

            Parameters
            ----------
            index: MultiIndex
            certificate: DataFrame
            certificate_before: Series
            certificate_before_heater: Series
            energy_saved_3uses: DataFrame

            Returns
            -------
            condition: dict
                Contains boolean DataFrame that established condition to get subsidies.
            certificate_jump: DataFrame
                Insulation (without account for heater replacement) allowed to jump of at least one certificate.
            certificate_jump_all: DataFrame
                Renovation (including heater replacement) allowed to jump of at least one certificate.
            """

            condition = dict()

            self.out_worst = (~certificate.isin(['G', 'F'])).T.multiply(certificate_before.isin(['G', 'F'])).T
            self.out_worst = reindex_mi(self.out_worst, index).fillna(False).astype('float')
            self.in_best = (certificate.isin(['A', 'B'])).T.multiply(~certificate_before.isin(['A', 'B'])).T
            self.in_best = reindex_mi(self.in_best, index).fillna(False).astype('float')

            condition.update({'bonus_worst': self.out_worst})
            condition.update({'bonus_best': self.in_best})

            minimum_gest_condition, global_condition = 1, 2
            energy_condition = 0.35

            certificate_jump = - certificate.replace(self._epc2int).sub(certificate_before.replace(self._epc2int),
                                                                        axis=0)
            certificate_jump = reindex_mi(certificate_jump, index)
            certificate_jump_condition = certificate_jump >= minimum_gest_condition

            certificate_before_heater = reindex_mi(certificate_before_heater, index)
            certificate = reindex_mi(certificate, index)
            certificate_jump_all = - certificate.replace(self._epc2int).sub(
                certificate_before_heater.replace(self._epc2int),
                axis=0)

            condition.update({'certificate_jump': certificate_jump_all >= minimum_gest_condition})
            condition.update({'global_renovation': certificate_jump_all >= global_condition})

            low_income_condition = ['D1', 'D2', 'D3', 'D4']
            if self.quintiles:
                low_income_condition = ['C1', 'C2']
            low_income_condition = index.get_level_values('Income owner').isin(low_income_condition)
            low_income_condition = pd.Series(low_income_condition, index=index)

            high_income_condition = ['D5', 'D6', 'D7', 'D8', 'D9', 'D10']
            if self.quintiles:
                high_income_condition = ['C3', 'C4', 'C5']
            high_income_condition = index.get_level_values('Income owner').isin(high_income_condition)
            high_income_condition = pd.Series(high_income_condition, index=index)

            global_renovation_low_income = (low_income_condition * condition['global_renovation'].T).T
            condition.update({'global_renovation_low_income': global_renovation_low_income})

            global_renovation_high_income = (high_income_condition * condition['global_renovation'].T).T
            condition.update({'global_renovation_high_income': global_renovation_high_income})

            energy_condition = energy_saved_3uses >= energy_condition

            condition_mpr_serenite = (reindex_mi(energy_condition, index).T * low_income_condition).T
            condition.update({'mpr_serenite': condition_mpr_serenite})

            condition_zil = define_zil_target(certificate, certificate_before, energy_saved_3uses)
            condition.update({'zero_interest_loan': condition_zil})

            return condition, certificate_jump, certificate_jump_all

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
                                                                              energy_saved_3uses)

        for policy in policies_insulation:
            if policy.name not in self.policies:
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

        return cost_insulation, tax_insulation, tax, subsidies_details, subsidies_total, condition, certificate_jump_all

    def store_information_insulation(self, certificate_jump_all, condition, cost_insulation_raw, tax, cost_insulation,
                                     tax_insulation, subsidies_details, subsidies_total, retrofit_rate):

        """Store insulation information.

        Parameters
        ----------
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
        retrofit_rate: Series
        """

        # self.certificate_jump = condition['certificate_jump']
        self.certificate_jump_all = certificate_jump_all
        self.global_renovation = condition['global_renovation']
        self.global_renovation_high_income = condition['global_renovation_high_income']
        self.global_renovation_low_income = condition['global_renovation_low_income']
        self.cost_component = cost_insulation_raw * self.surface_insulation * (1 + tax)
        self.subsidies_details_insulation = subsidies_details
        self.subsidies_insulation_indiv = subsidies_total
        self.cost_insulation_indiv = cost_insulation
        self.tax_insulation = tax_insulation
        self.retrofit_rate = retrofit_rate

    def store_information_retrofit(self, replaced_by):
        """Calculate and store main outputs based on yearly retrofit.

        Parameters
        ----------
        replaced_by: DataFrame
            Retrofitting for each dwelling and each insulation gesture.
        """

        levels = [i for i in replaced_by.index.names if i not in ['Heater replacement', 'Heating system final']]
        self.global_renovation_high_income = (replaced_by * self.global_renovation_high_income).sum().sum()
        self.global_renovation_low_income = (replaced_by * self.global_renovation_low_income).sum().sum()
        self.bonus_best = (replaced_by * self.in_best).sum().sum()
        self.bonus_worst = (replaced_by * self.out_worst).sum().sum()
        self.replacement_insulation = replaced_by.groupby(levels).sum()
        self.investment_insulation = (replaced_by * self.cost_insulation_indiv).groupby(levels).sum()
        self.taxed_insulation = (replaced_by * self.tax_insulation).groupby(levels).sum()
        self.subsidies_insulation = (replaced_by * self.subsidies_insulation_indiv).groupby(levels).sum()

        for key in self.subsidies_details_insulation.keys():
            self.subsidies_details_insulation[key] = (replaced_by * reindex_mi(
                self.subsidies_details_insulation[key], replaced_by.index)).groupby(levels).sum()

        rslt = {}
        l = unique(self.certificate_jump_all.values.ravel('K'))
        for i in l:
            rslt.update({i: ((self.certificate_jump_all == i) * replaced_by).sum(axis=1)})
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

        for key, sub in self.subsidies_details_insulation.items():
            mask = sub.copy()
            mask[mask > 0] = 1
            self.subsidies_count_insulation.update({key: (replaced_by.fillna(0) * mask).sum().sum()})
            self.subsidies_average_insulation.update({key: sub.sum().sum() / replaced_by.fillna(0).sum().sum()})

            if key == 'zero_interest_loan':
                total_loaned = (replaced_by.fillna(0) * self.zil_loaned).sum().sum()
                self.zil_loaned_avg = total_loaned / self.zil_count

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

    def endogenous_retrofit(self, index, prices, subsidies_total, cost_insulation, ms_insulation=None,
                            renovation_rate_ini=None, utility_zil=None, stock=None, supply_constraint=False,
                            delta_subsidies=None, target_freeriders=0.85, financing_cost=None):
        """Calculate endogenous retrofit based on discrete choice model.

        Utility variables are investment cost, energy bill saving, and subsidies.
        Preferences are object attributes defined initially.

        # bill saved calculated based on the new heating system
        # certificate before work and so subsidies before the new heating system

        Parameters
        ----------
        index: MultiIndex
        prices: Series
        subsidies_total: DataFrame
        cost_insulation: DataFrame
        ms_insulation: Series, default None
        renovation_rate_ini: Series, default None
        utility_zil: DataFrame, default None
        stock: Series, default None
        supply_constraint: bool
        delta_subsidies: DataFrame, default None
        target_freeriders: float, default 0.85

        Returns
        -------
        Series
            Retrofit rate
        DataFrame
            Market-share insulation
        """

        def to_market_share(bill_saved, subsidies, investment, utility_zil=utility_zil, scale=1.0):

            pref_subsidies = reindex_mi(self.pref_subsidy_insulation_int, subsidies.index).rename(None)
            utility_subsidies = (subsidies.T * pref_subsidies).T / 1000

            pref_investment = reindex_mi(self.pref_investment_insulation_int, investment.index).rename(None)
            utility_investment = (investment.T * pref_investment).T / 1000

            utility_bill_saving = (bill_saved.T * reindex_mi(self.pref_bill_insulation_int, bill_saved.index)).T / 1000

            utility_intensive = utility_bill_saving + utility_investment + utility_subsidies

            if utility_zil is not None:
                utility_zil[utility_zil > 0] = self.pref_zil_int
                utility_intensive += utility_zil

            if self.constant_insulation_intensive is not None:
                # constant = reindex_mi(self.constant_insulation_intensive, utility_intensive.index)
                utility_intensive += self.constant_insulation_intensive

            # removing floor and roof insulation for multi-family
            if False:
                cond1 = utility_intensive.index.get_level_values('Housing type') == 'Multi-family'
                cond2 = (utility_intensive.columns.get_level_values('Floor') == True) | (utility_intensive.columns.get_level_values('Roof') == True)
                utility_intensive.loc[cond1, cond2] = float('nan')

            market_share = (exp(scale * utility_intensive).T / exp(scale * utility_intensive).sum(axis=1)).T
            return market_share, utility_intensive

        def retrofit_func(utility, rate_max=1.0):
            return 1 / (1 + exp(- utility)) * rate_max

        def to_retrofit_rate(bill_saved, subsidies, investment, bool_zil=None, debug_mode=self._debug_mode):
            utility_bill_saving = reindex_mi(self.pref_bill_insulation_ext, bill_saved.index) * bill_saved / 1000

            pref_subsidies = reindex_mi(self.pref_subsidy_insulation_ext, subsidies.index).rename(None)
            utility_subsidies = (pref_subsidies * subsidies) / 1000

            pref_investment = reindex_mi(self.pref_investment_insulation_ext, investment.index).rename(None)
            utility_investment = (pref_investment * investment) / 1000

            utility = utility_investment + utility_bill_saving + utility_subsidies

            if bool_zil is not None:
                utility_zil = bool_zil * self.pref_zil_ext
                utility += utility_zil

            if self.constant_insulation_extensive is not None:
                _utility = self.add_certificate(utility.copy())
                utility_constant = reindex_mi(self.constant_insulation_extensive, _utility.index)
                _utility += utility_constant
                utility = _utility.droplevel('Performance')

            retrofit_rate = retrofit_func(utility)

            if debug_mode and self.constant_insulation_extensive is not None:
                levels = [l for l in self.constant_insulation_extensive.index.names if l != 'Performance']
                df = concat((utility_investment.groupby(levels).mean(),
                                utility_subsidies.groupby(levels).mean(),
                                utility_bill_saving.groupby(levels).mean(),
                                self.constant_insulation_extensive.groupby(levels).mean()), axis=1,
                               keys=['Investment', 'Subsidies', 'Saving', 'Constant'])

                if bool_zil is not None:
                    zil_mean = utility_zil.groupby(levels).mean().rename('ZIL')
                    df = concat((df, zil_mean), axis=1)

            return retrofit_rate, utility

        def impact_subsidies(scale, utility, stock, delta_subsidies, pref_subsidies, indicator='freeriders'):
            """Calculate freeriders due to implementation of subsidy.

            Parameters
            ----------
            scale: int
            utility: Series
            stock: Series
            delta_subsidies: Series
            pref_subsidies: Series
            indicator: {'freeriders', 'elasticity'}

            Returns
            -------
            float
            """

            retrofit = retrofit_func(utility * scale)
            flow = (retrofit * stock).sum()
            retrofit = flow / stock.sum()

            utility_plus = (utility + pref_subsidies * delta_subsidies).dropna() * scale
            retrofit_plus = retrofit_func(utility_plus)
            flow_plus = (retrofit_plus * stock).sum()
            retrofit_plus = flow_plus / stock.sum()

            if indicator == 'elasticity':
                return (retrofit_plus - retrofit) / retrofit

            if indicator == 'freeriders':
                return min(flow, flow_plus) / max(flow, flow_plus)

        def calibration_intensive_fsolve(utility, stock, ms_insulation, retrofit_rate_ini):
            def solve(constant, utility_ref, ms_insulation, flow_retrofit):
                # constant = append(0, constant)
                constant = Series(constant, index=utility_ref.columns)
                constant.iloc[0] = 0
                utility = utility_ref + constant
                market_share = (exp(utility).T / exp(utility).sum(axis=1)).T
                agg = (market_share.T * flow_retrofit).T
                market_share_agg = (agg.sum() / agg.sum().sum()).reindex(ms_insulation.index)

                if (market_share_agg.round(decimals=2) == ms_insulation.round(decimals=2)).all():
                    return zeros(ms_insulation.shape[0])
                else:
                    return market_share_agg - ms_insulation

            if 'Performance' in retrofit_rate_ini.index.names:
                levels = [l for l in retrofit_rate_ini.index.names if l != 'Performance']
                certificate = self.certificate.rename('Performance').groupby(
                    [l for l in self.stock.index.names if l != 'Income tenant']).first()
                certificate = reindex_mi(certificate, stock.index)
                stock = concat((stock, certificate), axis=1).set_index('Performance', append=True).squeeze()
                retrofit_rate_simple = (stock * reindex_mi(retrofit_rate_ini, stock.index)).groupby(
                    levels).sum() / stock.groupby(levels).sum()
            else:
                retrofit_rate_simple = retrofit_rate_ini

            probability_replacement = self._probability_replacement
            if isinstance(probability_replacement, Series):
                probability_replacement.index = probability_replacement.index.rename('Heating system')
                probability_replacement = reindex_mi(probability_replacement, self._stock.index)

            stock = concat((self.stock * probability_replacement,
                               self.stock * (1 - probability_replacement)), axis=0, keys=[True, False],
                              names=['Heater replacement'])
            stock_single = stock.xs('Single-family', level='Housing type', drop_level=False)

            flow_retrofit = stock_single * reindex_mi(retrofit_rate_simple, stock_single.index)

            utility = utility.groupby([i for i in utility.index.names if i != 'Heating system final']).mean()
            utility_ref = reindex_mi(utility, flow_retrofit.index).dropna()
            flow_retrofit = flow_retrofit.reindex(utility_ref.index)

            x0 = zeros(ms_insulation.shape[0])
            constant = fsolve(solve, x0, args=(utility_ref, ms_insulation, flow_retrofit))
            constant = Series(constant, index=utility_ref.columns)

            utility = utility_ref + constant
            market_share = (exp(utility).T / exp(utility).sum(axis=1)).T
            agg = (market_share.T * flow_retrofit).T
            market_share_agg = (agg.sum() / agg.sum().sum()).reindex(ms_insulation.index)
            details = concat((constant, market_share_agg, ms_insulation), axis=1,
                                keys=['constant', 'calcul', 'observed']).round(decimals=3)
            if self.path is not None:
                details.to_csv(os.path.join(self.path_calibration, 'calibration_constant_insulation.csv'))
            return constant

        def calibration_intensive_iteration(utility, stock, ms_insulation, retrofit_rate_ini, iteration=100,
                                            all=True):
            """Calibrate alternative-specific constant to match observed market-share.

            Parameters
            ----------
            utility: Series
            ms_insulation: Series
                Observed market-share.
            retrofit_rate_ini: Series
                Observed renovation rate.

            Returns
            -------
            Series
            """

            if 'Performance' in retrofit_rate_ini.index.names:
                stock = self.add_certificate(stock)

            if all:
                flow_retrofit = stock * reindex_mi(retrofit_rate_ini, stock.index)
                utility_ref = reindex_mi(utility, flow_retrofit.index).dropna()
                constant = ms_insulation.reindex(utility_ref.columns, axis=0).copy()
                constant[constant > 0] = 0
                market_share_ini, market_share_agg = None, None
                for i in range(iteration):
                    _utility = (utility_ref + constant).copy()
                    constant.iloc[0] = 0
                    market_share = (exp(_utility).T / exp(_utility).sum(axis=1)).T
                    agg = (market_share.T * flow_retrofit).T
                    market_share_agg = (agg.sum() / agg.sum().sum()).reindex(ms_insulation.index)
                    if i == 0:
                        market_share_ini = market_share_agg.copy()
                    constant = constant + log(ms_insulation / market_share_agg)

                    if (market_share_agg.round(decimals=2) == ms_insulation.round(decimals=2)).all():
                        self.logger.debug('Constant intensive optim worked')
                        break

                constant.iloc[0] = 0
                nb_renovation = (stock * reindex_mi(retrofit_rate_ini, stock.index)).sum()
                details = concat((constant, market_share_ini, market_share_agg, ms_insulation, (ms_insulation * nb_renovation) / 10**3), axis=1,
                                 keys=['constant', 'calcul ini', 'calcul', 'observed', 'thousand']).round(decimals=3)
                if self.path is not None:
                    details.to_csv(os.path.join(self.path_calibration, 'calibration_constant_insulation.csv'))

                return constant

            else:
                # single-family
                stock_single = stock.xs('Single-family', level='Housing type', drop_level=False)
                flow_retrofit = stock_single * reindex_mi(retrofit_rate_ini, stock_single.index)

                utility_ref = reindex_mi(utility, flow_retrofit.index).dropna()

                constant = ms_insulation.reindex(utility_ref.columns, axis=0).copy()
                constant[constant > 0] = 0
                market_share_ini, market_share_agg = None, None
                for i in range(iteration):
                    _utility = (utility_ref + constant).copy()
                    constant.iloc[0] = 0
                    market_share = (exp(_utility).T / exp(_utility).sum(axis=1)).T
                    agg = (market_share.T * flow_retrofit).T
                    market_share_agg = (agg.sum() / agg.sum().sum()).reindex(ms_insulation.index)
                    if i == 0:
                        market_share_ini = market_share_agg.copy()
                    constant = constant + log(ms_insulation / market_share_agg)

                    if (market_share_agg.round(decimals=2) == ms_insulation.round(decimals=2)).all():
                        self.logger.debug('Constant intensive optim worked')
                        break

                constant.iloc[0] = 0
                details = concat((constant, market_share_ini, market_share_agg, ms_insulation), axis=1,
                                    keys=['constant', 'calcul ini', 'calcul', 'observed']).round(decimals=3)
                if self.path is not None:
                    details.to_csv(os.path.join(self.path_calibration, 'calibration_constant_insulation_single.csv'))
                constant_insulation = constant.rename('Single-family')

                # multi-family
                ms_insulation = ms_insulation.xs(False, level='Floor', drop_level=False).xs(False, level='Roof', drop_level=False)
                ms_insulation = ms_insulation / ms_insulation.sum()

                stock_multi = stock.xs('Multi-family', level='Housing type', drop_level=False)
                flow_retrofit = stock_multi * reindex_mi(retrofit_rate_ini, stock_multi.index)

                utility_ref = reindex_mi(utility, flow_retrofit.index).dropna(how='all', axis=0).dropna(how='all', axis=1)

                constant = ms_insulation.reindex(utility_ref.columns, axis=0).copy()
                constant[constant > 0] = 0
                market_share_ini, market_share_agg = None, None
                for i in range(iteration):
                    _utility = (utility_ref + constant).copy()
                    constant.iloc[0] = 0
                    market_share = (exp(_utility).T / exp(_utility).sum(axis=1)).T
                    agg = (market_share.T * flow_retrofit).T
                    market_share_agg = (agg.sum() / agg.sum().sum()).reindex(ms_insulation.index)
                    if i == 0:
                        market_share_ini = market_share_agg.copy()
                    constant = constant + log(ms_insulation / market_share_agg)

                    if (market_share_agg.round(decimals=2) == ms_insulation.round(decimals=2)).all():
                        self.logger.debug('Constant intensive optim worked')
                        break

                constant.iloc[0] = 0
                details = concat((constant, market_share_ini, market_share_agg, ms_insulation), axis=1,
                                    keys=['constant', 'calcul ini', 'calcul', 'observed']).round(decimals=3)
                if self.path is not None:
                    details.to_csv(os.path.join(self.path_calibration, 'calibration_constant_insulation_multi.csv'))
                constant = constant.rename('Multi-family')

                constant = concat((constant_insulation, constant), axis=1).T
                constant.index.names = ['Housing type']
                return constant

        def calibration_intensive(utility, stock, ms_insulation, retrofit_rate_ini, solver='iteration'):
            if solver == 'iteration':
                return calibration_intensive_iteration(utility, stock, ms_insulation, retrofit_rate_ini)
            elif solver == 'fsolve':
                return calibration_intensive_fsolve(utility, stock, ms_insulation, retrofit_rate_ini)

        def calibration_constant_scale_ext(utility, stock, retrofit_rate_ini, target_freeriders, delta_subsidies,
                                           pref_subsidies):
            """Simultaneously calibrate constant and scale to match freeriders and retrofit rate.

            Parameters
            ----------
            utility
            stock
            retrofit_rate_ini
            target_freeriders
            delta_subsidies
            pref_subsidies

            Returns
            -------

            """

            def solve(x, utility_ini, stock_ini, retrofit_rate_target, freeride, delta_sub, pref_sub):
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
                calcul = impact_subsidies(scale, u, stock_ini, delta_sub, pref_sub, indicator='freeriders')
                rslt = append(rslt, calcul - freeride)

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

            if 'Performance' in retrofit_rate_ini.index.names:
                stock = self.add_certificate(stock)
                stock_retrofit = stock[stock.index.get_level_values('Performance') > 'B']
                utility = self.add_certificate(utility)
            else:
                stock_retrofit = stock

            constant = retrofit_rate_ini.copy()
            constant[retrofit_rate_ini > 0] = 0
            a = stock.groupby(retrofit_rate_ini.index.names).mean()
            b = utility.groupby(retrofit_rate_ini.index.names).mean()
            pd.concat((retrofit_rate_ini, a, b), axis=1)
            if self._calib_scale:
                x = append(constant.to_numpy(), 1)
                root, infodict, ier, mess = fsolve(solve, x, args=(
                utility, stock_retrofit, retrofit_rate_ini, target_freeriders, - delta_subsidies / 1000, pref_subsidies),
                              full_output=True)
                self.logger.info(mess)
                scale = root[-1]
                self.logger.info('Scale: {}'.format(scale))
                constant = Series(root[:-1], index=retrofit_rate_ini.index) * scale
            else:
                x = constant.to_numpy()
                root, infodict, _, _ = fsolve(solve_noscale, x, args=(utility, stock_retrofit, retrofit_rate_ini),
                                              full_output=True)
                scale = 1.0
                constant = Series(root, index=retrofit_rate_ini.index) * scale

            utility_constant = reindex_mi(constant, utility.index)
            utility = utility * scale + utility_constant
            retrofit_rate = retrofit_func(utility)
            agg = (retrofit_rate * stock).groupby(retrofit_rate_ini.index.names).sum()
            retrofit_rate_agg = agg / stock.groupby(retrofit_rate_ini.index.names).sum()

            details = concat((constant, retrofit_rate_agg, retrofit_rate_ini, agg / 10 ** 3), axis=1,
                                keys=['constant', 'calcul', 'observed', 'thousand']).round(decimals=3)
            if self.path is not None:
                details.to_csv(os.path.join(self.path_calibration, 'calibration_constant_extensive.csv'))

            return constant, scale

        def calculation_intensive_margin(stock, retrofit_rate_ini, bill_saved, subsidies_total, cost_insulation,
                                         delta_subsidies, target_invest=0.2, utility_zil=utility_zil):
            """ This function can be adapted to calibrate intensive margin on Risch 2020 result. (using target_invest)
            However, for now just returns percentage of intesive margin difference

            Parameters
            ----------
            stock
            retrofit_rate_ini
            bill_saved: DataFrame
            subsidies_total: DataFrame
            cost_insulation: DataFrame
            delta_subsidies: DataFrame, policies used to calibrate the scale.
            target_invest: float
            utility_zil

            Returns
            -------

            """
            if 'Performance' in retrofit_rate_ini.index.names:
                stock = self.add_certificate(stock)
            flow_retrofit = stock * reindex_mi(retrofit_rate_ini, stock.index)
            flow_retrofit = flow_retrofit.droplevel('Performance').dropna()

            def solve(scale, flow_retrofit, bill_saved, subsidies_total, cost_insulation, delta_subsidies,
                      target_invest, utility_zil):
                scale = float(scale)
                ms_before, _ = to_market_share(bill_saved, subsidies_total, cost_insulation,
                                               utility_zil=utility_zil, scale=scale)
                investment_insulation_before = (cost_insulation.reindex(ms_before.index) * ms_before).sum(axis=1)
                investment_insulation_before = (investment_insulation_before * flow_retrofit).sum() / flow_retrofit.sum()
                new_sub = subsidies_total + delta_subsidies
                ms_after, _ = to_market_share(bill_saved, new_sub, cost_insulation, utility_zil=utility_zil,
                                              scale=scale)
                investment_insulation_after = (cost_insulation.reindex(ms_after.index) * ms_after).sum(axis=1)
                investment_insulation_after = (investment_insulation_after * flow_retrofit).sum() / flow_retrofit.sum()

                delta_invest = (
                                       investment_insulation_before - investment_insulation_after) / investment_insulation_before
                return delta_invest - target_invest

            x0 = ones(1)

            """scale = fsolve(solve, x0, args=(
                flow_retrofit, bill_saved, subsidies_total, cost_insulation, -delta_subsidies,
                target_invest, utility_zil))"""

            return solve(1, flow_retrofit, bill_saved, subsidies_total, cost_insulation, -0.5 * delta_subsidies,
                         target_invest, utility_zil) + target_invest

        def supply_interaction(retrofit_rate, stock, investment_insulation):
            """NOT IMPLEMENTED YET.

            Parameters
            ----------
            retrofit_rate
            stock
            investment_insulation

            Returns
            -------

            """
            market_size = (retrofit_rate * stock * investment_insulation).sum()
            etp_size = self.factor_etp * market_size

            if self.param_supply is None:

                self.capacity_utilization = etp_size / 0.8
                self.param_supply = dict()
                self.param_supply['a'], self.param_supply['b'], self.param_supply['c'] = self.calibration_supply()

                x = arange(0, 1.05, 0.05)
                y = self.factor_function(self.param_supply['a'], self.param_supply['b'], self.param_supply['c'], x)
                fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))
                ax.plot(x, y, color='black')
                format_ax(ax, y_label='Cost factor')
                ax.set_xlabel('Utilization rate (%)')
                save_fig(fig, save=os.path.join(self.path_calibration, 'marginal_cost_curve.png'))

                x, y_supply, y_demand = [], [], []
                for factor in arange(0.81, 1.5, 0.05):
                    retrofit_rate = to_retrofit_rate(bill_saved_insulation, subsidies_insulation,
                                                     investment_insulation * factor)[0]
                    y_demand.append((retrofit_rate * stock * investment_insulation * self.factor_etp).sum())
                    utilization_rate = self.supply_function(self.param_supply['a'], self.param_supply['b'],
                                                            self.param_supply['c'], factor)
                    y_supply.append(utilization_rate * self.capacity_utilization)
                    x.append(factor)

                df = concat((Series(x), Series(y_supply)/10**3, Series(y_demand)/10**3,),
                               axis=1).set_axis(['Cost factor', 'Supply', 'Demand'], axis=1)

                fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))
                df.plot(ax=ax, x='Cost factor', color={'Supply': 'darkorange', 'Demand': 'royalblue'})
                format_ax(ax, y_label='Quantity (thousands of jobs)', format_y=lambda y, _: '{:.0f}'.format(y))
                format_legend(ax)
                save_fig(fig, save=os.path.join(self.path_calibration, 'supply_demand_inverse.png'))

                fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))
                df.plot(ax=ax, x='Supply', y='Cost factor', color='darkorange')
                df.plot(ax=ax, x='Demand', y='Cost factor', color='royalblue')
                format_ax(ax, y_label='Cost factor', format_y=lambda y, _: '{:.1f}'.format(y))
                ax.set_xlabel('Quantity (thousands of jobs)')
                format_legend(ax, labels=['Supply', 'Demand'])
                save_fig(fig, save=os.path.join(self.path_calibration, 'supply_demand.png'))

            def solve_equilibrium(factor, bill_saved_insulation, subsidies_insulation, investment_insulation, stock):
                retrofit_rate = to_retrofit_rate(bill_saved_insulation, subsidies_insulation,
                                                 investment_insulation * factor)[0]
                demand = (retrofit_rate * stock * investment_insulation * self.factor_etp).sum()
                offer = self.supply_function(self.param_supply['a'], self.param_supply['b'],
                                             self.param_supply['c'], factor) * self.capacity_utilization
                return demand - offer

            factor_equilibrium = fsolve(solve_equilibrium, array([1]), args=(
            bill_saved_insulation, subsidies_insulation, investment_insulation, stock))

            retrofit_rate = to_retrofit_rate(bill_saved_insulation, subsidies_insulation,
                                             investment_insulation * factor_equilibrium)[0]

            self.factor = factor_equilibrium
            if self._debug_mode:
                self.factor_yrs.update({self.year: factor_equilibrium})
            return retrofit_rate

        cost_insulation = reindex_mi(cost_insulation, index)
        cost_total = cost_insulation.copy()
        if financing_cost is not None and self.financing_cost:
            share_debt = financing_cost['share_debt'][0] + cost_insulation * financing_cost['share_debt'][1]
            cost_debt = financing_cost['interest_rate'] * share_debt * cost_insulation * financing_cost['duration']
            cost_saving = (financing_cost['saving_rate'] * reindex_mi(financing_cost['factor_saving_rate'], cost_insulation.index) * ((1 - share_debt) * cost_insulation).T).T * financing_cost['duration']
            cost_financing = cost_debt + cost_saving

            cost_total += cost_financing

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

        # idx = (False, True, 'Owner-occupied', 'D1', 'Multi-family', 'Electricity-Performance boiler', 0.5,  0.2, 0.1, 1.6, 'Electricity-Performance boiler')

        market_share, utility_intensive = to_market_share(bill_saved, subsidies_total, cost_total,
                                                          utility_zil=utility_zil)

        if self.constant_insulation_intensive is None:
            self.logger.info('Calibration intensive')
            self.constant_insulation_intensive = calibration_intensive(utility_intensive, stock, ms_insulation,
                                                                       renovation_rate_ini, solver='iteration')
            market_share, utility_intensive = to_market_share(bill_saved, subsidies_total, cost_total,
                                                              utility_zil=utility_zil)

            percentage_intensive_margin = calculation_intensive_margin(stock, renovation_rate_ini, bill_saved,
                                                                       subsidies_total,
                                                                       cost_insulation, delta_subsidies,
                                                                       utility_zil=utility_zil)

            if self._debug_mode:
                fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))
                utility_intensive.boxplot(ax=ax, fontsize=12, figsize=(8, 10))
                plt.xticks(fontsize=7, rotation=45)
                fig.savefig(os.path.join(self.path_calibration_renovation, 'utility_insulation_distribution.png'))
                plt.close(fig)

                fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))
                market_share.boxplot(ax=ax, fontsize=12, figsize=(8, 10))
                plt.xticks(fontsize=7, rotation=45)
                fig.savefig(os.path.join(self.path_calibration_renovation, 'market_share_distribution.png'))
                plt.close(fig)

            """scale_intensive = calibration_intensive_margin(stock, retrofit_rate_ini, bill_saved, subsidies_total,
                                                           cost_insulation, delta_subsidies,
                                                           target_invest=0.2, utility_zil=utility_zil)

            self.constant_insulation_intensive *= scale_intensive
            self.pref_subsidy_insulation_int *= scale_intensive
            self.pref_investment_insulation_int *= scale_intensive
            self.pref_bill_insulation_int *= scale_intensive

            ms_after, _ = to_market_share(bill_saved, subsidies_total, cost_insulation,
                                              utility_zil=utility_zil)

            s = self.add_certificate(stock)
            stock_single = s.xs('Single-family', level='Housing type', drop_level=False)
            flow_retrofit = stock_single * reindex_mi(retrofit_rate_ini, stock_single.index)
            ms = ms_after.groupby([i for i in ms_after.index.names if i != 'Heating system final']).first()
            ms = reindex_mi(ms, flow_retrofit.index).dropna()
            flow_retrofit = flow_retrofit.reindex(ms.index)
            agg = (ms.T * flow_retrofit).T
            market_share_agg_after = (agg.sum() / agg.sum().sum()).reindex(ms_insulation.index)

            test = calibration_intensive_margin(stock, retrofit_rate_ini, bill_saved, subsidies_total,
                                                           cost_insulation, delta_subsidies,
                                                           target_invest=0.2, utility_zil=utility_zil)
            """

        # calculating market-shares
        if self._debug_mode:
            s = self.add_certificate(stock)
            stock_single = s.xs('Single-family', level='Housing type', drop_level=False)
            flow_retrofit = stock_single * reindex_mi(renovation_rate_ini, stock_single.index)
            ms = market_share.groupby([i for i in market_share.index.names if i != 'Heating system final']).first()
            ms = reindex_mi(ms, flow_retrofit.index).dropna()
            flow_retrofit = flow_retrofit.reindex(ms.index)
            agg = (ms.T * flow_retrofit).T
            market_share_agg = (agg.sum() / agg.sum().sum()).reindex(ms_insulation.index)
            self.market_share = market_share_agg

        # extensive margin
        bool_zil_ext, bool_zil = None, None
        if utility_zil is not None:
            bool_zil = utility_zil.copy()
            bool_zil[bool_zil > 0] = 1

        bill_saved_insulation, subsidies_insulation, investment_insulation = None, None, None
        if self._utility_extensive == 'market_share':
            bill_saved_insulation = (bill_saved.reindex(market_share.index) * market_share).sum(axis=1)
            subsidies_insulation = (subsidies_total.reindex(market_share.index) * market_share).sum(axis=1)
            investment_insulation = (cost_total.reindex(market_share.index) * market_share).sum(axis=1)
            if utility_zil is not None:
                bool_zil_ext = (bool_zil.reindex(market_share.index) * market_share).sum(axis=1)
        elif self._utility_extensive == 'max':
            def rename_tuple(tuple, names):
                idx = tuple.index
                tuple = DataFrame([[a, b, c, d] for a, b, c, d in tuple.values])
                tuple.columns = names
                for i in names:
                    tuple.loc[tuple[i] == True, i] = i
                    tuple.loc[tuple[i] == False, i] = ''
                return Series(list(zip(*(tuple[i] for i in names))), index=idx)

            columns = utility_intensive.idxmax(axis=1)
            """work_umax = rename_tuple(columns, utility_intensive.columns.names)
            utility_intensive_max = utility_intensive.max(axis=1)"""
            bill_saved_insulation, investment_insulation, subsidies_insulation = Series(dtype=float), Series(
                dtype=float), Series(dtype=float)
            if utility_zil is not None:
                bool_zil_ext = Series(dtype=float)

            for c in columns.unique():
                idx = columns.index[columns == c]
                bill_saved_insulation = concat((bill_saved_insulation, bill_saved.loc[idx, c]), axis=0)
                investment_insulation = concat((investment_insulation, cost_insulation.loc[idx, c]), axis=0)
                subsidies_insulation = concat((subsidies_insulation, subsidies_total.loc[idx, c]), axis=0)
                if utility_zil is not None:
                    bool_zil_ext = concat((bool_zil_ext, bool_zil.loc[idx, c]), axis=0)

            bill_saved_insulation.index = MultiIndex.from_tuples(bill_saved_insulation.index).set_names(
                bill_saved.index.names)
            investment_insulation.index = MultiIndex.from_tuples(investment_insulation.index).set_names(
                cost_insulation.index.names)
            subsidies_insulation.index = MultiIndex.from_tuples(subsidies_insulation.index).set_names(
                subsidies_total.index.names)
            if utility_zil is not None:
                bool_zil_ext.index = MultiIndex.from_tuples(bool_zil_ext.index).set_names(
                    bool_zil.index.names)

        idx = bill_saved_insulation[bill_saved_insulation <= 0].index
        bill_saved_insulation.drop(idx, inplace=True)
        subsidies_insulation.drop(idx, inplace=True)
        # More index in investment_insulation
        investment_insulation.drop(idx, inplace=True)
        if bool_zil_ext:
            bool_zil_ext.drop(idx, inplace=True)

        retrofit_rate, utility = to_retrofit_rate(bill_saved_insulation, subsidies_insulation, investment_insulation,
                                                  bool_zil=bool_zil_ext)

        if self.constant_insulation_extensive is None:
            self.logger.debug('Calibration renovation rate')
            if self._utility_extensive == 'market_share':
                delta_subsidies_sum = (delta_subsidies.reindex(market_share.index) * market_share).sum(axis=1)
            else:
                raise NotImplemented
            pref_subsidies = reindex_mi(self.pref_subsidy_insulation_ext, subsidies_insulation.index).rename(None)

            # graphic showing the impact of the scale in a general case
            if self._debug_mode:
                x, free_riders, elasticity = [], [], []
                for scale in arange(0.1, 5, 0.1):
                    x.append(scale)
                    free_riders.append(impact_subsidies(scale, utility, stock, - delta_subsidies_sum / 1000, pref_subsidies,
                                                        indicator='freeriders'))
                    elasticity.append(impact_subsidies(scale, utility, stock, subsidies_insulation / 1000 * 0.01, pref_subsidies,
                                                       indicator='elasticity'))

                graphs = {'Freeriders cite': free_riders}
                for name, data in graphs.items():
                    df = Series(data, index=Index(x, name='Scale'), name=name)
                    fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))
                    df.plot(ax=ax)
                    format_ax(ax, format_y=lambda y, _: '{:.0%}'.format(y), y_label=name)
                    save_fig(fig, save=os.path.join(self.path_calibration, 'scale_calibration_{}.png'.format(name.lower())))

            constant, scale = calibration_constant_scale_ext(utility, stock, renovation_rate_ini, target_freeriders,
                                                             delta_subsidies_sum, pref_subsidies)
            self.constant_insulation_extensive = constant

            # graphic showing the impact of the scale
            if self._debug_mode:
                stock_single = stock.xs('Single-family', level='Housing type', drop_level=False)
                stock_multi = stock.xs('Multi-family', level='Housing type', drop_level=False)
                x_before, y_before, y_before_single, y_before_multi = [], [], [], []
                for delta in arange(0, 2, 0.1):
                    sub = subsidies_insulation * (1 + delta)
                    x_before.append((sub * stock).sum() / stock.sum())
                    rate = to_retrofit_rate(bill_saved_insulation, sub, investment_insulation)[0]
                    y_before.append((rate * stock).sum() / stock.sum())
                    y_before_single.append((rate * stock_single).sum() / stock_single.sum())
                    y_before_multi.append((rate * stock_multi).sum() / stock_multi.sum())

            self.scale_ext = scale
            self.pref_subsidy_insulation_ext *= self.scale_ext
            self.pref_investment_insulation_ext *= self.scale_ext
            self.pref_bill_insulation_ext *= self.scale_ext
            self.pref_zil_ext *= self.scale_ext

            # graphic showing the impact of the scale
            if self._debug_mode:
                x_after, y_after, y_after_single, y_after_multi = [], [], [], []
                for delta in arange(0, 2, 0.1):
                    sub = subsidies_insulation * (1 + delta)
                    x_after.append((sub * stock).sum() / stock.sum())
                    rate = to_retrofit_rate(bill_saved_insulation, sub, investment_insulation)[0]
                    y_after.append((rate * stock).sum() / stock.sum())
                    y_after_single.append((rate * stock_single).sum() / stock_single.sum())
                    y_after_multi.append((rate * stock_multi).sum() / stock_multi.sum())

                df = concat(
                    (Series(x_before), Series(y_before), Series(y_after), Series(y_before_single),
                     Series(y_after_single), Series(y_before_multi),
                     Series(y_after_multi)), axis=1)

                df.columns = ['Subsidies (€)', 'Before', 'After', 'Before single', 'After single', 'Before multi',
                              'After multi']
                color = {'Before': 'black', 'After': 'black',
                         'Before single': 'darkorange', 'After single': 'darkorange',
                         'Before multi': 'royalblue', 'After multi': 'royalblue'
                         }
                style = ['--', '-'] * 10
                fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))
                df.plot(ax=ax, x='Subsidies (€)', color=color, style=style)
                format_ax(ax, format_y=lambda y, _: '{:.0%}'.format(y), y_label='Retrofit rate')
                format_legend(ax)
                save_fig(fig, save=os.path.join(self.path_calibration, 'scale_effect.png'))

            retrofit_rate, utility = to_retrofit_rate(bill_saved_insulation, subsidies_insulation,
                                                      investment_insulation, bool_zil=bool_zil_ext)

            impact_subsidies(1.0, utility, stock, - delta_subsidies_sum / 1000, self.pref_subsidy_insulation_ext)

            # graphics showing the distribution of retrofit rate after calibration
            if self._debug_mode:
                r = retrofit_rate.xs(False, level='Heater replacement').rename('')

                fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))
                r.plot.box(ax=ax)
                format_ax(ax, format_y=lambda y, _: '{:.0%}'.format(y))
                ax.set_xlabel('')
                ax.set_ylabel('Retrofit rate (%)')
                save_fig(fig, save=os.path.join(self.path_calibration_renovation, 'retrofit_rate_distribution.png'))

                certificate = self.certificate.groupby([l for l in self.stock.index.names if l != 'Income tenant']).first()
                certificate = reindex_mi(certificate, r.index)
                r.to_frame().groupby(certificate).boxplot(fontsize=12, figsize=(8, 10))
                plt.savefig(os.path.join(self.path_calibration_renovation, 'retrofit_rate_distribution_dpe.png'))
                plt.close()

                r = retrofit_rate.rename('')
                certificate = self.certificate.groupby([l for l in self.stock.index.names if l != 'Income tenant']).first()
                certificate = reindex_mi(certificate, r.index)
                temp = concat((r, certificate.rename('Performance')), axis=1).set_index('Performance', append=True)
                temp.groupby(renovation_rate_ini.index.names).describe().to_csv(
                    os.path.join(self.path_calibration_renovation, 'retrofit_rate_desription.csv'))

                consumption_sd = self.consumption_standard(r.index)[2]
                consumption_sd = reindex_mi(consumption_sd, index)

                consumption_sd = reindex_mi(consumption_sd, r.index)
                df = concat((consumption_sd, r), axis=1, keys=['Consumption', 'Retrofit rate'])

                make_plot(df.set_index('Consumption').squeeze().sort_index(), 'Retrofit rate (%)',
                          format_y=lambda x, _: '{:.0%}'.format(x),
                          save=os.path.join(self.path_calibration_renovation, 'retrofit_rate_consumption_calib.png'),
                          legend=False, integer=False)

                df.reset_index('Income owner', inplace=True)
                df['Income owner'] = df['Income owner'].replace(resources_data['colors'])
                for i in renovation_rate_ini.index:
                    if not i[2]:
                        d = df.xs(i[0], level='Housing type').xs(i[1], level='Occupancy status')
                        name = '{}_{}_{}'.format(i[0].lower(), i[1].lower(), str(i[2]).lower())
                        fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))
                        d.plot.scatter(ax=ax, x='Consumption', y='Retrofit rate', c=d['Income owner'])
                        format_ax(ax, format_y=lambda y, _: '{:.0%}'.format(y), y_label=name)
                        save_fig(fig, save=os.path.join(self.path_calibration_renovation, 'retrofit_rate_{}_calib.png'.format(name)))

            # file concatenating all calibration results
            scale = Series(self.scale_ext, index=['Scale'])
            constant_ext = self.constant_insulation_extensive.copy()
            constant_ext.index = constant_ext.index.to_flat_index()
            constant_int = self.constant_insulation_intensive.copy()
            constant_int.index = constant_int.index.to_flat_index()
            if isinstance(constant_int, DataFrame):
                constant_int = constant_int.stack(constant_int.columns.names)

            r = self.add_certificate(retrofit_rate)
            s = self.add_certificate(stock)
            flow_retrofit = r * s
            retrofit_rate_mean = flow_retrofit.sum() / s.sum()
            retrofit_rate_mean = Series(retrofit_rate_mean, index=['Retrofit rate mean (%)'])
            retrofit_calibrated = flow_retrofit.groupby(renovation_rate_ini.index.names).sum() / s.groupby(renovation_rate_ini.index.names).sum()
            retrofit_calibrated.index = retrofit_calibrated.index.to_flat_index()
            flow_retrofit = flow_retrofit.droplevel('Performance')
            flow_insulation = (flow_retrofit * market_share.T).T.sum()
            flow_insulation_agg, name = list(), ''
            for i in flow_insulation.index.names:
                flow_insulation_agg.append(flow_insulation.xs(True, level=i).sum())
                name = '{}{},'.format(name, i)
            name = Series('', index=[name])
            flow_insulation_agg = Series(flow_insulation_agg, index=flow_insulation.index.names)
            flow_insulation.index = flow_insulation.index.to_flat_index()
            flow_insulation_sum = Series(flow_insulation.sum(), index=['Replacement insulation'])
            ms_calibrated = flow_insulation / flow_insulation.sum()
            ms_calibrated.index = ms_calibrated.index.to_flat_index()
            percentage_intensive_margin = Series(percentage_intensive_margin,
                                                    index=['Percentage intensive margin'])
            result = concat((scale, constant_ext, retrofit_rate_mean, retrofit_calibrated, flow_insulation_sum,
                                flow_insulation_agg, name, constant_int, flow_insulation, ms_calibrated,
                                percentage_intensive_margin), axis=0)
            if self.path is not None:
                result.to_csv(os.path.join(self.path_calibration, 'result_calibration.csv'))

        return retrofit_rate, market_share

    @staticmethod
    def exogenous_retrofit(index, choice_insulation):
        """Format retrofit rate and market share for each segment.

        Global retrofit and retrofit rate to match exogenous numbers.
        Retrofit all heating system replacement dwelling.

        Parameters
        ----------
        index: MultiIndex
        choice_insulation: MultiIndex

        Returns
        -------
        Series
            Retrofit rate by segment.
        DataFrame
            Market-share by segment and insulation choice.
        """

        retrofit_rate = concat([Series(1, dtype=float, index=index)], keys=[True], names=['Heater replacement'])

        market_share = DataFrame(0, index=index, columns=choice_insulation)
        market_share.loc[:, (True, True, True, True)] = 1

        return retrofit_rate, market_share

    def remove_market_failures(self):
        """NotImplemented

        Returns
        -------

        """
        if self.constant_insulation_extensive is not None:

            if self._remove_market_failures['landlord']:
                c = self.constant_insulation_extensive.copy()
                c = c.reset_index('Occupancy status')
                c = c[c['Occupancy status'] == 'Owner-occupied'].drop('Occupancy status', axis=1).squeeze()
                c = concat((c, c, c), axis=0, keys=['Owner-occupied', 'Privately rented', 'Social housing'], names=['Occupancy status'])
                c = c.reorder_levels(self.constant_insulation_extensive.index.names)
                self.constant_insulation_extensive = c

            if self._remove_market_failures['multi-family']:
                c = self.constant_insulation_extensive.copy()
                c = c.reset_index('Housing type')
                c = c[c['Housing type'] == 'Single-family'].drop('Housing type', axis=1).squeeze()
                c = concat((c, c, c), axis=0, keys=['Single-family', 'Multi-family'], names=['Housing type'])
                c = c.reorder_levels(self.constant_insulation_extensive.index.names)
                self.constant_insulation_extensive = c

            if self._remove_market_failures['credit constraint']:
                self.pref_bill_insulation_int = pd.Series(self.pref_bill_insulation_int.loc['D10'], index=self.pref_bill_insulation_int.index)
                self.pref_bill_insulation_ext = pd.Series(self.pref_bill_insulation_ext.loc['D10'], index=self.pref_bill_insulation_ext.index)

    def flow_retrofit(self, prices, cost_heater, cost_insulation, policies_heater=None, policies_insulation=None,
                      ms_heater=None, ms_insulation=None, renovation_rate_ini=None, target_freeriders=None,
                      supply_constraint=False, financing_cost=None
                      ):
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
        ms_insulation: Series
        renovation_rate_ini: Series
        policies_heater: list
            List of policies for heating system.
        policies_insulation: list
            List of policies for insulation.
        target_freeriders: float, default None
            Number of freeriders in calibration year for the income tax credit.
        supply_constraint: bool, default False
        financing_cost: optional, dict

        Returns
        -------
        Series
        """

        if self._remove_market_failures is not None:
            self.remove_market_failures()

        stock = self.heater_replacement(prices, cost_heater, policies_heater, ms_heater=ms_heater)

        self.logger.debug('Agents: {:,.0f}'.format(stock.shape[0]))
        stock_existing = stock.xs(True, level='Existing', drop_level=False)
        retrofit_rate, market_share = self.insulation_replacement(prices, cost_insulation,
                                                                  ms_insulation=ms_insulation,
                                                                  renovation_rate_ini=renovation_rate_ini,
                                                                  policies_insulation=policies_insulation,
                                                                  target_freeriders=target_freeriders,
                                                                  index=stock_existing.index, stock=stock_existing,
                                                                  supply_constraint=supply_constraint,
                                                                  financing_cost=financing_cost)

        flow_only_heater = (1 - retrofit_rate.reindex(stock.index).fillna(0)) * stock
        flow_only_heater = flow_only_heater.xs(True, level='Heater replacement', drop_level=False).unstack('Heating system final')
        flow_only_heater_sum = flow_only_heater.sum().sum()

        flow = (retrofit_rate * stock).dropna()
        replacement_sum = flow.sum().sum()

        replaced_by = (flow * market_share.T).T

        assert round(replaced_by.sum().sum(), 0) == round(replacement_sum, 0), 'Sum problem'


        only_heater = (stock - flow.reindex(stock.index, fill_value=0)).xs(True, level='Heater replacement')
        certificate_jump = self.certificate_jump_heater.stack()
        rslt = {}
        l = unique(certificate_jump)
        for i in l:
            rslt.update({i: ((certificate_jump == i) * only_heater).sum()})
        self.certificate_jump_heater = Series(rslt).sort_index()
        if self.detailed_mode:
            self.store_information_retrofit(replaced_by)

        # removing heater replacement level
        replaced_by = replaced_by.groupby(
            [c for c in replaced_by.index.names if c != 'Heater replacement']).sum()
        flow_only_heater = flow_only_heater.groupby(
            [c for c in flow_only_heater.index.names if c != 'Heater replacement']).sum()
        # TODO perhaps can be optimized

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
        output['Consumption standard (TWh)'] = (self.heat_consumption_sd * self.stock).sum() / 10 ** 9

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

        output['Heating intensity (%)'] = self.heating_intensity_avg

        output['Energy poverty (Million)'] = self.energy_poverty / 10 ** 6

        temp = self.stock.groupby(self.certificate).sum()
        temp.index = temp.index.map(lambda x: 'Stock {} (Million)'.format(x))
        output.update(temp.T / 10 ** 6)
        try:
            output['Stock efficient (Million)'] = output['Stock A (Million)'] + output['Stock B (Million)']
        except KeyError:
            output['Stock efficient (Million)'] = output['Stock B (Million)']

        output['Stock low-efficient (Million)'] = output['Stock F (Million)'] + output['Stock G (Million)']

        if self.year > self.first_year:
            temp = self.retrofit_rate.dropna(how='all')
            temp = temp.groupby([i for i in temp.index.names if i not in ['Heating system final']]).mean()
            t = temp.xs(False, level='Heater replacement')

            s_temp = self.stock
            s_temp = s_temp.groupby([i for i in s_temp.index.names if i != 'Income tenant']).sum()

            # Weighted average with stock to calculate real retrofit rate
            output['Renovation rate (%)'] = ((t * s_temp).sum() / s_temp.sum())
            t_grouped = (t * s_temp).groupby(['Housing type', 'Occupancy status']).sum() / s_temp.groupby(
                ['Housing type',
                 'Occupancy status']).sum()
            t_grouped.index = t_grouped.index.map(lambda x: 'Renovation rate {} - {} (%)'.format(x[0], x[1]))
            output.update(t_grouped.T)

            """output['Non-weighted retrofit rate (%)'] = t.mean()
            t = t.groupby(['Housing type', 'Occupancy status']).mean()
            t.index = t.index.map(lambda x: 'Non-weighted retrofit rate {} - {} (%)'.format(x[0], x[1]))
            output.update(t.T)
            
            output['Non-weighted retrofit rate w/ heater (%)'] = t.mean()
            t = t.groupby(['Housing type', 'Occupancy status']).mean()
            t.index = t.index.map(lambda x: 'Non-weighted retrofit rate heater {} - {} (%)'.format(x[0], x[1]))
            output.update(t.T)
            """

            t = temp.xs(True, level='Heater replacement')
            s_temp = self.stock
            s_temp = s_temp.groupby([i for i in s_temp.index.names if i != 'Income tenant']).sum()
            output['Renovation rate w/ heater (%)'] = ((t * s_temp).sum() / s_temp.sum())

            t_grouped = (t * s_temp).groupby(['Housing type', 'Occupancy status']).sum() / s_temp.groupby(
                ['Housing type',
                 'Occupancy status']).sum()
            t_grouped.index = t_grouped.index.map(lambda x: 'Renovation rate heater {} - {} (%)'.format(x[0], x[1]))
            output.update(t_grouped.T)

            temp = self.gest_nb.copy()
            temp.index = temp.index.map(lambda x: 'Renovation types {} (Thousand households)'.format(x))
            output['Renovation (Thousand households)'] = temp.sum() / 10 ** 3
            output['Renovation with heater replacement (Thousand households)'] = self.retrofit_with_heater / 10 ** 3
            output['Replacement renovation (Thousand)'] = (self.gest_nb * self.gest_nb.index).sum() / 10 ** 3
            output.update(temp.T / 10 ** 3)
            output['Replacement total (Thousand)'] = output['Replacement renovation (Thousand)'] - output[
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

            # representative insulation investment: weighted average with number of insulation actions as weights
            if False:
                investment_insulation_repr = DataFrame(self.investment_insulation_repr_yrs)
                gest = DataFrame({year: item.sum(axis=1) for year, item in replacement_insulation.items()})
                gest = reindex_mi(gest, investment_insulation_repr.index)
                temp = gest * investment_insulation_repr

                t = temp.groupby('Income owner').sum() / gest.groupby('Income owner').sum()
                t.index = t.index.map(lambda x: 'Investment per insulation action {} (euro)'.format(x))
                output.update(t.T)

                t = temp.groupby(['Housing type', 'Occupancy status']).sum() / gest.groupby(['Housing type',
                                                                                             'Occupancy status']).sum()
                t.index = t.index.map(lambda x: 'Investment per insulation action {} - {} (euro)'.format(x[0], x[1]))
                output.update(t.T)

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

            subsidies, subsidies_count, sub_count = None, None, None
            for gest, subsidies_details in {'heater': self.subsidies_details_heater,
                                            'insulation': self.subsidies_details_insulation}.items():
                if gest == 'heater':
                    sub_count = Series(self.subsidies_count_heater)
                elif gest == 'insulation':
                    sub_count = Series(self.subsidies_count_insulation)

                subsidies_details = Series({k: i.sum().sum() for k, i in subsidies_details.items()}, dtype='float64')

                for i in subsidies_details.index:
                    output['{} {} (Thousand)'.format(i.capitalize().replace('_', ' '), gest)] = sub_count[i] / 10**3
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
                    output['{} (Thousand)'.format(i.capitalize().replace('_', ' '))] = subsidies_count.loc[i] / 10 ** 3
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
    def find_best_option(criteria, dict_df, func='max'):
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
        # carbon_emission = inputs['carbon_emission']
        # carbon_value = inputs['carbon_value_kwh']

        output = dict()

        if index is None:
            index = self.stock.index

        consumption_before = self.consumption_standard(index)[0]
        consumption_after, _, certificate_after = self.prepare_consumption(self._choice_insulation, index=index)
        consumption_saved = (consumption_before - consumption_after.T).T

        consumption_before = reindex_mi(consumption_before, index)
        consumption_after = reindex_mi(consumption_after, index)
        consumption_saved = reindex_mi(consumption_saved, index)

        consumption_actual_before = self.consumption_actual(prices.loc[self.year, :], consumption_before)
        consumption_actual_after = self.consumption_actual(prices.loc[self.year, :], consumption_after)
        consumption_actual_saved = (consumption_actual_before - consumption_actual_after.T).T

        consumption_before = (reindex_mi(self._surface, index) * consumption_before.T).T
        consumption_after = (reindex_mi(self._surface, index) * consumption_after.T).T
        consumption_saved = (reindex_mi(self._surface, index) * consumption_saved.T).T

        consumption_actual_before = (reindex_mi(self._surface, index) * consumption_actual_before.T).T
        consumption_actual_after = (reindex_mi(self._surface, index) * consumption_actual_after.T).T
        consumption_actual_saved = (reindex_mi(self._surface, index) * consumption_actual_saved.T).T

        output.update({'Stock (dwellings/segment)': self.stock,
                       'Surface (m2/segment)': self.stock * reindex_mi(self._surface, index),
                       'Consumption before (kWh/dwelling)': consumption_before,
                       'Consumption before (kWh/segment)': consumption_before * self.stock,
                       'Consumption actual before (kWh/dwelling)': consumption_actual_before,
                       'Consumption actual before (kWh/segment)': consumption_actual_before * self.stock,
                       'Consumption actual after (kWh/dwelling)': consumption_actual_after,
                       'Consumption actual after (kWh/segment)': (consumption_actual_after.T * self.stock).T,
                       'Consumption saved (kWh/dwelling)': consumption_saved,
                       'Consumption saved (kWh/segment)': (consumption_saved.T * self.stock).T,
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


    def calibration_exogenous(self, energy_prices, taxes, path_heater=None, path_insulation_int=None,
                              path_insulation_ext=None, scale=1.19651508552344):
        """Function calibrating buildings object with exogenous data.

        Parameters
        ----------
        energy_prices: Series
            Energy prices for year y. Index are energy carriers {'Electricity', 'Natural gas', 'Oil fuel', 'Wood fuel'}.
        taxes: Series
            Energy taxes for year y.
        """
        # calibration energy consumption first year
        self.calculate_consumption(energy_prices.loc[self.first_year, :], taxes)

        # calibration flow retrofit second year
        self.year = 2019

        if path_heater is not None:
            calibration_constant_heater = read_csv(path_heater, index_col=[0, 1, 2]).squeeze()
        else:
            calibration_constant_heater = get_pandas('project/input/calibration/calibration_constant_heater.csv',
                                                     lambda x: pd.read_csv(x, index_col=[0, 1, 2]).squeeze())
        self.constant_heater = calibration_constant_heater.unstack('Heating system final')
        self._choice_heater = list(self.constant_heater.columns)

        if path_insulation_int is not None:
            calibration_constant_insulation = read_csv(path_insulation_int, index_col=[0, 1, 2, 3]).squeeze()
        else:
            calibration_constant_insulation = get_pandas('project/input/calibration/calibration_constant_insulation.csv',
                                                         lambda x: pd.read_csv(x, index_col=[0, 1, 2, 3]).squeeze())
        self.constant_insulation_intensive = calibration_constant_insulation

        if path_insulation_ext is not None:
            calibration_constant_extensive = read_csv(path_insulation_ext, index_col=[0, 1, 2, 3]).squeeze()
        else:
            calibration_constant_extensive = get_pandas('project/input/calibration/calibration_constant_extensive.csv',
                                                         lambda x: pd.read_csv(x, index_col=[0, 1, 2, 3]).squeeze())
        self.constant_insulation_extensive = calibration_constant_extensive.dropna()

        self.scale_ext = scale






