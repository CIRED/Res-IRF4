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
import numpy as np
from utils import reindex_mi, timing
import thermal
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from utils import format_ax, save_fig, format_legend


class SegmentsIndex:
    """Not used.
    """
    def __init__(self, index, efficiency):
        self._efficiency = efficiency

        self.wall = pd.Series(index.get_level_values('Wall'), index=index)
        self.floor = pd.Series(index.get_level_values('Floor'), index=index)
        self.roof = pd.Series(index.get_level_values('Roof'), index=index)
        self.windows = pd.Series(index.get_level_values('Windows'), index=index)

        self.heating_system = pd.Series(index.get_level_values('Heating system'), index=index)
        self.energy = self.heating_system.str.split('-').str[0].rename('Energy')
        self.heater = self.heating_system.str.split('-').str[1]
        self.efficiency = pd.to_numeric(self.heater.replace(self._efficiency))


class ThermalBuildings:
    """ThermalBuildings classes.

    Parameters:
    ----------
    stock: pd.Series
        Building stock.
    surface: pd.Series
        Surface by dwelling type.
    param: dict
        Generic input.
    efficiency: pd.Series
        Heating system efficiency.
    income: pd.Series
    consumption_ini: pd.Series
    path: str
    year: int, default 2018
    data_calibration: default None

    Attributes:
    ----------

    stock_yrs: dict
        Dwellings by segment.
    surface_yrs: dict
        Surface by segment.
    heat_consumption_sd_yrs: dict
        Standard consumption (kWh) by segment.
    heat_consumption_yrs: dict
        Consumption (kWh) by segment before calibration.
    heat_consumption_energy_yrs: dict
        Consumption (kWh) by energy after calibration.
    budget_share_yrs: dict
        Budget share (%) by segment.
    heating_intensity_yrs: dict
        Heating intensity (%) by segment.
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
    def __init__(self, stock, surface, param, efficiency, income, consumption_ini, path, year=2018,
                 data_calibration=None):

        if isinstance(stock, pd.MultiIndex):
            stock = pd.Series(index=stock, dtype=float)

        self._efficiency = efficiency
        self._param = param
        self._dh = 55706
        self.path = path
        self.path_calibration = os.path.join(path, 'calibration')
        if not os.path.isdir(self.path_calibration):
            os.mkdir(self.path_calibration)

        self._consumption_ini = consumption_ini
        self.coefficient_consumption = None
        self._data_calibration = data_calibration

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
        self.stock_yrs, self.surface_yrs, self.heat_consumption_sd_yrs = {}, {}, {}
        self.budget_share_yrs, self.heating_intensity_yrs = {}, {}
        self.heating_intensity_tenant, self.heating_intensity_avg, self.energy_poverty = {}, {}, {}
        self.heat_consumption_yrs, self.heat_consumption_calib_yrs, self.heat_consumption_energy_yrs = {}, {}, {}
        self.energy_expenditure_yrs, self.energy_expenditure_energy_yrs = {}, {}
        self.taxes_expenditure_details, self.taxes_expenditure = {}, {}
        self.certificate_nb = {}

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
        stock: pd.Series

        Returns
        -------

        """

        self._stock = stock
        self.index = stock.index
        self.levels = list(stock.index.names)

        self.stock_mobile = stock - self._stock_residual.reindex(stock.index, fill_value=0)

        self.surface = reindex_mi(self._surface, stock.index)
        self.stock_surface = self.stock * self.surface
        self.income_owner = reindex_mi(self._income_owner, stock.index)
        if 'Income tenant' in stock.index.names:
            self.income_tenant = reindex_mi(self._income_tenant, stock.index)

        self.wall = pd.Series(stock.index.get_level_values('Wall'), index=stock.index)
        self.floor = pd.Series(stock.index.get_level_values('Floor'), index=stock.index)
        self.roof = pd.Series(stock.index.get_level_values('Roof'), index=stock.index)
        self.windows = pd.Series(stock.index.get_level_values('Windows'), index=stock.index)

        self.housing_type = pd.Series(stock.index.get_level_values('Housing type'), index=stock.index)

        self.heating_system = pd.Series(stock.index.get_level_values('Heating system'), index=stock.index)
        self.energy = self.heating_system.str.split('-').str[0].rename('Energy')
        self.heater = self.heating_system.str.split('-').str[1]
        self.efficiency = pd.to_numeric(self.heater.replace(self._efficiency))

        self.certificate = self.certificates()

        self.stock_yrs.update({self.year: self.stock})
        self.surface_yrs.update({self.year: self.stock * self.surface})
        self.heat_consumption_sd_yrs.update({self.year: self.stock * self.surface * self.heating_consumption_sd()})
        self.certificate_nb.update({self.year: self.stock.groupby(self.certificate).sum()})

    def new(self, stock):
        return ThermalBuildings(stock, self._surface_yrs, self._param, self._efficiency, self._income,
                                self._consumption_ini, self.path)

    def self_prepare(self, wall=None, floor=None, roof=None, windows=None, efficiency=None, energy=None):
        if wall is None:
            wall = self.wall
        if floor is None:
            floor = self.floor
        if roof is None:
            roof = self.roof
        if windows is None:
            windows = self.windows
        if efficiency is None:
            efficiency = self.efficiency
        if energy is None:
            energy = self.energy

        return wall, floor, roof, windows, efficiency, energy

    def heating_consumption_sd(self, wall=None, floor=None, roof=None, windows=None, efficiency=None):
        """Return standard heating consumption.

        Parameters
        ----------
        wall: pd.Series
        floor: pd.Series
        roof: pd.Series
        windows: pd.Series
        efficiency: pd.Series

        Returns
        -------
        pd.Series
        """
        wall, floor, roof, windows, efficiency, _ = self.self_prepare(wall=wall, floor=floor, roof=roof,
                                                                      windows=windows, efficiency=efficiency)
        return thermal.heating_consumption(wall, floor, roof, windows, self._dh, efficiency, self._param)

    def primary_heating_consumption_sd(self, wall=None, floor=None, roof=None, windows=None, efficiency=None,
                                       energy=None):
        wall, floor, roof, windows, efficiency, energy = self.self_prepare(wall=wall, floor=floor, roof=roof,
                                                                           windows=windows, efficiency=efficiency,
                                                                           energy=energy)
        return thermal.primary_heating_consumption(wall, floor, roof, windows, self._dh, efficiency, energy, self._param)
    
    def heating_consumption(self, prices):
        """Calculate actual space heating consumption based on standard space heating consumption and
        heating intensity.

        Space heating consumption is in kWh/year.
        Equation is based on Allibe (2012).

        Parameters
        ----------
        prices: pd.Series

        Returns
        -------
        pd.Series
        """
        energy_bill_sd = self.energy_bill_sd(prices)
        budget_share = energy_bill_sd / self.income_tenant
        heating_intensity = -0.191 * budget_share.apply(np.log) + 0.1105
        heat_consumption = self.heat_consumption_sd_yrs[self.year] * heating_intensity

        self.heating_intensity_yrs.update({self.year: heating_intensity})
        self.budget_share_yrs.update({self.year: budget_share})
        self.energy_poverty.update({self.year: self.stock[budget_share >= 0.1].sum()})
        self.heating_intensity_avg.update({self.year: (self.stock * heating_intensity).sum() / self.stock.sum()})
        self.heating_intensity_tenant.update({self.year: (self.stock * heating_intensity).groupby(
            'Income tenant').sum() / self.stock.groupby('Income tenant').sum()})

        return heat_consumption

    def calculate(self, prices, taxes):
        """Calculate energy indicators.

        Parameters
        ----------
        prices: pd.Series
        taxes: pd.Series

        Returns
        -------

        """
        self.heat_consumption_yrs.update({self.year: self.heating_consumption(prices)})
        heat_consumption_energy = self.heat_consumption_yrs[self.year].groupby(self.energy).sum()
        if self.coefficient_consumption is None:

            consumption = pd.concat((self.heat_consumption_yrs[self.year], self.energy), axis=1).groupby(
                ['Housing type', 'Energy']).sum().iloc[:, 0] / 10**9

            # considering 20% of electricity got wood stove - 50% electricity
            electricity_wood = 0.2 * consumption[('Single-family', 'Electricity')] * 1
            consumption[('Single-family', 'Wood fuel')] += electricity_wood
            consumption[('Single-family', 'Electricity')] -= electricity_wood
            consumption.groupby('Energy').sum()

            self.heat_consumption_yrs[self.year].groupby('Housing type').sum() / 10**9

            validation = dict()

            # stock initial
            temp = pd.concat((self.stock, self.energy), axis=1).groupby(
                ['Housing type', 'Energy']).sum().iloc[:, 0] / 10**3
            temp.index = temp.index.map(lambda x: 'Stock {} {} (Thousands)'.format(x[0], x[1]))
            validation.update(temp)
            temp = self.stock.groupby('Housing type').sum() / 10**3
            temp.index = temp.index.map(lambda x: 'Stock {} (Thousands)'.format(x))
            validation.update(temp)
            validation.update({'Stock (Thousands)': self.stock.sum() / 10**3})

            # surface intial
            temp = pd.concat((self.stock_surface, self.energy), axis=1).groupby(
                ['Housing type', 'Energy']).sum().iloc[:, 0] / 10**6
            temp.index = temp.index.map(lambda x: 'Surface {} {} (Million m2)'.format(x[0], x[1]))
            validation.update(temp)
            temp = self.stock_surface.groupby('Housing type').sum() / 10**6
            temp.index = temp.index.map(lambda x: 'Surface {} (Million m2)'.format(x))
            validation.update(temp)
            validation.update({'Surface (Million m2)': self.stock_surface.sum() / 10**6})

            # heating consumption initial
            temp = pd.concat((self.heat_consumption_yrs[self.year], self.energy), axis=1).groupby(
                ['Housing type', 'Energy']).sum().iloc[:, 0] / 10**9
            temp.index = temp.index.map(lambda x: 'Consumption {} {} (TWh)'.format(x[0], x[1]))
            validation.update(temp)
            temp = self.heat_consumption_yrs[self.year].groupby('Housing type').sum() / 10**9
            temp.index = temp.index.map(lambda x: 'Consumption {} (TWh)'.format(x))
            validation.update(temp)
            validation.update({'Consumption (TWh)': self.heat_consumption_yrs[self.year].sum() / 10**9})

            self.coefficient_consumption = self._consumption_ini * 10**9 / heat_consumption_energy

            temp = self.coefficient_consumption.copy()
            temp.index = temp.index.map(lambda x: 'Coefficient calibration {} (%)'.format(x))
            validation.update(temp)

            temp = heat_consumption_energy / 10**9
            temp.index = temp.index.map(lambda x: 'Consumption {} (TWh)'.format(x))
            validation.update(temp)

            validation = pd.Series(validation)
            if self._data_calibration is not None:
                validation = pd.concat((validation, self._data_calibration), keys=['Calcul', 'Data'], axis=1)
                validation['Error'] = (validation['Calcul'] - validation['Data']) / validation['Data']

            validation.round(2).to_csv(os.path.join(self.path_calibration, 'validation_stock.csv'))

        # self.heat_consumption_calib_yrs
        coefficient = self.coefficient_consumption.reindex(self.energy).set_axis(self.index, axis=0)
        self.heat_consumption_calib_yrs.update({self.year: (coefficient * self.heat_consumption_yrs[self.year]).copy()})
        self.heat_consumption_energy_yrs.update({self.year: self.heat_consumption_calib_yrs[self.year].groupby(self.energy).sum()})

        prices_reindex = prices.reindex(self.energy).set_axis(self.index, axis=0)
        self.energy_expenditure_yrs.update({self.year: prices_reindex * self.heat_consumption_calib_yrs[self.year]})
        self.energy_expenditure_energy_yrs.update({self.year: self.energy_expenditure_yrs[self.year].groupby(self.energy).sum()})

        total_taxes = pd.Series(0, index=prices.index)
        for tax in taxes:
            if tax.name not in self.taxes_expenditure_details.keys():
                self.taxes_expenditure_details[tax.name] = {}
            if self.year in tax.value.index:
                amount = tax.value.loc[self.year, :] * heat_consumption_energy
                self.taxes_expenditure_details[tax.name].update({self.year: amount})
                total_taxes += amount

        self.taxes_expenditure.update({self.year: total_taxes})

    def certificates(self, wall=None, floor=None, roof=None, windows=None, efficiency=None, energy=None):
        wall, floor, roof, windows, efficiency, energy = self.self_prepare(wall=wall, floor=floor, roof=roof,
                                                                           windows=windows, efficiency=efficiency,
                                                                           energy=energy)
        return thermal.certificate_buildings(wall, floor, roof, windows, self._dh, efficiency, energy, self._param)[1]

    def energy_prices(self, prices):
        prices = prices.reindex(self.energy)
        prices.index = self.index
        return prices

    def energy_bill_sd(self, prices):
        return self.heating_consumption_sd() * self.surface * self.energy_prices(prices)


class AgentBuildings(ThermalBuildings):

    """Class AgentBuildings represents thermal dynamic building stock.


    Attributes
    ----------
    pref_inertia:  float or pd.Series

    pref_investment_insulation: float or pd.Series
    pref_bill_insulation: float or pd.Series
    pref_subsidy_insulation: float or pd.Series


    cost_insulation: pd.DataFrame
        Cost by segment and by insulation choice (€).
    investment_insulation: pd.DataFrame
        Investment realized by segment and by insulation choice (€).
    tax_insulation: pd.DataFrame
        Tax by segment and by insulation choice (€).
    certificate_jump: pd.DataFrame
        Number of jump of energy performance certificate.
    retrofit_rate: dict


    """

    def __init__(self, stock, surface, param, efficiency, income, consumption_ini, path, preferences, restrict_heater,
                 ms_heater, choice_insulation, performance_insulation, demolition_rate=0.0, year=2018,
                 data_calibration=None, endogenous=True, number_exogenous=300000):
        super().__init__(stock, surface, param, efficiency, income, consumption_ini, path, year=year,
                         data_calibration=data_calibration)

        self.vta = 0.1
        self.factor_etp = 7.44 / 10**6 # ETP/€
        self.lifetime_insulation = 30

        self.pref_investment_heater = preferences['heater']['investment'].copy()
        self.pref_investment_insulation_int = preferences['insulation']['investment'].copy()
        self.pref_investment_insulation_ext = preferences['insulation']['investment'].copy()

        self.pref_subsidy_heater = preferences['heater']['subsidy']
        self.pref_subsidy_insulation_int = preferences['insulation']['subsidy'].copy()
        self.pref_subsidy_insulation_ext = preferences['insulation']['subsidy'].copy()

        self.pref_bill_heater = preferences['heater']['bill_saved'].copy()
        self.pref_bill_insulation_int = preferences['insulation']['bill_saved'].copy()
        self.pref_bill_insulation_ext = preferences['insulation']['bill_saved'].copy()

        self.pref_inertia = preferences['heater']['inertia']
        self.pref_zil_int = preferences['insulation']['zero_interest_loan']
        self.pref_zil_ext = preferences['insulation']['zero_interest_loan']

        self.discount_rate = - self.pref_investment_insulation_ext / self.pref_bill_insulation_ext
        self.discount_factor = (1 - (1 + self.discount_rate) ** -self.lifetime_insulation) / self.discount_rate

        _ext = None
        self.calibration_scale = 'cite'
        self.param_supply = None
        self.capacity_utilization = None
        self.factor_yrs = {}

        self._demolition_rate = demolition_rate
        self._demolition_total = (stock * self._demolition_rate).sum()
        self._target_demolition = ['E', 'F', 'G']

        self._choice_heater = list(ms_heater.columns)
        self._restrict_heater = restrict_heater
        self._ms_heater = ms_heater
        self._endogenous = endogenous
        self._probability_replacement = None

        self._target_exogenous = ['F', 'G']
        self._market_share_exogenous = None
        self._number_exogenous = number_exogenous

        self._choice_insulation = choice_insulation
        self._performance_insulation = performance_insulation
        # TODO: clean assign by housing type (mean is done)
        self.surface_insulation = pd.Series({'Wall': param['ratio_surface']['Wall'].mean(),
                                             'Floor': param['ratio_surface']['Floor'].mean(),
                                             'Roof': param['ratio_surface']['Roof'].mean(),
                                             'Windows': param['ratio_surface']['Windows'].mean()})

        self.utility_insulation_extensive, self.utility_insulation_intensive, self.constant_heater = None, None, None
        self.utility_yrs = {}

        self.certificate_jump, self.certificate_jump_yrs = None, {}
        self.efficient_renovation, self.efficient_renovation_yrs = None, {}

        self.replacement_heater, self.heater_replaced = {}, {}
        self.cost_heater, self.investment_heater = {}, {}
        self.tax_heater = {}
        self.subsidies_details_heater, self.subsidies_heater_indiv, self.subsidies_heater = {}, {}, {}

        self.market_share_yrs = {}

        self.replacement_insulation, self.retrofit_rate = {}, {}
        self.cost_component, self.cost_insulation_indiv, self.investment_insulation = {}, {}, {}
        self.tax_insulation, self.taxed_insulation = {}, {}
        self.subsidies_details_insulation, self.subsidies_insulation_indiv, self.subsidies_insulation = {}, {}, {}

        self.bill_saved_indiv = {}
        self.bill_saved_mean, self.bill_saved_efficient = {}, {}
        self.investment_insulation_mean, self.investment_insulation_efficient = {}, {}
        self.subsidies_insulation_mean, self.subsidies_insulation_efficient = {}, {}
        self.bill_saved_repr, self.subsidies_insulation_repr, self.investment_insulation_repr = {}, {}, {}

        self.private_npv_mean, self.private_npv_efficient = {}, {}

        self._share_decision_maker = stock.groupby(
            ['Occupancy status', 'Housing type', 'Income owner', 'Income tenant']).sum().unstack(
            ['Occupancy status', 'Income owner', 'Income tenant'])
        self._share_decision_maker = (self._share_decision_maker.T / self._share_decision_maker.sum(axis=1)).T

    def add_flows(self, flows):
        """Update stock attribute by adding flow series.

        Parameters
        ----------
        flows: pd.Series, list
        """
        flow_total = None
        if isinstance(flows, pd.Series):
            flow_total = flows
        if isinstance(flows, list):
            for flow in flows:
                if flow_total is None:
                    flow_total = flow.copy()
                else:
                    union = flow.index.union(flow_total.index)
                    flow_total = flow.reindex(union, fill_value=0) + flow_total.reindex(union, fill_value=0)

        union = flow_total.index.union(self.index)
        stock = flow_total.reindex(union, fill_value=0) + self.stock.reindex(union, fill_value=0)
        stock[stock < 0] = 0
        self.stock = stock

    def endogenous_market_share_heater(self, index, prices, subsidies_total, cost_heater, ms_heater):

        choice_heater = self._choice_heater
        choice_heater_idx = pd.Index(choice_heater, name='Heating system final')
        agent = self.new(pd.Series(index=index, dtype=float))

        efficiency = pd.to_numeric(
            pd.Series(choice_heater).str.split('-').str[1].replace(self._efficiency)).set_axis(
            choice_heater_idx)
        energy = pd.Series(choice_heater).str.split('-').str[0].set_axis(choice_heater_idx)
        heat_consumption_sd = agent.heating_consumption_sd(efficiency=efficiency)
        prices_re = prices.reindex(energy).set_axis(heat_consumption_sd.columns)
        energy_bill_sd = ((heat_consumption_sd * prices_re).T * agent.surface).T
        bill_saved = - energy_bill_sd.sub(agent.energy_bill_sd(prices), axis=0)
        utility_bill_saving = (bill_saved.T * reindex_mi(self.pref_bill_heater, bill_saved.index)).T / 1000
        utility_bill_saving = utility_bill_saving.loc[:, choice_heater]

        utility_subsidies = subsidies_total * self.pref_subsidy_heater / 1000

        cost_heater = cost_heater.reindex(utility_bill_saving.columns)
        pref_investment = reindex_mi(self.pref_investment_heater, utility_bill_saving.index).rename(None)
        utility_investment = pref_investment.to_frame().dot(cost_heater.to_frame().T) / 1000

        utility_inertia = pd.DataFrame(0, index=utility_bill_saving.index, columns=utility_bill_saving.columns)
        for hs in choice_heater:
            utility_inertia.loc[
                utility_inertia.index.get_level_values('Heating system') == hs, hs] = self.pref_inertia

        utility = utility_inertia + utility_investment + utility_bill_saving + utility_subsidies

        restrict_heater = reindex_mi(self._restrict_heater.reindex(utility.columns, axis=1), utility.index).astype(
            bool)
        utility[restrict_heater] = float('nan')

        if self.constant_heater is None:
            self.constant_heater = self.calibration_constant_heater(utility, ms_heater)
        utility_constant = reindex_mi(self.constant_heater.reindex(utility.columns, axis=1), utility.index)

        utility += utility_constant
        market_share = (np.exp(utility).T / np.exp(utility).sum(axis=1)).T

        return market_share

    def exogenous_market_share_heater(self, index, choice_heater_idx):
        """Define exogenous market-share.

        Market-share is defined by _market_share_exogenous attribute.
        Replacement

        Parameters
        ----------
        index: pd.MultiIndex
        choice_heater_idx: Index

        Returns
        -------
        pd.DataFrame
            Market-share by segment and possible heater choice.
        pd.Series
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

        market_share = pd.Series(index=index, dtype=float).to_frame().dot(
            pd.Series(index=choice_heater_idx, dtype=float).to_frame().T)

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
        # (probability_replacement * self.stock_mobile.groupby(index.names).sum()).sum()
        probability_replacement = probability_replacement.reindex(market_share.index)
        return market_share, probability_replacement

    @timing
    def heater_replacement(self, prices, cost_heater, ms_heater, policies_heater, probability_replacement=1/20,
                           index=None):
        """Function returns new building stock after heater replacement.

        Parameters
        ----------
        prices: pd.Series
        cost_heater: pd.Series
        ms_heater: pd.DataFrame
        policies_heater: list
        probability_replacement: float or pd.Series, default 1/17
        index: pd.MultiIndex optional, default None

        Returns
        -------
        pd.Series
        """

        if isinstance(probability_replacement, float):
            probability_replacement = pd.Series(len(self._choice_heater)*[probability_replacement],
                                                pd.Index(self._choice_heater, name='Heating system final') )

        self._probability_replacement = probability_replacement

        if index is None:
            index = self.index
            index = index.droplevel('Income tenant')
            index = index[~index.duplicated()]

        # prohibited energies can be a string or a list of strings
        energy_regulations = [policy for policy in policies_heater if policy.policy == 'heater_regulation']
        prohibited_energies = pd.Series(list(np.array([policy.name.replace('_elimination', "").replace("_", " ").capitalize()
                                             for policy in energy_regulations]).flat), index=[policy.name for policy in energy_regulations])

        for regulation in energy_regulations:
            if regulation.value is not None:
                heater = next(x for x in self._choice_heater if prohibited_energies[regulation.name] in x)
                probability_replacement[heater] = regulation.value

        self._probability_replacement = probability_replacement

        list_heaters = self._choice_heater
        for energy in prohibited_energies:
            list_heaters = list(set(list_heaters) & set([heater for heater in self._choice_heater if energy not in heater]))

        if energy_regulations:
            choice_heater_idx = pd.Index(list_heaters, name='Heating system final')
        else:
            choice_heater_idx = pd.Index(self._choice_heater, name='Heating system final')

        frame = pd.Series(dtype=float, index=index).to_frame().dot(
            pd.Series(dtype=float, index=choice_heater_idx).to_frame().T)
        cost_heater, tax_heater, subsidies_details, subsidies_total = self.apply_subsidies_heater(policies_heater,
                                                                                                  cost_heater.copy(),
                                                                                                  frame)
        if self._endogenous:
            subsidies_utility = subsidies_total.copy()
            if 'reduced_tax' in subsidies_details.keys():
                subsidies_utility -= subsidies_details['reduced_tax']
            market_share = self.endogenous_market_share_heater(index, prices, subsidies_utility, cost_heater, ms_heater)

            #je pense que ce if est inutile, c'est déja
            if isinstance(probability_replacement, pd.Series):
                pass
                #probability_replacement = reindex_mi(probability_replacement, market_share.index)

        else:
            market_share, probability_replacement = self.exogenous_market_share_heater(index, choice_heater_idx)

        replacement = ((market_share * probability_replacement ).T * self.stock_mobile.groupby(
            market_share.index.names).sum()).T

        stock_replacement = replacement.stack('Heating system final')
        to_replace = replacement.sum(axis=1)
        stock = self.stock_mobile.groupby(to_replace.index.names).sum() - to_replace
        stock = pd.concat((stock, pd.Series(stock.index.get_level_values('Heating system'), index=stock.index,
                                            name='Heating system final')), axis=1).set_index('Heating system final',
                                                                                             append=True).squeeze()
        stock = pd.concat((stock.reorder_levels(stock_replacement.index.names), stock_replacement),
                          axis=0, keys=[False, True], names=['Heater replacement'])

        replaced_by = stock.droplevel('Heating system').rename_axis(index={'Heating system final': 'Heating system'})

        self.store_information_heater(cost_heater, subsidies_total, subsidies_details, replacement, tax_heater,
                                      replaced_by)

        return stock

    def apply_subsidies_heater(self, policies_heater, cost_heater, frame):
        """Calculate subsidies for each dwelling and each heating system.

        Parameters
        ----------
        policies_heater: list
        cost_heater: pd.Series
        frame: pd.DataFrame
            Index matches segments and columns heating system.

        Returns
        -------

        """

        subsidies_total = pd.DataFrame(0, index=frame.index, columns=frame.columns)
        subsidies_details = {}

        tax = self.vta
        p = [p for p in policies_heater if 'reduced_tax' == p.policy]
        if p:
            tax = p[0].value
            sub = cost_heater * (self.vta - tax)
            subsidies_details.update({'reduced_tax': pd.concat([sub] * frame.shape[0], keys=frame.index, axis=1).T})
            subsidies_total += subsidies_details['reduced_tax']

        tax_heater = cost_heater * tax
        cost_heater += tax_heater

        sub = None
        for policy in policies_heater:
            if policy.policy == 'subsidy_target':
                sub = policy.value.reindex(frame.columns, axis=1).fillna(0)
                sub = reindex_mi(sub, frame.index)
            elif policy.policy == 'subsidy_ad_volarem':

                if isinstance(policy.value, (float, int)):
                    sub = policy.value * cost_heater
                    sub = pd.concat([sub] * frame.shape[0], keys=frame.index, axis=1).T

                if isinstance(policy.value, pd.DataFrame):
                    sub = policy.value * cost_heater
                    sub = reindex_mi(sub, frame.index).fillna(0)

                if isinstance(policy.value, pd.Series):
                    sub = policy.value.to_frame().dot(cost_heater.to_frame().T)
                    sub = reindex_mi(sub, frame.index).fillna(0)

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
        cost_heater: pd.Series
            Cost of each heating system (€).
        subsidies_total: pd.DataFrame
            Total amount of eligible subsidies by dwelling and heating system (€).
        subsidies_details: dict
            Amount of eligible subsidies by dwelling and heating system (€).
        replacement: pd.DataFrame
            Number of heating system replacement by dwelling and heating system chosen.
        tax_heater: pd.Series
            VTA tax of each heating system (€).
        replaced_by: pd.Series
            Dwelling updated with a new heating system.
        """
        # information stored during
        self.cost_heater.update({self.year: cost_heater})
        self.subsidies_heater_indiv.update({self.year: subsidies_total})
        self.subsidies_details_heater.update({self.year: subsidies_details})
        self.replacement_heater.update({self.year: replacement})
        self.investment_heater.update({self.year: replacement * cost_heater})
        self.tax_heater.update({self.year: replacement * tax_heater})
        self.subsidies_heater.update({self.year: replacement * subsidies_total})

        for key in self.subsidies_details_heater[self.year].keys():
            self.subsidies_details_heater[self.year][key] *= replacement
        self.heater_replaced.update({self.year: replaced_by})

    @timing
    def calibration_constant_heater(self, utility, ms_heater):
        """Constant to match the observed market-share.

        Market-share is defined by initial and final heating system.

        Parameters
        ----------
        utility: pd.DataFrame
        ms_heater: pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """

        # removing unnecessary level
        utility_ref = utility.droplevel(['Occupancy status']).copy()
        utility_ref = utility_ref[~utility_ref.index.duplicated(keep='first')]

        stock = self.stock.groupby(utility_ref.index.names).sum()

        # initializing constant to 0
        constant = ms_heater.copy()
        constant[constant > 0] = 0
        market_share_ini, market_share_agg = None, None
        for i in range(100):
            constant.loc[:, 'Wood boiler'] = 0
            utility_constant = reindex_mi(constant.reindex(utility_ref.columns, axis=1), utility.index)
            utility = utility_ref + utility_constant
            market_share = (np.exp(utility).T / np.exp(utility).sum(axis=1)).T
            agg = (market_share.T * stock).T.groupby('Heating system').sum()
            market_share_agg = (agg.T / agg.sum(axis=1)).T
            if i == 0:
                market_share_ini = market_share_agg.copy()
            constant = constant + np.log(ms_heater / market_share_agg)

            ms_heater = ms_heater.reindex(market_share_agg.index)

            if (market_share_agg.round(decimals=3) == ms_heater.round(decimals=3).fillna(0)).all().all():
                print('Constant heater optim worked')
                break

        constant.loc[:, 'Wood boiler'] = 0
        details = pd.concat((constant.stack(), market_share_ini.stack(), market_share_agg.stack(), ms_heater.stack()),
                            axis=1, keys=['constant', 'calcul ini', 'calcul', 'observed']).round(decimals=3)
        details.to_csv(os.path.join(self.path_calibration, 'calibration_constant_heater.csv'))

        return constant

    @timing
    def prepare_consumption(self, choice_insulation=None, performance_insulation=None, index=None,
                            levels_heater='Heating system'):
        """Constitute building components' performance.

        Returns
        -------
        pd.DataFrame
        pd.DataFrame
        """

        if index is None:
            index = self.index

        if not isinstance(choice_insulation, pd.MultiIndex):
            choice_insulation = self._choice_insulation

        if not isinstance(performance_insulation, pd.MultiIndex):
            performance_insulation = self._performance_insulation

        # only selecting useful levels
        idx = index.copy()
        idx = idx.droplevel([i for i in idx.names if i not in ['Housing type', 'Wall', 'Floor', 'Roof', 'Windows'] + [levels_heater]])
        idx = idx[~idx.duplicated()]

        s = pd.concat([pd.Series(index=idx, dtype=float)] * len(choice_insulation), axis=1).set_axis(choice_insulation, axis=1)

        wall_buildings, floor_buildings, roof_buildings, windows_buildings = {}, {}, {}, {}
        for name, series in s.iteritems():
            wall, floor, roof, windows = name
            if wall:
                wall_buildings[name] = pd.Series(performance_insulation['Wall'], index=series.index)
            else:
                wall_buildings[name] = pd.Series(series.index.get_level_values('Wall'), index=series.index)

            if floor:
                floor_buildings[name] = pd.Series(performance_insulation['Floor'], index=series.index)
            else:
                floor_buildings[name] = pd.Series(series.index.get_level_values('Floor'), index=series.index)

            if roof:
                roof_buildings[name] = pd.Series(performance_insulation['Roof'], index=series.index)
            else:
                roof_buildings[name] = pd.Series(series.index.get_level_values('Roof'), index=series.index)

            if windows:
                windows_buildings[name] = pd.Series(performance_insulation['Windows'], index=series.index)
            else:
                windows_buildings[name] = pd.Series(series.index.get_level_values('Windows'), index=series.index)

        heating_system = pd.Series(idx.get_level_values(levels_heater), index=idx)
        energy = heating_system.str.split('-').str[0].rename('Energy')
        heater = heating_system.str.split('-').str[1]
        efficiency = pd.to_numeric(heater.replace(self._efficiency))

        consumption_sd, certificate = {}, {}
        for name in choice_insulation:
            consumption_sd[name] = self.heating_consumption_sd(wall=wall_buildings[name],
                                                               floor=floor_buildings[name],
                                                               roof=roof_buildings[name],
                                                               windows=windows_buildings[name],
                                                               efficiency=efficiency)

            certificate[name] = self.certificates(wall=wall_buildings[name],
                                                  floor=floor_buildings[name],
                                                  roof=roof_buildings[name],
                                                  windows=windows_buildings[name],
                                                  energy=energy,
                                                  efficiency=efficiency
                                                  )

        consumption_sd = pd.DataFrame(consumption_sd).rename_axis(choice_insulation.names, axis=1)
        consumption_sd = reindex_mi(consumption_sd, index)
        certificate = pd.DataFrame(certificate).rename_axis(choice_insulation.names, axis=1)
        certificate = reindex_mi(certificate, index)
        return consumption_sd, certificate

    @timing
    def prepare_cost_insulation(self, cost_insulation):
        """Constitute insulation choice set cost. Cost is equal to the sum of each individual cost component.

        Parameters
        ----------
        cost_insulation: pd.Series

        Returns
        -------
        pd.Series
            Multiindex series. Levels are Wall, Floor, Roof and Windows and values are boolean.
        """
        cost = pd.Series(0, index=self._choice_insulation)
        idx = pd.IndexSlice
        cost.loc[idx[True, :, :, :]] = cost.loc[idx[True, :, :, :]] + cost_insulation['Wall']
        cost.loc[idx[:, True, :, :]] = cost.loc[idx[:, True, :, :]] + cost_insulation['Floor']
        cost.loc[idx[:, :, True, :]] = cost.loc[idx[:, :, True, :]] + cost_insulation['Roof']
        cost.loc[idx[:, :, :, True]] = cost.loc[idx[:, :, :, True]] + cost_insulation['Windows']
        return cost

    @timing
    def prepare_subsidy_insulation(self, subsidies_insulation):
        """Constitute insulation choice set subsidies. Subsidies are equal to the sum of each individual subsidy.

        Parameters
        ----------
        subsidies_insulation: pd.DataFrame

        Returns
        -------
        pd.DataFrame
            Multiindex columns. Levels are Wall, Floor, Roof and Windows and values are boolean.
        """
        subsidy = pd.DataFrame(0, index=subsidies_insulation.index, columns=self._choice_insulation)
        idx = pd.IndexSlice
        subsidy.loc[:, idx[True, :, :, :]] = subsidy.loc[:, idx[True, :, :, :]].add(
            subsidies_insulation['Wall'], axis=0) * self.surface_insulation['Wall']
        subsidy.loc[:, idx[:, True, :, :]] = subsidy.loc[:, idx[:, True, :, :]].add(
            subsidies_insulation['Floor'], axis=0) * self.surface_insulation['Floor']
        subsidy.loc[:, idx[:, :, True, :]] = subsidy.loc[:, idx[:, :, True, :]].add(
            subsidies_insulation['Roof'], axis=0) * self.surface_insulation['Roof']
        subsidy.loc[:, idx[:, :, :, True]] = subsidy.loc[:, idx[:, :, :, True]].add(
            subsidies_insulation['Windows'], axis=0) * self.surface_insulation['Windows']
        return subsidy

    @timing
    def endogenous_retrofit(self, index, prices, subsidies_total, cost_insulation, ms_insulation=None, ms_extensive=None,
                            utility_zil=None, stock=None, supply_constraint=False, delta_subsidies=None,
                            target_freeriders=0.85, detailed=True):
        """Calculate endogenous retrofit based on discrete choice model.

        Utility variables are investment cost, energy bill saving, and subsidies.
        Preferences are object attributes defined initially.

        Parameters
        ----------
        index: pd.MultiIndex
        prices: pd.Series
        subsidies_total: pd.DataFrame
        cost_insulation: pd.DataFrame
        ms_insulation: pd.Series, default None
        ms_extensive: pd.Series, default None
        utility_zil: pd.DataFrame, default None
        stock: pd.Series, default None
        supply_constraint: bool
        delta_subsidies: pd.DataFrame, default None
        detailed: bool, default True
        target_freeriders: float, default 0.85

        Returns
        -------
        pd.Series
            Retrofit rate
        pd.DataFrame
            Market-share insulation
        """

        def to_market_share(bill_saved, subsidies, investment, utility_zil=utility_zil):

            pref_subsidies = reindex_mi(self.pref_subsidy_insulation_int, subsidies.index).rename(None)
            utility_subsidies = (subsidies.T * pref_subsidies).T / 1000

            pref_investment = reindex_mi(self.pref_investment_insulation_int, investment.index).rename(None)
            utility_investment = (cost_insulation.T * pref_investment).T / 1000

            utility_bill_saving = (bill_saved.T * reindex_mi(self.pref_bill_insulation_int, bill_saved.index)).T / 1000

            utility_intensive = utility_bill_saving + utility_investment + utility_subsidies

            if utility_zil is not None:
                utility_zil[utility_zil > 0] = self.pref_zil_int
                utility_intensive += utility_zil

            if self.utility_insulation_intensive is not None:
                utility_intensive += self.utility_insulation_intensive

            market_share = (np.exp(utility_intensive).T / np.exp(utility_intensive).sum(axis=1)).T
            return market_share, utility_intensive

        def to_retrofit_rate(bill_saved, subsidies, investment, bool_zil_mean=None, detailed=True):
            utility_bill_saving = reindex_mi(self.pref_bill_insulation_ext, bill_saved.index) * bill_saved / 1000

            pref_subsidies = reindex_mi(self.pref_subsidy_insulation_ext, subsidies.index).rename(None)
            utility_subsidies = (pref_subsidies * subsidies) / 1000

            pref_investment = reindex_mi(self.pref_investment_insulation_ext, investment.index).rename(None)
            utility_investment = (pref_investment * investment) / 1000

            utility = utility_investment + utility_bill_saving + utility_subsidies

            if bool_zil_mean is not None:
                utility_zil = bool_zil_mean * self.pref_zil_ext
                utility += utility_zil

            if self.utility_insulation_extensive is not None:
                utility_constant = reindex_mi(self.utility_insulation_extensive, utility.index)
                utility += utility_constant

            retrofit_rate = 1 / (1 + np.exp(- utility))

            if detailed and self.utility_insulation_extensive is not None:
                levels = self.utility_insulation_extensive.index.names
                df = pd.concat((utility_investment.groupby(
                    levels).mean(), utility_subsidies.groupby(levels).mean(), utility_bill_saving.groupby(
                    levels).mean(), self.utility_insulation_extensive), axis=1, keys=['Investment', 'Subsidies', 'Saving', 'Constant'])

                if bool_zil_mean is not None:
                    zil_mean = utility_zil.groupby(levels).mean().rename('ZIL')
                    df = pd.concat((df, zil_mean), axis=1)

                self.utility_yrs.update({self.year: df})

            # retrofit_rate_mean = (retrofit_rate * stock).sum() / stock.sum()
            return retrofit_rate, utility

        def impact_subsidies(scale, utility, stock, pref_subsidies, delta_subsidies, indicator='freeriders'):
            retrofit = 1 / (1 + np.exp(- utility * scale))
            flow = (retrofit * stock).sum()
            retrofit = flow / stock.sum()

            utility_plus = (utility + pref_subsidies * delta_subsidies) * scale
            retrofit_plus = 1 / (1 + np.exp(- utility_plus))
            flow_plus = (retrofit_plus * stock).sum()
            retrofit_plus = flow_plus / stock.sum()

            if indicator == 'elasticity':
                return (retrofit_plus - retrofit) / retrofit

            if indicator == 'freeriders':
                return min(flow, flow_plus) / max(flow, flow_plus)

        @timing
        def calibration_constant_int(utility, ms_insulation, ms_extensive):
            def solve(constant, utility_ref, ms_insulation, flow_retrofit):
                # constant = np.append(0, constant)
                constant = pd.Series(constant, index=utility_ref.columns)
                constant.iloc[0] = 0
                utility = utility_ref + constant
                market_share = (np.exp(utility).T / np.exp(utility).sum(axis=1)).T
                agg = (market_share.T * flow_retrofit).T
                market_share_agg = (agg.sum() / agg.sum().sum()).reindex(ms_insulation.index)

                if (market_share_agg.round(decimals=2) == ms_insulation.round(decimals=2)).all():
                    return np.zeros(ms_insulation.shape[0])
                else:
                    return market_share_agg - ms_insulation

            stock_single = self.stock.xs('Single-family', level='Housing type', drop_level=False)

            probability_replacement = self._probability_replacement
            if isinstance(probability_replacement, pd.Series):
                probability_replacement.index = probability_replacement.index.rename('Heating system')
                probability_replacement = reindex_mi(probability_replacement, stock_single.index)

            flow_retrofit = pd.concat((stock_single * probability_replacement,
                                       stock_single * (1 - probability_replacement)), axis=0, keys=[True, False],
                                      names=['Heater replacement'])
            flow_retrofit = flow_retrofit * reindex_mi(ms_extensive, flow_retrofit.index)

            utility = utility.groupby([i for i in utility.index.names if i != 'Heating system final']).mean()
            utility_ref = reindex_mi(utility, flow_retrofit.index).dropna()
            flow_retrofit = flow_retrofit.reindex(utility_ref.index)

            x0 = np.zeros(ms_insulation.shape[0])
            constant = fsolve(solve, x0, args=(utility_ref, ms_insulation, flow_retrofit))
            # solve(root, utility_ref, ms_insulation, flow_retrofit)
            constant = pd.Series(constant, index=utility_ref.columns)

            utility = utility_ref + constant
            market_share = (np.exp(utility).T / np.exp(utility).sum(axis=1)).T
            agg = (market_share.T * flow_retrofit).T
            market_share_agg = (agg.sum() / agg.sum().sum()).reindex(ms_insulation.index)
            details = pd.concat((constant, market_share_agg, ms_insulation), axis=1,
                                keys=['constant', 'calcul', 'observed']).round(decimals=3)
            details.to_csv(os.path.join(self.path_calibration, 'calibration_constant_insulation.csv'))
            return constant

        @timing
        def calibration_constant_scale_ext(utility, stock, ms_extensive, target_freeriders, delta_subsidies, pref_subsidies):
            """Simultaneously calibrate constant and scale to match freeriders and retrofit rate.

            Parameters
            ----------
            x
            utility
            stock
            ms_extensive
            freeriders
            delta_subsidies
            pref_subsidies

            Returns
            -------

            """

            def solve(x, utility, stock, ms_extensive, freeriders, delta_subsidies, pref_subsidies):
                scale = x[-1]
                constant = x[:-1]

                # calibration constant
                constant = pd.Series(constant, index=ms_extensive.index)
                # utility_ref = utility.groupby([i for i in utility.index.names if i != 'Heating system final']).mean()
                utility_ref = utility.copy()
                # stock_ref = stock.groupby([i for i in stock.index.names if i != 'Heating system final']).mean()
                stock_ref = stock.copy()
                utility_constant = reindex_mi(constant, utility_ref.index)
                u = (utility_ref + utility_constant).copy()
                retrofit_rate = 1 / (1 + np.exp(- u * scale))
                agg = (retrofit_rate * stock_ref).groupby(ms_extensive.index.names).sum()
                retrofit_rate_agg = agg / stock_ref.groupby(ms_extensive.index.names).sum()
                result = retrofit_rate_agg - ms_extensive

                # calibration scale
                calcul = impact_subsidies(scale, u, stock, pref_subsidies, delta_subsidies, indicator='freeriders')
                result = np.append(result, calcul - freeriders)

                return result

            constant = ms_extensive.copy()
            constant[ms_extensive > 0] = 0
            x = np.append(constant.to_numpy(), 1)
            root = fsolve(solve, x, args=(
            utility, stock, ms_extensive, target_freeriders, - delta_subsidies / 1000, pref_subsidies))
            scale = root[-1]
            constant = pd.Series(root[:-1], index=ms_extensive.index) * scale

            utility_constant = reindex_mi(constant, utility.index)
            utility = utility * scale + utility_constant
            retrofit_rate = 1 / (1 + np.exp(- utility))
            agg = (retrofit_rate * stock).groupby(ms_extensive.index.names).sum()
            retrofit_rate_agg = agg / stock.groupby(
                ms_extensive.index.names).sum()

            details = pd.concat((constant, retrofit_rate_agg, ms_extensive, agg / 10 ** 3), axis=1,
                                keys=['constant', 'calcul', 'observed', 'thousand']).round(decimals=3)
            details.to_csv(os.path.join(self.path_calibration, 'calibration_constant_extensive.csv'))

            return constant, root[-1]

        agent = self.new(pd.Series(index=index, dtype=float))
        certificate_before = agent.certificate
        index = certificate_before[certificate_before > 'B'].index
        agent = self.new(pd.Series(index=index, dtype=float))

        surface = agent.surface
        energy_prices = agent.energy_prices(prices)
        energy_bill_sd_before = agent.energy_bill_sd(prices)

        choice_insulation = self._choice_insulation
        consumption_sd = self.prepare_consumption(choice_insulation, index=index)[0]

        energy_bill_sd = (consumption_sd.T * energy_prices * surface).T
        bill_saved = - energy_bill_sd.sub(energy_bill_sd_before, axis=0).dropna()
        self.bill_saved_indiv.update({self.year: bill_saved})

        market_share, utility_intensive = to_market_share(bill_saved, subsidies_total, cost_insulation,
                                                          utility_zil=utility_zil)

        if self.utility_insulation_intensive is None:
            self.utility_insulation_intensive = calibration_constant_int(utility_intensive, ms_insulation, ms_extensive)
            market_share, _ = to_market_share(bill_saved, subsidies_total, cost_insulation,
                                              utility_zil=utility_zil)

        stock_single = self.stock.xs('Single-family', level='Housing type', drop_level=False)
        probability_replacement = self._probability_replacement
        if isinstance(probability_replacement, pd.Series):
            probability_replacement.index = probability_replacement.index.rename('Heating system')
            probability_replacement = reindex_mi(probability_replacement, stock_single.index)
        flow_retrofit = pd.concat((stock_single * probability_replacement,
                                   stock_single * (1 - probability_replacement)), axis=0, keys=[True, False],
                                  names=['Heater replacement'])
        flow_retrofit = flow_retrofit * reindex_mi(ms_extensive, flow_retrofit.index)
        ms = market_share.groupby([i for i in market_share.index.names if i != 'Heating system final']).first()
        ms = reindex_mi(ms, flow_retrofit.index).dropna()
        flow_retrofit = flow_retrofit.reindex(ms.index)
        agg = (ms.T * flow_retrofit).T
        market_share_agg = (agg.sum() / agg.sum().sum()).reindex(ms_insulation.index)
        self.market_share_yrs.update({self.year: market_share_agg})

        # extensive margin

        bill_saved_insulation = (bill_saved.reindex(market_share.index) * market_share).sum(axis=1)
        self.bill_saved_repr.update({self.year: bill_saved_insulation})
        subsidies_insulation = (subsidies_total.reindex(market_share.index) * market_share).sum(axis=1)
        self.subsidies_insulation_repr.update({self.year: subsidies_insulation})
        investment_insulation = (cost_insulation.reindex(market_share.index) * market_share).sum(axis=1)
        self.investment_insulation_repr.update({self.year: investment_insulation})

        bool_zil_mean = None
        if utility_zil is not None:
            bool_zil = utility_zil.copy()
            bool_zil[bool_zil > 0] = 1
            bool_zil_mean = (bool_zil.reindex(market_share.index) * market_share).sum(axis=1)

        retrofit_rate, utility = to_retrofit_rate(bill_saved_insulation, subsidies_insulation, investment_insulation,
                                                  bool_zil_mean=bool_zil_mean)

        if self.utility_insulation_extensive is None:

            delta_subsidies = (delta_subsidies.reindex(market_share.index) * market_share).sum(axis=1)
            pref_subsidies = reindex_mi(self.pref_subsidy_insulation_ext, subsidies_insulation.index).rename(None)
            if detailed:
                x, free_riders, elasticity = [], [], []
                for scale in np.arange(0.1, 5, 0.1):
                    x.append(scale)
                    free_riders.append(impact_subsidies(scale, utility, stock, pref_subsidies, - delta_subsidies / 1000,
                                                        indicator='freeriders'))
                    elasticity.append(impact_subsidies(scale, utility, stock, pref_subsidies,
                                                       subsidies_insulation / 1000 * 0.01,
                                                       indicator='elasticity'))

                graphs = {'Freeriders cite': free_riders, 'Elasticity': elasticity}
                for name, data in graphs.items():
                    df = pd.Series(data, index=pd.Index(x, name='Scale'), name=name)
                    fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))
                    df.plot(ax=ax)
                    format_ax(ax, format_y=lambda y, _: '{:.0%}'.format(y), y_label=name)
                    save_fig(fig, save=os.path.join(self.path_calibration, 'scale_calibration_{}.png'.format(name.lower())))

            """constant = ms_extensive.copy()
            constant[ms_extensive > 0] = 0
            x = np.append(constant.to_numpy(), 1)
            root = fsolve(calibration_constant_scale_ext, x, args=(
            utility, stock, ms_extensive, target_freeriders, - delta_subsidies / 1000, pref_subsidies))
            self.utility_insulation_extensive = pd.Series(root[:-1], index=ms_extensive.index) * root[-1]"""

            constant, scale = calibration_constant_scale_ext(utility, stock, ms_extensive, target_freeriders,
                                                             delta_subsidies, pref_subsidies)
            self.utility_insulation_extensive = constant

            if detailed:
                stock_single = stock.xs('Single-family', level='Housing type', drop_level=False)
                stock_multi = stock.xs('Multi-family', level='Housing type', drop_level=False)
                x_before, y_before, y_before_single, y_before_multi = [], [], [], []
                for delta in np.arange(0, 2, 0.1):
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
            self.scale_ext = self.scale_ext

            if detailed:
                x_after, y_after, y_after_single, y_after_multi = [], [], [], []
                for delta in np.arange(0, 2, 0.1):
                    sub = subsidies_insulation * (1 + delta)
                    x_after.append((sub * stock).sum() / stock.sum())
                    rate = to_retrofit_rate(bill_saved_insulation, sub, investment_insulation)[0]
                    y_after.append((rate * stock).sum() / stock.sum())
                    y_after_single.append((rate * stock_single).sum() / stock_single.sum())
                    y_after_multi.append((rate * stock_multi).sum() / stock_multi.sum())

                df = pd.concat(
                    (pd.Series(x_before), pd.Series(y_before), pd.Series(y_after), pd.Series(y_before_single),
                     pd.Series(y_after_single), pd.Series(y_before_multi),
                     pd.Series(y_after_multi)), axis=1)

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

            retrofit_rate, utility = to_retrofit_rate(bill_saved_insulation, subsidies_insulation, investment_insulation,
                                                      bool_zil_mean=bool_zil_mean)

            if detailed:
                scale = pd.Series(self.scale_ext, index=['Scale'])
                constant_ext = self.utility_insulation_extensive.copy()
                constant_ext.index = constant_ext.index.to_flat_index()
                constant_int = self.utility_insulation_intensive.copy()
                constant_int.index = constant_int.index.to_flat_index()
                flow_retrofit = retrofit_rate * stock
                retrofit_rate_mean = flow_retrofit.sum() / stock.sum()
                retrofit_rate_mean = pd.Series(retrofit_rate_mean, index=['Retrofit rate mean (%)'])
                retrofit_calibrated = flow_retrofit.groupby(ms_extensive.index.names).sum() / stock.groupby(ms_extensive.index.names).sum()
                retrofit_calibrated.index = retrofit_calibrated.index.to_flat_index()
                flow_insulation = (flow_retrofit * market_share.T).T.sum()
                flow_insulation_agg, name = list(), ''
                for i in flow_insulation.index.names:
                    flow_insulation_agg.append(flow_insulation.xs(True, level=i).sum())
                    name = '{}{},'.format(name, i)
                name = pd.Series('', index=[name])
                flow_insulation_agg = pd.Series(flow_insulation_agg, index=flow_insulation.index.names)
                flow_insulation.index = flow_insulation.index.to_flat_index()
                flow_insulation_sum = pd.Series(flow_insulation.sum(), index=['Replacement insulation'])
                ms_calibrated = flow_insulation / flow_insulation.sum()
                ms_calibrated.index = ms_calibrated.index.to_flat_index()
                result = pd.concat((scale, constant_ext, retrofit_rate_mean, retrofit_calibrated, flow_insulation_sum,
                                    flow_insulation_agg, name, constant_int, flow_insulation, ms_calibrated), axis=0)
                result.to_csv(os.path.join(self.path_calibration, 'result_calibration.csv'))

        if supply_constraint is True:
            market_size = (retrofit_rate * stock * investment_insulation).sum()
            etp_size = self.factor_etp * market_size

            if self.param_supply is None:

                self.capacity_utilization = etp_size / 0.8
                self.param_supply = dict()
                self.param_supply['a'], self.param_supply['b'], self.param_supply['c'] = self.calibration_supply()

                x = np.arange(0, 1.05, 0.05)
                y = self.factor_function(self.param_supply['a'], self.param_supply['b'], self.param_supply['c'], x)
                fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))
                ax.plot(x, y, color='black')
                format_ax(ax, y_label='Cost factor')
                ax.set_xlabel('Utilization rate (%)')
                save_fig(fig, save=os.path.join(self.path_calibration, 'marginal_cost_curve.png'))

                x, y_supply, y_demand = [], [], []
                for factor in np.arange(0.81, 1.5, 0.05):
                    retrofit_rate = to_retrofit_rate(bill_saved_insulation, subsidies_insulation,
                                                     investment_insulation * factor)[0]
                    y_demand.append((retrofit_rate * stock * investment_insulation * self.factor_etp).sum())
                    utilization_rate = self.supply_function(self.param_supply['a'], self.param_supply['b'],
                                                            self.param_supply['c'], factor)
                    y_supply.append(utilization_rate * self.capacity_utilization)
                    x.append(factor)

                df = pd.concat((pd.Series(x), pd.Series(y_supply)/10**3, pd.Series(y_demand)/10**3,),
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

            factor_equilibrium = fsolve(solve_equilibrium, np.array([1]), args=(
            bill_saved_insulation, subsidies_insulation, investment_insulation, stock))

            retrofit_rate = to_retrofit_rate(bill_saved_insulation, subsidies_insulation,
                                             investment_insulation * factor_equilibrium)[0]

            self.factor_yrs.update({self.year: factor_equilibrium})

        return retrofit_rate, market_share

    @staticmethod
    def factor_function(a, b, c, utilization_rate):
        """Represents an increasing cost (or decreasing returns) function of the productive capacities utilisation rate.

        Based on Imaclim-R (Waisman et al., 2012)

        Parameters
        ----------
        a: float
        b: float
        c: float
        utilization_rate: float or np.array

        Returns
        -------
        float or np.array
        """
        return a - b * np.tanh(c * (1 - utilization_rate))

    @staticmethod
    def supply_function(a, b, c, factor):
        """

        Parameters
        ----------
        a
        b
        c
        factor

        Returns
        -------
        utilization_rate
        """
        return 1 - np.arctanh((a - factor) / b) / c

    @staticmethod
    def calibration_supply():
        """

        Returns
        -------
        np.array
        """
        def factor_calibration(x, factor_max=30, factor_min=0.8, factor_norm=1,
                               utilization_norm=0.8):
            a, b, c = x
            y = [factor_max - AgentBuildings.factor_function(a, b, c, 1),
                 factor_min - AgentBuildings.factor_function(a, b, c, 0),
                 factor_norm - AgentBuildings.factor_function(a, b, c, utilization_norm)
                 ]
            return y

        root = fsolve(factor_calibration, np.ones(3))
        return root

    @staticmethod
    @timing
    def exogenous_retrofit(index, choice_insulation):
        """Format retrofit rate and market share for each segment.

        Global retrofit and retrofit rate to match exogenous numbers.
        Retrofit all heating system replacement dwelling.

        Parameters
        ----------
        index: pd.MultiIndex
        choice_insulation: pd.MultiIndex

        Returns
        -------
        pd.Series
            Retrofit rate by segment.
        pd.DataFrame
            Market-share by segment and insulation choice.
        """

        retrofit_rate = pd.concat([pd.Series(1, dtype=float, index=index)], keys=[True], names=['Heater replacement'])

        market_share = pd.DataFrame(0, index=index, columns=choice_insulation)
        market_share.loc[:, (True, True, True, True)] = 1

        return retrofit_rate, market_share

    @timing
    def insulation_replacement(self, prices, cost_insulation_raw, ms_insulation, ms_extensive, policies_insulation,
                               target_freeriders, index=None, stock=None, supply_constraint=False):
        """Calculate insulation retrofit in the dwelling stock.

        1. Intensive margin
        2. Extensive margin
        Calibrate function first year.

        To reduce calculation time attributes are grouped.
        Cost, subsidies and constant depends on Housing type, Occupancy status, Housing type and Insulation performance.

        Parameters
        ----------
        prices: pd.Series
        cost_insulation_raw: pd.Series
            €/m2 of losses area by component.
        ms_insulation: pd.Series
        ms_extensive: pd.Series
        policies_insulation: list
        index: pd.MultiIndex or pd.Index, default None
        stock: pd.Series, default None

        Returns
        -------
        pd.Series
            Retrofit rate
        pd.DataFrame
            Market-share insulation
        """
        if index is None:
            index = self.index

        agent = self.new(pd.Series(index=index, dtype=float))
        certificate_before = agent.certificate
        index = certificate_before[certificate_before > 'B'].index

        agent = self.new(pd.Series(index=index, dtype=float))
        certificate_before = agent.certificate
        surface = agent.surface

        choice_insulation = self._choice_insulation
        certificate = self.prepare_consumption(choice_insulation, index=index, levels_heater='Heating system final')[1]

        cost_insulation = self.prepare_cost_insulation(cost_insulation_raw * self.surface_insulation)
        cost_insulation = surface.rename(None).to_frame().dot(cost_insulation.to_frame().T)
        cost_insulation = cost_insulation.reindex(certificate.index)

        cost_insulation, tax_insulation, tax, subsidies_details, subsidies_total, certificate_jump = self.apply_subsidies_insulation(
            policies_insulation,
            cost_insulation, surface,
            certificate,
            certificate_before)

        if self._endogenous:

            utility_subsidies = subsidies_total.copy()
            for sub in ['reduced_tax', 'zero_interest_loan']:
                if sub in subsidies_details.keys():
                    utility_subsidies -= subsidies_details[sub]

            utility_zil = None
            if 'zero_interest_loan' in subsidies_details:
                utility_zil = subsidies_details['zero_interest_loan'].copy()

            delta_subsidies = None
            if self.year == self.first_year + 1:
                delta_subsidies = subsidies_details['cite']

            retrofit_rate, market_share = self.endogenous_retrofit(index, prices, utility_subsidies, cost_insulation,
                                                                   ms_insulation=ms_insulation,
                                                                   ms_extensive=ms_extensive,
                                                                   utility_zil=utility_zil,
                                                                   stock=stock,
                                                                   supply_constraint=supply_constraint,
                                                                   delta_subsidies=delta_subsidies,
                                                                   target_freeriders=target_freeriders)

        else:
            retrofit_rate, market_share = self.exogenous_retrofit(index, choice_insulation)

        self.store_information_insulation(certificate, certificate_jump, cost_insulation_raw, tax, cost_insulation,
                                          tax_insulation, subsidies_details, subsidies_total, retrofit_rate)

        return retrofit_rate, market_share

    @timing
    def apply_subsidies_insulation(self, policies_insulation, cost_insulation, surface, certificate, certificate_before):
        """Calculate subsidies amount for each possible insulation choice.

        Parameters
        ----------
        policies_insulation: list
        cost_insulation: pd.DataFrame
            Cost for each segment and each possible insulation choice (€).
        surface: pd.Series
            Surface / dwelling for each segment (m2/dwelling).
        certificate: : pd.Series
            Certificate by segment after insulation replacement for each possible insulation choice.
        certificate_before: : pd.Series
            Certificate by segment before.

        Returns
        -------
        pd.DataFrame
        pd.DataFrame
        float
        dict
        pd.DataFrame
        pd.DataFrame
        """

        frame = certificate.copy()
        subsidies_total = pd.DataFrame(0, index=frame.index, columns=frame.columns)
        subsidies_details = {}

        tax = self.vta
        p = [p for p in policies_insulation if 'reduced_tax' == p.policy]
        if p:
            tax = p[0].value
            subsidies_details.update({p[0].name: cost_insulation * (self.vta - tax)})
            subsidies_total += subsidies_details['reduced_tax']

        tax_insulation = cost_insulation * tax
        cost_insulation += tax_insulation

        out_worst = ~certificate.isin(['G', 'F']).astype(int).mul(certificate_before.isin(['G', 'F']).astype(int),
                                                                  axis=0).astype(bool)
        in_best = certificate.isin(['A', 'B']).astype(int).mul(~certificate_before.isin(['A', 'B']).astype(int),
                                                               axis=0).astype(bool)

        target_0 = certificate.isin(['D', 'C', 'B', 'A']).astype(int).mul(
            certificate_before.isin(['G', 'F', 'E']).astype(int), axis=0).astype(bool)
        target_1 = certificate.isin(['B', 'A']).astype(int).mul(certificate_before.isin(['D', 'C']).astype(int),
                                                                axis=0).astype(bool)
        target_subsidies = target_0 | target_1

        # certificate_matrix = certificate.set_axis(certificate_before.reindex(certificate.index).values, axis=0)

        epc2int = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
        certificate_jump = - certificate.replace(epc2int).sub(certificate_before.replace(epc2int), axis=0)
        global_retrofit = certificate_jump >= 2

        for policy in policies_insulation:
            if policy.policy == 'subsidy_target':

                temp = (reindex_mi(self.prepare_subsidy_insulation(policy.value),
                                   frame.index).T * surface).T
                subsidies_total += temp

                if policy.name in subsidies_details.keys():
                    subsidies_details[policy.name] = subsidies_details[policy.name] + temp

                else:
                    subsidies_details[policy.name] = temp.copy()

            elif policy.policy == 'bonus_best':
                temp = (reindex_mi(policy.value, in_best.index) * in_best.T).T
                subsidies_total += temp
                if policy.name in subsidies_details.keys():
                    subsidies_details[policy.name] = subsidies_details[policy.name] + temp

                else:
                    subsidies_details[policy.name] = temp.copy()

            elif policy.policy == 'bonus_worst':
                temp = (reindex_mi(policy.value, out_worst.index) * out_worst.T).T
                subsidies_total += temp

                if policy.name in subsidies_details.keys():
                    subsidies_details[policy.name] = subsidies_details[policy.name] + temp

                else:
                    subsidies_details[policy.name] = temp.copy()

            elif policy.policy == 'subsidy_ad_volarem':

                if isinstance(policy.value, pd.Series):
                    temp = reindex_mi(policy.value, cost_insulation.index)
                    subsidies_details[policy.name] = (temp * cost_insulation.T).T
                    subsidies_total += subsidies_details[policy.name]
                else:
                    subsidies_details[policy.name] = policy.value * cost_insulation
                    subsidies_total += subsidies_details[policy.name]

            elif policy.policy == 'zero_interest_loan':
                cost = cost_insulation.copy()
                if policy.cost_max is not None:
                    cost_max = reindex_mi(policy.cost_max, cost.index)
                    cost_max = pd.concat([cost_max] * cost.shape[1], axis=1).set_axis(
                        cost.columns, axis=1)
                    cost[cost > cost_max] = cost_max
                if policy.cost_min is not None:
                    cost_min = reindex_mi(policy.cost_min, cost.index)
                    cost_min = pd.concat([cost_min] * cost.shape[1], axis=1).set_axis(
                        cost.columns, axis=1)
                    cost[cost < cost_min] = 0
                if policy.target is not None:
                    cost = cost[target_subsidies].fillna(0)

                subsidies_details[policy.name] = policy.value * cost
                subsidies_total += subsidies_details[policy.name]

        subsidies_cap = [p for p in policies_insulation if p.policy == 'subsidies_cap']

        subsidies_uncaped = subsidies_total.copy()
        for sub in ['reduced_tax', 'zero_interest_loan']:
            if sub in subsidies_details.keys():
                subsidies_uncaped -= subsidies_details[sub]

        if subsidies_cap:
            subsidies_cap = subsidies_cap[0]
            subsidies_cap = reindex_mi(subsidies_cap.value, subsidies_uncaped.index)
            cap = (cost_insulation.T * subsidies_cap).T
            over_cap = subsidies_uncaped > cap
            subsidies_details['over_cap'] = (subsidies_uncaped - cap)[over_cap].fillna(0)
            if 'mpr' in subsidies_details.keys():
                subsidies_details['mpr'] -= subsidies_details['over_cap']
                assert (subsidies_details['mpr'].values >= 0).all(), 'MPR got negative values'
            subsidies_total -= subsidies_details['over_cap']

        return cost_insulation, tax_insulation, tax, subsidies_details, subsidies_total, certificate_jump

    @timing
    def store_information_insulation(self, certificate, certificate_jump, cost_insulation_raw, tax, cost_insulation,
                                     tax_insulation, subsidies_details, subsidies_total, retrofit_rate):
        """Store insulation information.

        Parameters
        ----------
        certificate: pd.DataFrame
            Energy performance certificate after retrofitting insulation.
        certificate_jump: pd.DataFrame
            Number of epc jump.
        cost_insulation_raw: pd.Series
            Cost of insulation for each envelope component of losses surface (€/m2).
        tax: float
            VTA to apply (%).
        cost_insulation: pd.DataFrame
            Cost insulation for each dwelling and each insulation gesture (€).
        tax_insulation: pd.DataFrame
            VTA applied to each insulation gesture cost (€).
        subsidies_details: dict
            Amount of subsidies for each dwelling and each insulation gesture (€).
        subsidies_total: pd.DataFrame
            Total mount of subsidies for each dwelling and each insulation gesture (€).
        retrofit_rate: pd.Series
        """
        self.efficient_renovation = certificate.isin(['A', 'B'])
        self.certificate_jump = certificate_jump
        self.cost_component.update({self.year: cost_insulation_raw * self.surface_insulation * (1 + tax)})

        self.subsidies_details_insulation.update({self.year: subsidies_details})
        self.subsidies_insulation_indiv.update({self.year: subsidies_total})
        self.cost_insulation_indiv.update({self.year: cost_insulation})
        self.tax_insulation.update({self.year: tax_insulation})
        self.retrofit_rate.update({self.year: retrofit_rate})

    @timing
    def calibration_constant_intensive(self, utility, ms_insulation, ms_extensive):
        """Calibrate alternative-specific constant to match observed market-share.

        Parameters
        ----------
        utility: pd.Series
        ms_insulation: pd.Series
            Observed market-share.
        ms_extensive: pd.Series
            Observed renovation rate.
        Returns
        -------
        pd.Series
        """

        stock_single = self.stock.xs('Single-family', level='Housing type', drop_level=False)
        flow_retrofit = pd.concat((stock_single * self._probability_replacement,
                                   stock_single * (1 - self._probability_replacement)), axis=0, keys=[True, False],
                                  names=['Heater replacement'])
        flow_retrofit = flow_retrofit * reindex_mi(ms_extensive, flow_retrofit.index)

        utility = utility.groupby([i for i in utility.index.names if i != 'Heating system final']).mean()
        utility_ref = reindex_mi(utility, flow_retrofit.index)

        constant = ms_insulation.reindex(utility_ref.columns, axis=0).copy()
        constant[constant > 0] = 0
        market_share_ini, market_share_agg = None, None
        for i in range(150):
            utility = (utility_ref + constant).copy()
            constant.iloc[0] = 0
            market_share = (np.exp(utility).T / np.exp(utility).sum(axis=1)).T
            agg = (market_share.T * flow_retrofit).T
            market_share_agg = (agg.sum() / agg.sum().sum()).reindex(ms_insulation.index)
            if i == 0:
                market_share_ini = market_share_agg.copy()
            constant = constant + np.log(ms_insulation / market_share_agg)

            if (market_share_agg.round(decimals=2) == ms_insulation.round(decimals=2)).all():
                print('Constant intensive optim worked')
                break

        constant.iloc[0] = 0
        details = pd.concat((constant, market_share_ini, market_share_agg, ms_insulation), axis=1,
                            keys=['constant', 'calcul ini', 'calcul', 'observed']).round(decimals=3)
        details.to_csv(os.path.join(self.path_calibration, 'calibration_constant_insulation.csv'))

        return constant

    @timing
    def calibration_constant_extensive(self, utility, ms_extensive):
        """Calibrate alternative-specific constant to match observed market-share.

        Parameters
        ----------
        utility: pd.Series
        ms_extensive: pd.Series
            Observed market-share.

        Returns
        -------
        pd.Series
        """

        utility_ref = utility.groupby([i for i in utility.index.names if i != 'Heating system final']).mean()

        stock = pd.concat([self.stock, self.stock], keys=[True, False], names=['Heater replacement']).groupby(
            utility_ref.index.names).sum().reindex(utility_ref.index)
        stock = stock.groupby(utility_ref.index.names).sum()

        constant = ms_extensive.copy()
        constant[ms_extensive > 0] = 0
        retrofit_rate_ini, retrofit_rate_agg, agg = None, None, None
        for i in range(100):
            utility_constant = reindex_mi(constant, utility_ref.index)
            utility = (utility_ref + utility_constant).copy()

            retrofit_rate = 1 / (1 + np.exp(- utility))
            agg = (retrofit_rate * stock).groupby(ms_extensive.index.names).sum()
            retrofit_rate_agg = agg / stock.groupby(
                ms_extensive.index.names).sum()
            if i == 0:
                retrofit_rate_ini = retrofit_rate_agg.copy()
            constant = constant + np.log(ms_extensive / retrofit_rate_agg)

            if (retrofit_rate_agg.round(decimals=3) == ms_extensive.round(decimals=3).reindex(
                    retrofit_rate_agg.index)).all():
                print('Constant extensive optim worked')
                break

        details = pd.concat((constant, retrofit_rate_ini, retrofit_rate_agg, ms_extensive, agg / 10**3), axis=1,
                            keys=['constant', 'calcul ini', 'calcul', 'observed', 'thousand']).round(decimals=3)
        details.to_csv(os.path.join(self.path_calibration, 'calibration_constant_extensive.csv'))

        return constant

    @timing
    def flow_retrofit(self, prices, cost_heater, ms_heater, cost_insulation, ms_insulation, ms_extensive,
                      policies_heater, policies_insulation, target_freeriders, supply_constraint=False):
        """Compute heater replacement and insulation retrofit.

        1. Heater replacement based on current stock segment.
        2. Knowing heater replacement (and new heating system) calculating retrofit rate by segment and market
        share by segment.
        3. Then, managing inflow and outflow.

        Parameters
        ----------
        prices: pd.Series
        cost_heater: pd.Series
        ms_heater
        cost_insulation
        ms_insulation
        ms_extensive
        policies_heater
        policies_insulation
        supply_constraint: bool

        Returns
        -------
        pd.Series
        """
        stock = self.heater_replacement(prices, cost_heater, ms_heater, policies_heater)

        print('Index: {}'.format(stock.shape[0]))
        retrofit_rate, market_share = self.insulation_replacement(prices, cost_insulation, ms_insulation,
                                                                  ms_extensive, policies_insulation,
                                                                  target_freeriders, index=stock.index, stock=stock,
                                                                  supply_constraint=supply_constraint)

        retrofit_rate = reindex_mi(retrofit_rate, stock.index)
        retrofit_stock = (retrofit_rate * stock).dropna()
        replacement_sum = retrofit_stock.sum().sum()
        market_share = reindex_mi(market_share, retrofit_stock.index)
        replaced_by = (retrofit_stock * market_share.T).T.copy()
        assert round(replaced_by.sum().sum(), 0) == round(replacement_sum, 0), 'Sum problem'
        self.store_information_retrofit(replaced_by, market_share)

        replaced_by = replaced_by.groupby(
            [c for c in replaced_by.index.names if c != 'Heater replacement']).sum()

        share = (self.stock_mobile.unstack('Income tenant').T / self.stock_mobile.unstack('Income tenant').sum(axis=1)).T
        temp = pd.concat([replaced_by] * share.shape[1], keys=share.columns, names=share.columns.names, axis=1)
        share = reindex_mi(share, temp.columns, axis=1)
        share = reindex_mi(share, temp.index)
        replaced_by = (share * temp).stack('Income tenant').dropna()
        assert round(replaced_by.sum().sum(), 0) == round(replacement_sum, 0), 'Sum problem'

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
        replaced_by = replaced_by.reset_index()
        for component in ['Wall', 'Floor', 'Roof', 'Windows']:
            replaced_by[component] = replaced_by['{} before'.format(component)]
            replaced_by.loc[replaced_by['{} after'.format(component)], component] = self._performance_insulation[component]

        replaced_by.drop(
            ['Wall before', 'Wall after', 'Roof before', 'Roof after', 'Floor before', 'Floor after', 'Windows before',
             'Windows after'], axis=1, inplace=True)
        replaced_by = replaced_by.set_index(self.index.names).loc[:, 'Data']

        to_replace = to_replace.reorder_levels(replaced_by.index.names)
        flow_retrofit = pd.concat((-to_replace, replaced_by), axis=0)
        flow_retrofit = flow_retrofit.groupby(flow_retrofit.index.names).sum()
        assert round(flow_retrofit.sum(), 0) == 0, 'Sum problem'

        return flow_retrofit

    @timing
    def store_information_retrofit(self, replaced_by, market_share):
        """Calculate and store main outputs based on yearly retrofit.

        Parameters
        ----------
        replaced_by: pd.DataFrame
            Retrofitting for each dwelling and each insulation gesture.
        """

        def npv(investment, subsidies, bill_saved):
            discount_factor = reindex_mi(self.discount_factor, investment.index)
            return - investment + subsidies + discount_factor * bill_saved

        def tri(investment, subsidies, bill_saved):
            discount_factor = (investment - subsidies) / bill_saved
            pass

        def payback(investment, subsidies, bill_saved):
            pass

        levels = [i for i in replaced_by.index.names if i not in ['Heater replacement', 'Heating system final']]
        self.efficient_renovation_yrs.update({self.year: (replaced_by * self.efficient_renovation).sum().sum()})

        rslt = {}
        for i in range(6):
            rslt.update({i: ((self.certificate_jump == i) * replaced_by).sum(axis=1)})
        rslt = pd.DataFrame(rslt).groupby(levels).sum()
        self.certificate_jump_yrs.update({self.year: rslt})

        self.replacement_insulation.update({self.year: replaced_by.groupby(levels).sum()})

        self.investment_insulation.update(
            {self.year: (replaced_by * self.cost_insulation_indiv[self.year]).groupby(levels).sum()})
        self.investment_insulation_mean.update(
            {self.year: (market_share * self.cost_insulation_indiv[self.year]).sum(axis=1).groupby(levels).first()})
        self.investment_insulation_efficient.update(
            {self.year: (self.efficient_renovation * self.cost_insulation_indiv[self.year]).mean(axis=1).replace(0, float('nan')).groupby(levels).first()})

        self.taxed_insulation.update(
            {self.year: (replaced_by * self.tax_insulation[self.year]).groupby(levels).sum()})
        # replaced_by / replaced_by.copy()
        self.subsidies_insulation.update(
            {self.year: (replaced_by * self.subsidies_insulation_indiv[self.year]).groupby(levels).sum()})
        self.subsidies_insulation_mean.update(
            {self.year: (market_share * self.subsidies_insulation_indiv[self.year]).sum(axis=1).groupby(levels).first()})
        self.subsidies_insulation_efficient.update(
            {self.year: (self.efficient_renovation * self.subsidies_insulation_indiv[self.year]).mean(axis=1).groupby(levels).first()})

        if self.year in self.bill_saved_indiv.keys():
            self.bill_saved_mean.update(
                {self.year: (market_share * self.bill_saved_indiv[self.year]).sum(axis=1).groupby(
                    levels).first()})
            self.bill_saved_efficient.update(
                {self.year: (self.efficient_renovation * self.bill_saved_indiv[self.year]).mean(
                    axis=1).groupby(levels).first()})

            private_npv_mean = npv(self.investment_insulation_mean[self.year],
                                   self.subsidies_insulation_mean[self.year], self.bill_saved_mean[self.year])
            self.private_npv_mean.update({self.year: private_npv_mean})
            private_npv_efficient = npv(self.investment_insulation_efficient[self.year],
                                        self.subsidies_insulation_efficient[self.year],
                                        self.bill_saved_efficient[self.year])
            self.private_npv_efficient.update({self.year: private_npv_efficient})

        for key in self.subsidies_details_insulation[self.year].keys():
            self.subsidies_details_insulation[self.year][key] = (replaced_by * reindex_mi(
                self.subsidies_details_insulation[self.year][key], replaced_by.index)).groupby(levels).sum()

    def flow_demolition(self):
        """Demolition of E, F and G buildings based on their share in the mobile stock.

        Returns
        -------
        pd.Series
        """
        stock_demolition = self.stock_mobile[self.certificate.isin(self._target_demolition)]
        if stock_demolition.sum() < self._demolition_total:
            self._target_demolition = ['G', 'F', 'E', 'D']
            stock_demolition = self.stock_mobile[self.certificate.isin(self._target_demolition)]
            if stock_demolition.sum() < self._demolition_total:
                self._target_demolition = ['G', 'F', 'E', 'D', 'C']
                stock_demolition = self.stock_mobile[self.certificate.isin(self._target_demolition)]

        stock_demolition = stock_demolition / stock_demolition.sum()
        flow_demolition = (stock_demolition * self._demolition_total).dropna()
        return flow_demolition.reorder_levels(self.index.names)
