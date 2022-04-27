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
from utils import reindex_mi
import thermal
from time import time


class ThermalBuildings:
    def __init__(self, stock, surface, param, efficiency, income, consumption_ini, path, year=2018, data_calibration=None):

        system2energy = {'Wood boiler': 'Wood fuel', 'Water/air heat pump': 'Electricity',
                         'Air/air heat pump': 'Electricity', 'Direct electric': 'Electricity',
                         'Gas boiler': 'Natural gas', 'Oil boiler': 'Oil fuel'}
        self._system2energy = system2energy
        self._efficiency = efficiency
        self._param = param
        self._dh = 55706
        self._path = path

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
        self.stock_yrs, self.surface_yrs, self.heat_consumption_sd_yrs, self.heat_consumption_yrs = {}, {}, {}, {}
        self.budget_share_yrs, self.heating_intensity_yrs = {}, {}
        self.heating_intensity_avg, self.energy_poverty = {}, {}
        self.heating_intensity_tenant = {}
        self.heat_consumption_yrs = {}

        self.heat_consumption_energy_yrs, self.energy_expenditure, self.taxes_expenditure = {}, {}, {}
        self.taxes_expenditure_details = {}

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
        self.stock_mobile = stock - self._stock_residual.reindex(stock.index, fill_value=0)
        self.levels = list(stock.index.names)
        self.index = stock.index

        self.surface = reindex_mi(self._surface, stock.index)
        self.stock_surface = self.stock * self.surface
        self.income_owner = reindex_mi(self._income_owner, stock.index)
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

        self.heat_consumption_sd = self.heating_consumption_sd()
        self.certificate = self.certificates()

        self.stock_yrs.update({self.year: self.stock})
        self.surface_yrs.update({self.year: self.stock * self.surface})
        self.heat_consumption_sd_yrs.update({self.year: self.stock * self.surface * self.heat_consumption_sd})
        self.certificate_nb.update({self.year: self.stock.groupby(self.certificate).sum()})

    def prepare(self, wall=None, floor=None, roof=None, windows=None, efficiency=None, energy=None):
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
        wall, floor, roof, windows, efficiency, _ = self.prepare(wall=wall, floor=floor, roof=roof, windows=windows,
                                                                 efficiency=efficiency)
        return thermal.heating_consumption(wall, floor, roof, windows, self._dh, efficiency, self._param)

    def primary_heating_consumption_sd(self, wall=None, floor=None, roof=None, windows=None, efficiency=None,
                                       energy=None):
        wall, floor, roof, windows, efficiency, energy = self.prepare(wall=wall, floor=floor, roof=roof,
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
                ['Housing type', 'Energy']).sum().iloc[:, 0] / 10 ** 9

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

            validation.round(2).to_csv(os.path.join(self._path, 'validation_stock.csv'))

        heat_consumption_energy = heat_consumption_energy * self.coefficient_consumption
        self.heat_consumption_energy_yrs.update({self.year: heat_consumption_energy})
        self.energy_expenditure.update({self.year: prices * heat_consumption_energy})

        total_taxes = pd.Series(0, index=prices.index)
        for tax in taxes:
            if tax.name not in self.taxes_expenditure_details.keys():
                self.taxes_expenditure_details[tax.name] = {}
            amount = tax.value.loc[self.year, :] * heat_consumption_energy
            self.taxes_expenditure_details[tax.name].update({self.year: amount})
            total_taxes += amount

        self.taxes_expenditure.update({self.year: total_taxes})

    def certificates(self, wall=None, floor=None, roof=None, windows=None, efficiency=None, energy=None):
        wall, floor, roof, windows, efficiency, energy = self.prepare(wall=wall, floor=floor, roof=roof,
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

    def __init__(self, stock, surface, param, efficiency, income, consumption_ini, path, preferences, restrict_heater,
                 ms_heater, choice_insulation, performance_insulation, demolition_rate=0.0, year=2018,
                 data_calibration=None):
        super().__init__(stock, surface, param, efficiency, income, consumption_ini, path, year=year,
                         data_calibration=data_calibration)

        self.vta = 0.2

        self.pref_investment = preferences['investment']
        self.pref_subsidy = preferences['subsidy']
        self.pref_bill = preferences['bill_saved']
        self.pref_inertia = preferences['inertia']

        self._demolition_rate = demolition_rate
        self._demolition_total = (stock * self._demolition_rate).sum()

        self._choice_heater = list(ms_heater.columns)
        self._restrict_heater = restrict_heater
        self._ms_heater = ms_heater

        self._choice_insulation = choice_insulation
        self._performance_insulation = performance_insulation

        # TODO: clean assign by housing type (mean is done)
        self.surface_insulation = pd.Series({'Wall': param['ratio_surface']['Wall'].mean(),
                                             'Floor': param['ratio_surface']['Floor'].mean(),
                                             'Roof': param['ratio_surface']['Roof'].mean(),
                                             'Windows': param['ratio_surface']['Windows'].mean()})

        # self.surface_insulation = param['ratio_surface']

        self.utility_insulation_extensive, self.utility_insulation_intensive, self.constant_heater = None, None, None
        self.heater_replaced = {}

        self.certificate_jump, self.efficient_renovation = None, None
        self.efficient_renovation_yrs, self.certificate_jump_yrs = {}, {}

        self.subsidies_details_heater, self.subsidies_total_heater = {}, {}
        self.subsidies_details_insulation, self.subsidies_total_insulation = {}, {}

        self.replacement_heater, self.investment_heater, self.subsidies_heater = {}, {}, {}
        self.cost_insulation, self.tax_insulation, self.tax_heater = {}, {}, {}
        self.cost_component, self.cost_heater = {}, {}
        self.replacement_insulation, self.investment_insulation, self.subsidies_insulation = {}, {}, {}

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
        self.stock = flow_total.reindex(union, fill_value=0) + self.stock.reindex(union, fill_value=0)

    def heater_replacement(self, prices, cost_heater, ms_heater, policies_heater):
        """Function returns new building stock after heater replacement.

        Parameters
        ----------
        prices: pd.Series
        cost_heater: pd.Series
        ms_heater: pd.DataFrame
        policies_heater: list

        Returns
        -------
        pd.Series
        """

        choice_heater = self._choice_heater
        self.cost_heater.update({self.year: cost_heater})
        choice_heater_idx = pd.Index(choice_heater, name='Heating system final')

        efficiency = pd.to_numeric(pd.Series(choice_heater).str.split('-').str[1].replace(self._efficiency))
        efficiency.index = choice_heater_idx

        energy = pd.Series(choice_heater).str.split('-').str[0]
        energy.index = choice_heater_idx

        heat_consumption_sd = self.heating_consumption_sd(efficiency=efficiency)
        levels = [i for i in heat_consumption_sd.index.names if i != 'Income tenant']
        heat_consumption_sd = heat_consumption_sd.groupby(levels).first()

        prices_re = prices.reindex(energy).copy()
        prices_re.index = heat_consumption_sd.columns

        energy_bill_sd = ((heat_consumption_sd * prices_re).T * self.surface.groupby(levels).first()).T
        bill_saved = - energy_bill_sd.sub(self.energy_bill_sd(prices).groupby(levels).first(), axis=0)

        utility_bill_saving = (bill_saved.T * reindex_mi(self.pref_bill, bill_saved.index)).T / 1000
        utility_bill_saving = utility_bill_saving.loc[:, choice_heater]

        subsidies_total = pd.DataFrame(0, index=utility_bill_saving.index, columns=utility_bill_saving.columns)
        subsidies_details = {}

        tax = self.vta
        p = [p for p in policies_heater if 'reduced_tax' == p.policy]
        if p:
            tax = p[0].value
            sub = cost_heater * (self.vta - tax)
            subsidies_details[p[0].name] = pd.concat([sub] * utility_bill_saving.shape[0],
                                                     keys=utility_bill_saving.index, axis=1).T

        tax_heater = cost_heater * tax
        cost_heater += tax_heater

        for policy in policies_heater:
            if policy.policy == 'subsidy_target':
                sub = policy.value.reindex(utility_bill_saving.columns, axis=1).fillna(0)
                sub = reindex_mi(sub, utility_bill_saving.index)
            elif policy.policy == 'subsidy_ad_volarem':
                if isinstance(policy.value, (float, int)):
                    sub = policy.value * cost_heater
                    sub = pd.concat([sub] * utility_bill_saving.shape[0], keys=utility_bill_saving.index, axis=1).T
                    if policy.cap:
                        sub[sub > policy.cap] = sub

                if isinstance(policy.value, pd.DataFrame):
                    sub = policy.value * cost_heater
                    sub = reindex_mi(sub, utility_bill_saving.index).fillna(0)
                    if policy.cap:
                        sub[sub > policy.cap] = sub
            else:
                continue

            subsidies_details[policy.name] = sub
            subsidies_total += subsidies_details[policy.name]

        cost_heater = cost_heater.reindex(utility_bill_saving.columns)
        pref_investment = reindex_mi(self.pref_investment, utility_bill_saving.index)
        pref_investment.name = None
        utility_investment = pref_investment.to_frame().dot(cost_heater.to_frame().T) / 1000

        utility_subsidies = subsidies_total * self.pref_subsidy / 1000
        if p:
            subsidies_total += subsidies_details['reduced_tax']

        self.subsidies_total_heater.update({self.year: subsidies_total})
        self.subsidies_details_heater.update({self.year: subsidies_details})

        utility_inertia = pd.DataFrame(0, index=utility_bill_saving.index, columns=utility_bill_saving.columns)
        for hs in choice_heater:
            utility_inertia.loc[self.heater.groupby(levels).first() == hs, hs] = self.pref_inertia

        utility = utility_inertia + utility_investment + utility_bill_saving + utility_subsidies

        restrict_heater = reindex_mi(self._restrict_heater.reindex(utility.columns, axis=1), utility.index).astype(bool)
        utility[restrict_heater] = float('nan')

        if self.constant_heater is None:
            self.constant_heater = self.calibration_constant_heater(utility, ms_heater)
        utility_constant = reindex_mi(self.constant_heater.reindex(utility.columns, axis=1), utility.index)

        utility += utility_constant
        market_share = (np.exp(utility).T / np.exp(utility).sum(axis=1)).T

        probability_replacement = 1 / 17
        replacement = (market_share.T * probability_replacement * self.stock_mobile.groupby(levels).sum()).T

        self.replacement_heater.update({self.year: replacement})
        self.investment_heater.update({self.year: replacement * cost_heater})
        self.tax_heater.update({self.year: replacement * tax_heater})

        self.subsidies_heater.update({self.year: replacement * subsidies_total})

        for key in self.subsidies_details_heater[self.year].keys():
            self.subsidies_details_heater[self.year][key] *= replacement

        to_replace = replacement.sum(axis=1)
        replaced_by = replacement.stack()
        replaced_by = replaced_by.groupby([c for c in replaced_by.index.names if c != 'Heating system']).sum()
        replaced_by.index.set_names({'Heating system final': 'Heating system'}, inplace=True)
        replaced_by = replaced_by.groupby([c for c in replaced_by.index.names if c != 'Heating energy']).sum()
        replaced_by = replaced_by.reorder_levels(to_replace.index.names)

        stock = self.stock_mobile.groupby(levels).sum() - to_replace
        stock_replacement = pd.concat((stock, replaced_by), axis=0, keys=[False, True], names=['Heater replacement'])

        self.heater_replaced.update({self.year: replaced_by})
        return stock_replacement

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
                            axis=1, keys=['constant', 'calcul ini', 'calcul', 'observed'])
        details.to_csv(os.path.join(self._path, 'calibration_constant_heater.csv'))

        return constant

    def prepare_consumption(self):
        """Constitute building components' performance.
        """
        s = pd.concat([self.stock] * len(self._choice_insulation), axis=1)
        s.columns = self._choice_insulation
        wall_buildings, floor_buildings, roof_buildings, windows_buildings = {}, {}, {}, {}
        for name, series in s.iteritems():
            wall, floor, roof, windows = name
            if wall:
                wall_buildings[name] = pd.Series(self._performance_insulation['Wall'], index=series.index)
            else:
                wall_buildings[name] = pd.Series(series.index.get_level_values('Wall'), index=series.index)

            if floor:
                floor_buildings[name] = pd.Series(self._performance_insulation['Floor'], index=series.index)
            else:
                floor_buildings[name] = pd.Series(series.index.get_level_values('Floor'), index=series.index)

            if roof:
                roof_buildings[name] = pd.Series(self._performance_insulation['Roof'], index=series.index)
            else:
                roof_buildings[name] = pd.Series(series.index.get_level_values('Roof'), index=series.index)

            if windows:
                windows_buildings[name] = pd.Series(self._performance_insulation['Windows'], index=series.index)
            else:
                windows_buildings[name] = pd.Series(series.index.get_level_values('Windows'), index=series.index)

        return wall_buildings, floor_buildings, roof_buildings, windows_buildings

    def prepare_cost(self, cost_insulation):
        """Constitute insulation choice set cost. Cost is equal to the  sum of individual cost component.
        """
        cost = pd.Series(0, index=self._choice_insulation)
        idx = pd.IndexSlice
        cost.loc[idx[True, :, :, :]] = cost.loc[idx[True, :, :, :]] + cost_insulation['Wall']
        cost.loc[idx[:, True, :, :]] = cost.loc[idx[:, True, :, :]] + cost_insulation['Floor']
        cost.loc[idx[:, :, True, :]] = cost.loc[idx[:, :, True, :]] + cost_insulation['Roof']
        cost.loc[idx[:, :, :, True]] = cost.loc[idx[:, :, :, True]] + cost_insulation['Windows']
        return cost

    def prepare_subsidy_component(self, subsidies_insulation):
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

    def insulation_retrofit(self, prices, cost_insulation, ms_insulation, ms_extensive, policies_insulation):
        """Calculate insulation retrofit in the dwelling stock.

        1. Intensive margin
        2. Extensive margin
        Calibrate function first year.

        To reduce calculation time attributes are grouped.
        Cost, subsidies and constant depends on Housing type, Occupancy status, Housing type and Insulation performance.

        Parameters
        ----------
        prices: pd.Series
        cost_insulation: pd.Series
        ms_insulation: pd.Series
        ms_extensive: pd.Series
        policies_insulation: list

        Returns
        -------
        pd.Series
            Retrofit rate
        pd.DataFrame
            Market-share insulation
        """
        choice_insulation = self._choice_insulation
        wall_buildings, floor_buildings, roof_buildings, windows_buildings = self.prepare_consumption()

        consumption_sd, certificate = {}, {}
        for name in choice_insulation:
            consumption_sd[name] = self.heating_consumption_sd(wall=wall_buildings[name],
                                                               floor=floor_buildings[name],
                                                               roof=roof_buildings[name],
                                                               windows=windows_buildings[name])

            certificate[name] = self.certificates(wall=wall_buildings[name],
                                                  floor=floor_buildings[name],
                                                  roof=roof_buildings[name],
                                                  windows=windows_buildings[name])

        consumption_sd = pd.DataFrame(consumption_sd)
        consumption_sd.columns.set_names(choice_insulation.names, inplace=True)
        certificate = pd.DataFrame(certificate)
        certificate.columns.set_names(choice_insulation.names, inplace=True)
        certificate_before = self.certificate

        # remove redundant rows (Income tenant)
        levels = [c for c in consumption_sd.index.names if c != 'Income tenant']
        consumption_sd = consumption_sd[certificate_before > 'B']
        consumption_sd = consumption_sd.groupby(levels).first()
        certificate = certificate[certificate_before > 'B']
        certificate = certificate.groupby(levels).first()
        certificate_before = certificate_before[certificate_before > 'B']
        certificate_before = certificate_before.groupby(levels).first()

        surface = self.surface.groupby(levels).first()
        energy_prices = self.energy_prices(prices).groupby(levels).first()
        energy_bill_sd_before = self.energy_bill_sd(prices).groupby(levels).first()

        epc2int = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
        certificate_jump = - certificate.replace(epc2int).sub(certificate_before.replace(epc2int), axis=0)

        self.certificate_jump = certificate_jump
        self.efficient_renovation = certificate.isin(['A', 'B'])

        global_retrofit = certificate_jump >= 2
        out_worst = ~certificate.isin(['G', 'F']).astype(int).mul(certificate_before.isin(['G', 'F']).astype(int),
                                                                  axis=0).astype(bool)
        in_best = certificate.isin(['A', 'B']).astype(int).mul(~certificate_before.isin(['A', 'B']).astype(int),
                                                               axis=0).astype(bool)

        energy_bill_sd = (consumption_sd.T * energy_prices * surface).T
        bill_saved = - energy_bill_sd.sub(energy_bill_sd_before, axis=0).dropna()
        utility_bill_saving = (bill_saved.T * reindex_mi(self.pref_bill, bill_saved.index)).T / 1000

        investment = self.prepare_cost(cost_insulation * self.surface_insulation)
        subsidies_total = pd.DataFrame(0, index=utility_bill_saving.index, columns=utility_bill_saving.columns)
        subsidies_details = {}

        tax = self.vta
        p = [p for p in policies_insulation if 'reduced_tax' == p.policy]
        if p:
            tax = p[0].value
            subsidies_details[p[0].name] = pd.concat([investment * (self.vta - tax)] * utility_bill_saving.shape[0],
                                                     axis=1, keys=utility_bill_saving.index).T
            subsidies_total += subsidies_details[p[0].name]

        self.cost_component.update({self.year: cost_insulation * self.surface_insulation * (1 + tax)})
        tax = investment * tax
        investment += tax
        self.cost_insulation.update({self.year: investment})
        self.tax_insulation.update({self.year: tax})
        investment.reindex(utility_bill_saving.columns)
        investment = surface.rename(None).to_frame().dot(investment.to_frame().T).reindex(utility_bill_saving.index)

        for policy in policies_insulation:
            if policy.policy == 'subsidy_target':
                subsidies_details[policy.name] = (reindex_mi(self.prepare_subsidy_component(policy.value),
                                                  utility_bill_saving.index).T * surface).T
                subsidies_total += subsidies_details[policy.name]

            elif policy.policy == 'bonus_best':
                subsidies_details[policy.name] = (reindex_mi(policy.value, in_best.index) * in_best.T).T
                subsidies_total += subsidies_details[policy.name]

            elif policy.policy == 'bonus_worst':
                subsidies_details[policy.name] = (reindex_mi(policy.value, out_worst.index) * out_worst.T).T
                subsidies_total += subsidies_details[policy.name]

            elif policy.policy == 'subsidy_ad_volarem':
                subsidies_details[policy.name] = policy.value * investment
                subsidies_total += subsidies_details[policy.name]

        pref_investment = reindex_mi(self.pref_investment, utility_bill_saving.index).rename(None)
        utility_investment = (investment.reindex(pref_investment.index).T * pref_investment).T / 1000

        subsidies_cap = [p for p in policies_insulation if p.policy == 'subsidies_cap']
        if subsidies_cap:
            subsidies_cap = subsidies_cap[0]
            subsidies_cap = reindex_mi(subsidies_cap.value, subsidies_total.index)
            cap = (investment.T * subsidies_cap).T
            over_cap = subsidies_total > cap
            subsidies_details['over_cap'] = (subsidies_total - cap)[over_cap].fillna(0)

            subsidies_total -= subsidies_details['over_cap']
        self.subsidies_details_insulation.update({self.year: subsidies_details})
        self.subsidies_total_insulation.update({self.year: subsidies_total})
        utility_subsidies = (self.pref_subsidy * subsidies_total) / 1000

        utility = utility_bill_saving + utility_investment + utility_subsidies

        if self.utility_insulation_intensive is None:
            self.utility_insulation_intensive = self.calibration_constant_intensive(utility, ms_insulation)

        utility += self.utility_insulation_intensive

        # restrict to coherent renovation work
        idx = pd.IndexSlice
        utility.loc[self.wall.groupby(levels).first().reindex(utility.index) <= self._performance_insulation['Wall'], idx[True, :, :, :]] = float('nan')
        utility.loc[self.floor.groupby(levels).first().reindex(utility.index) <= self._performance_insulation['Floor'], idx[:, True, :, :]] = float('nan')
        utility.loc[self.roof.groupby(levels).first().reindex(utility.index) <= self._performance_insulation['Roof'], idx[:, :, True, :]] = float('nan')
        utility.loc[self.windows.groupby(levels).first().reindex(utility.index) <= self._performance_insulation['Windows'], idx[:, :, :, True]] = float('nan')
        utility.dropna(inplace=True)

        market_share = (np.exp(utility).T / np.exp(utility).sum(axis=1)).T

        # extensive margin
        bill_saved_insulation = (bill_saved.reindex(market_share.index) * market_share).sum(axis=1)
        utility_bill_saving = reindex_mi(self.pref_bill, bill_saved_insulation.index) * bill_saved_insulation / 1000

        investment_insulation = (investment.reindex(market_share.index) * market_share).sum(axis=1)
        pref_investment = reindex_mi(self.pref_investment, utility_bill_saving.index).rename(None)
        utility_investment = (pref_investment * investment_insulation) / 1000

        utility = utility_investment + utility_bill_saving

        if self.utility_insulation_extensive is None:
            self.utility_insulation_extensive = self.calibration_constant_extensive(utility, ms_extensive)

        utility = pd.concat([utility, utility], keys=[True, False], names=['Heater replacement'])
        utility += self.utility_insulation_extensive
        retrofit_rate = 1 / (1 + np.exp(- utility))
        return retrofit_rate, market_share

    def calibration_constant_intensive(self, utility, ms_insulation):
        """Calibrate alternative-specific constant to match observed market-share.

        Parameters
        ----------
        utility: pd.Series
        ms_insulation: pd.Series
            Observed market-share.

        Returns
        -------
        pd.Series
        """

        # removing unnecessary level
        utility_ref = utility.droplevel(['Occupancy status']).copy()

        utility_ref = utility_ref[~utility_ref.index.duplicated(keep='first')]

        stock = self.stock.groupby(utility_ref.index.names).sum().reindex(utility_ref.index)

        constant = ms_insulation.reindex(utility.columns, axis=0).copy()
        constant[constant > 0] = 0
        for i in range(150):
            utility = (utility_ref + constant).copy()
            constant.iloc[0] = 0
            market_share = (np.exp(utility).T / np.exp(utility).sum(axis=1)).T
            agg = (market_share.T * stock).T
            market_share_agg = (agg.sum() / agg.sum().sum()).reindex(ms_insulation.index)
            if i == 0:
                market_share_ini = market_share_agg.copy()
            constant = constant + np.log(ms_insulation / market_share_agg)

            if (market_share_agg.round(decimals=2) == ms_insulation.round(decimals=2)).all():
                print('Constant intensive optim worked')
                break

        constant.iloc[0] = 0
        details = pd.concat((constant, market_share_ini, market_share_agg, ms_insulation), axis=1,
                            keys=['constant', 'calcul ini', 'calcul', 'observed'])
        details.to_csv(os.path.join(self._path, 'calibration_constant_insulation.csv'))

        return constant

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

        utility_ref = utility.copy()
        utility_ref = pd.concat([utility_ref, utility_ref], keys=[True, False], names=['Heater replacement'])
        idx = utility_ref.index.copy()
        # utility_ref = utility_ref.droplevel(['Income tenant'])
        # utility_ref = utility_ref[~utility_ref.index.duplicated(keep='first')]

        stock = pd.concat([self.stock, self.stock], keys=[True, False], names=['Heater replacement']).groupby(
            utility_ref.index.names).sum().reindex(utility_ref.index)
        stock = stock.groupby(utility_ref.index.names).sum()

        constant = ms_extensive.copy()
        constant[ms_extensive > 0] = 0

        for i in range(100):
            utility_constant = reindex_mi(constant, utility_ref.index)
            utility = (utility_ref + utility_constant).copy()

            market_share = 1 / (1 + np.exp(- utility))
            market_share_agg = (market_share * stock).groupby(ms_extensive.index.names).sum() / stock.groupby(
                ms_extensive.index.names).sum()
            if i == 0:
                market_share_ini = market_share_agg.copy()
            constant = constant + np.log(ms_extensive / market_share_agg)

            if (market_share_agg.round(decimals=3) == ms_extensive.round(decimals=3).reindex(
                    market_share_agg.index)).all():
                print('Constant extensive optim worked')
                break

        utility_constant = reindex_mi(constant, idx)
        details = pd.concat((constant, market_share_ini, market_share_agg, ms_extensive), axis=1,
                            keys=['constant', 'calcul ini', 'calcul', 'observed'])
        details.to_csv(os.path.join(self._path, 'calibration_constant_extensive.csv'))

        return utility_constant

    def flow_retrofit(self, prices, cost_heater, ms_heater, cost_insulation, ms_insulation, ms_extensive,
                      policies_heater, policies_insulation):
        """Compute heater replacement and insulation retrofit.

        Parameters
        ----------
        prices
        cost_heater
        ms_heater
        cost_insulation
        ms_insulation
        ms_extensive
        policies_heater
        policies_insulation

        Returns
        -------

        """
        t0 = time()
        stock_replacement = self.heater_replacement(prices, cost_heater, ms_heater, policies_heater)
        t1 = time()
        print('heater_replacement {:.0f}'.format((t1 - t0)))
        retrofit_rate, market_share = self.insulation_retrofit(prices, cost_insulation, ms_insulation,
                                                               ms_extensive, policies_insulation)
        retrofit_rate = reindex_mi(retrofit_rate, stock_replacement.index)

        t2 = time()
        print('insulation_retrofit {:.0f}'.format((t2 - t1)))

        # TODO: intersection entre stock_replacement et stock_mobile
        retrofit_stock = (retrofit_rate * stock_replacement).dropna().groupby(
            [c for c in stock_replacement.index.names if c != 'Heater replacement']).sum()
        market_share = reindex_mi(market_share, retrofit_stock.index)
        replaced_by = (retrofit_stock * market_share.T).T.copy()

        self.efficient_renovation_yrs.update({self.year: (replaced_by * self.efficient_renovation).sum().sum()})

        rslt = {}
        for i in range(6):
            rslt.update({i: ((self.certificate_jump == i) * replaced_by).sum(axis=1)})
        self.certificate_jump_yrs.update({self.year: pd.DataFrame(rslt)})

        surface = self.surface.groupby(replaced_by.index.names).first()

        self.replacement_insulation.update({self.year: replaced_by})
        self.investment_insulation.update(
            {self.year: (replaced_by.T * surface).T.dropna() * self.cost_insulation[self.year]})
        self.tax_insulation.update(
            {self.year: (replaced_by.T * surface).T.dropna() * self.tax_insulation[self.year]})
        self.subsidies_insulation.update(
            {self.year: self.subsidies_total_insulation[self.year] * replaced_by.copy()})

        for key in self.subsidies_details_insulation[self.year].keys():
            self.subsidies_details_insulation[self.year][key] = replaced_by * reindex_mi(
                self.subsidies_details_insulation[self.year][key], replaced_by.index)
        t3 = time()
        print('store {:.0f}'.format((t3 - t2)))

        share = (self.stock_mobile.unstack('Income tenant').T / self.stock_mobile.unstack('Income tenant').sum(axis=1)).T
        temp = pd.concat([replaced_by] * share.shape[1], keys=share.columns, names=share.columns.names, axis=1)
        share = reindex_mi(share, temp.columns, axis=1).reindex(temp.index)
        replaced_by = (share * temp).stack('Income tenant').dropna()

        to_replace = replaced_by.sum(axis=1).copy()

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

        flow_retrofit = replaced_by.sub(to_replace, fill_value=0)
        flow_retrofit = flow_retrofit[flow_retrofit != 0]
        flow_retrofit = flow_retrofit.groupby(flow_retrofit.index).sum()
        flow_retrofit.index = pd.MultiIndex.from_tuples(flow_retrofit.index, names=replaced_by.index.names)
        t4 = time()
        print('store {:.0f}'.format((t4 - t3)))
        return flow_retrofit

    def flow_demolition(self):
        """Demolition of E, F and G buildings based on their share in the mobile stock.

        Returns
        -------
        pd.Series
        """
        stock_demolition = self.stock_mobile[self.certificate.isin(['E', 'F', 'G'])]
        stock_demolition = stock_demolition / stock_demolition.sum()
        flow_demolition = (stock_demolition * self._demolition_total).dropna()
        return flow_demolition.reorder_levels(self.index.names)


