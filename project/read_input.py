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
from itertools import product
from utils import reindex_mi


class PublicPolicy:
    """Public policy parent class.

    Attributes
    ----------
    name : str
        Name of the policy.
    start : int
        Year policy starts.
    end : int
        Year policy ends.
    policy : {'energy_taxes', 'subsidies'}

    """
    def __init__(self, name, start, end, value, policy, gest=None, cap=None):
        self.name = name
        self.start = start
        self.end = end
        self.policy = policy
        self.value = value
        self.gest = gest
        self.cap = cap


def read_stock(config):
    stock = pd.read_csv(config['building_stock'], index_col=[0, 1, 2, 3, 4, 5, 6, 7, 8]).squeeze('columns')
    year = config['start']

    stock = pd.concat([stock], keys=[True], names=['Existing'])

    idx_names = ['Existing', 'Occupancy status', 'Income owner', 'Income tenant', 'Housing type', 'Heating system',
                 'Wall', 'Floor', 'Roof', 'Windows']
    stock = stock.reorder_levels(idx_names)

    return stock, year


def read_policies(config):
    def read_mpr(data):
        l = list()
        heater = pd.read_csv(data['heater'], index_col=[0, 1]).squeeze('columns').unstack('Heating system')
        insulation = pd.read_csv(data['insulation'], index_col=[0])
        bonus_best = pd.read_csv(data['bonus'], index_col=[0]).squeeze('columns')
        bonus_worst = pd.read_csv(data['bonus'], index_col=[0]).squeeze('columns')

        l.append(PublicPolicy('mpr', data['start'], data['end'], heater, 'subsidy_target', gest='heater'))
        l.append(PublicPolicy('mpr', data['start'], data['end'], insulation, 'subsidy_target', gest='insulation'))
        l.append(PublicPolicy('mpr', data['start'], data['end'], bonus_best, 'bonus_best', gest='insulation'))
        l.append(PublicPolicy('mpr', data['start'], data['end'], bonus_worst, 'bonus_worst', gest='insulation'))

        return l

    def read_cee(data):
        l = list()
        heater = pd.read_csv(data['heater'], index_col=[0, 1]).squeeze('columns').unstack('Heating system')
        insulation = pd.read_csv(data['insulation'], index_col=[0])
        tax = pd.read_csv(data['tax'], index_col=[0])

        l.append(PublicPolicy('cee', data['start'], data['end'], tax, 'tax'))
        l.append(PublicPolicy('cee', data['start'], data['end'], heater, 'subsidy_target', gest='heater'))
        l.append(PublicPolicy('cee', data['start'], data['end'], insulation, 'subsidy_target', gest='insulation'))
        return l

    def read_cap(data):
        cap = pd.read_csv(data['insulation'], index_col=[0]).squeeze('columns')
        return [PublicPolicy('subsidies_cap', data['start'], data['end'], cap, 'subsidies_cap', gest='insulation')]

    def read_carbon_tax(data):
        tax = pd.read_csv(data['tax'], index_col=[0]).squeeze('columns')
        emission = pd.read_csv(data['emission'], index_col=[0]).squeeze('columns')
        tax = (tax * emission).fillna(0) / 10 ** 6
        tax = tax.loc[(tax != 0).any(axis=1)]
        return [PublicPolicy('carbon_tax', data['start'], data['end'], tax, 'tax')]

    def read_cite(data):
        l = list()
        heater = pd.read_csv(data['heater'], index_col=[0, 1]).squeeze('columns').unstack('Heating system')
        l.append(PublicPolicy('cite', data['start'], data['end'], heater, 'subsidy_ad_volarem', gest='heater',
                              cap=data['cap']))
        l.append(
            PublicPolicy('cite', data['start'], data['end'], data['insulation'], 'subsidy_ad_volarem', gest='insulation',
                         cap=data['cap']))
        return l

    def read_reduced_tax(data):
        l = list()
        l.append(PublicPolicy('reduced_tax', data['start'], data['end'], data['value'], 'reduced_tax', gest='heater'))
        l.append(
            PublicPolicy('reduced_tax', data['start'], data['end'], data['value'], 'reduced_tax', gest='insulation'))
        return l

    read = {'mpr': read_mpr, 'cee': read_cee, 'cap': read_cap, 'carbon_tax': read_carbon_tax,
            'cite': read_cite, 'reduced_tax': read_reduced_tax}

    list_policies = list()
    for key, item in config['policies'].items():
        list_policies += read[key](item)

    policies_heater = [p for p in list_policies if p.gest == 'heater']
    policies_insulation = [p for p in list_policies if p.gest == 'insulation']
    taxes = [p for p in list_policies if p.policy == 'tax']

    return policies_heater, policies_insulation, taxes


def read_exogenous(config):
    cost_heater = pd.read_csv(config['cost_heater'], index_col=[0]).squeeze('columns').rename(None)
    cost_insulation = pd.read_csv(config['cost_insulation'], index_col=[0]).squeeze('columns').rename(None)

    energy_prices = pd.read_csv(config['energy_prices'], index_col=[0])

    return energy_prices, cost_heater, cost_insulation


def read_revealed(config):
    efficiency = pd.read_csv(config['efficiency'], index_col=[0]).squeeze('columns')

    choice_insulation = {'Wall': [False, True], 'Floor': [False, True], 'Roof': [False, True], 'Windows': [False, True]}
    names = list(choice_insulation.keys())
    choice_insulation = list(product(*[i for i in choice_insulation.values()]))
    choice_insulation.remove((False, False, False, False))
    choice_insulation = pd.MultiIndex.from_tuples(choice_insulation, names=names)

    ms_heater = pd.read_csv(config['ms_heater'], index_col=[0])

    restrict_heater = ms_heater < 0.01
    restrict_heater.loc['Air/air heat pump', ['Direct electric', 'Oil boiler']] = True
    restrict_heater.loc[
        'Water/air heat pump', ['Direct electric', 'Oil boiler', 'Air/air heat pump', 'Gas boiler']] = True
    restrict_heater.loc['Direct electric', ['Oil boiler']] = True
    restrict_heater.loc['Gas boiler', ['Oil boiler']] = True
    restrict_heater.loc['Wood boiler', ['Oil boiler']] = True

    ms_heater[restrict_heater] = float('nan')
    ms_heater = (ms_heater.T / ms_heater.sum(axis=1)).T

    choice_heater = list(ms_heater.columns)

    ms_extensive = pd.read_csv(config["insulation_extensive"], index_col=[0, 1]).squeeze('columns').rename(None).round(
        decimals=3)
    ms_intensive = pd.read_csv(config["ms_insulation"], index_col=[0, 1, 2, 3]).squeeze('columns').rename(None).round(
        decimals=3)

    return efficiency, choice_insulation, ms_heater, restrict_heater, choice_heater, ms_extensive, ms_intensive


def parse_parameters(config, param, stock):
    """Macro module : run exogenous dynamic parameters.

    Parameters
    ----------
    config
    param
    stock

    Returns
    -------

    """
    from dynamic import stock_need, share_multi_family, evolution_surface_built, share_type_built

    population = pd.read_csv(config['population'], header=None, index_col=[0]).squeeze('columns')
    param['population_total'] = population
    param['sizing_factor'] = stock.sum() / param['stock_ini']
    param['population'] = population * param['sizing_factor']
    param['stock_need'], param['pop_housing'] = stock_need(param['population'],
                                                           param['population'][config['start']] / stock.sum(),
                                                           param['pop_housing_min'],
                                                           config['start'], param['factor_pop_housing'])
    param['share_multi_family'] = share_multi_family(param['stock_need'], param['factor_multi_family'])

    idx = range(config['start'], config['end'])
    param['available_income'] = pd.Series(
        [param['available_income'] * (1 + config['income_rate']) ** (i - idx[0]) for i in idx], index=idx)
    param['available_income_pop'] = param['available_income'] / param['population_total']

    param['demolition_rate'] = config['demolition_rate']
    param['flow_demolition'] = pd.Series(param['demolition_rate'] * stock.sum(), index=idx[1:])

    param['flow_need'] = param['stock_need'] - param['stock_need'].shift(1)
    param['flow_construction'] = param['flow_need'] + param['flow_demolition']

    surface_built = evolution_surface_built(param['surface'].xs(False, level='Existing'), param['surface_max'],
                                            param['surface_elasticity'], param['available_income_pop'])

    surface_existing = pd.concat([param['surface'].xs(True, level='Existing')] * surface_built.shape[1], axis=1,
                                 keys=surface_built.columns)
    param['surface'] = pd.concat((surface_existing, surface_built), axis=0, keys=[True, False], names=['Existing'])

    type_built = share_type_built(param['stock_need'], param['share_multi_family'], param['flow_construction']) * param[
        'flow_construction']

    share_decision_maker = stock.groupby(
        ['Occupancy status', 'Housing type', 'Income owner', 'Income tenant']).sum().unstack(
        ['Occupancy status', 'Income owner', 'Income tenant'])
    share_decision_maker = (share_decision_maker.T / share_decision_maker.sum(axis=1)).T
    share_decision_maker = pd.concat([share_decision_maker] * type_built.shape[1], keys=type_built.columns, axis=1)
    construction = (reindex_mi(type_built, share_decision_maker.columns, axis=1) * share_decision_maker).stack(
        ['Occupancy status', 'Income owner', 'Income tenant']).fillna(0)

    ms_heater_built = pd.read_csv(config['ms_heater_built'], index_col=[0])
    ms_heater_built.columns.set_names(['Heating system'], inplace=True)
    ms_heater_built.index.set_names(['Housing type'], inplace=True)
    ms_heater_built = pd.concat([ms_heater_built] * construction.shape[1], axis=1, keys=construction.columns)
    construction = (reindex_mi(ms_heater_built, construction.index) * reindex_mi(construction, ms_heater_built.columns,
                                                                                 axis=1)).stack('Heating system')
    construction = construction.loc[(construction != 0).any(axis=1)]

    performance_insulation = pd.concat([pd.Series(param['performance_insulation'])] * construction.shape[0], axis=1,
                                       keys=construction.index).T
    param['flow_built'] = pd.concat((construction, performance_insulation), axis=1).set_index(
        list(performance_insulation.keys()), append=True)

    param['flow_built'] = pd.concat([param['flow_built']], keys=[False], names=['Existing']).reorder_levels(
        stock.index.names)

    param['health_cost'] = pd.read_csv(config['health_cost'], index_col=[0, 1]).squeeze('columns')
    param['carbon_value'] = pd.read_csv(config['carbon_value'], index_col=[0]).squeeze('columns')
    param['carbon_emission'] = pd.read_csv(config['carbon_emission'], index_col=[0])
    param['carbon_value_kwh'] = (param['carbon_value'] * param['carbon_emission'].T).T.dropna() / 10**6

    param['data_ceren'] = pd.read_csv(config['data_ceren'], index_col=[0]).squeeze('columns')

    summary_param = dict()
    summary_param['Sizing factor (%)'] = pd.Series(param['sizing_factor'], index=param['population'].index)
    summary_param['Total population (Millions)'] = param['population'] / 10 ** 6
    summary_param['Income (Billions euro)'] = param['available_income'] * param['sizing_factor'] / 10 ** 9
    summary_param['Buildings stock (Millions)'] = param['stock_need'] / 10 ** 6
    summary_param['Buildings additional (Thousands)'] = param['flow_need'] / 10 ** 6
    summary_param['Buildings built (Thousands)'] = param['flow_construction'] / 10 ** 3
    summary_param['Buildings demolished (Thousands)'] = param['flow_demolition'] / 10 ** 3
    summary_param['Person by housing'] = param['pop_housing']
    summary_param['Share multi-family (%)'] = param['share_multi_family']
    summary_param = pd.DataFrame(summary_param)
    summary_param = summary_param.loc[config['start']:, :]
    return param, summary_param
