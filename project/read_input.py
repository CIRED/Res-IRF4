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
import logging

logger = logging.getLogger(__name__)


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
    def __init__(self, name, start, end, value, policy, gest=None, cap=None, target=None, cost_min=None, cost_max=None, design=None):
        self.name = name
        self.start = start
        self.end = end
        self.value = value
        self.policy = policy
        self.gest = gest
        self.cap = cap
        self.target = target
        self.cost_max = cost_max
        self.cost_min = cost_min
        self.design = design

    def cost_targeted(self, cost_insulation, certificate=None, energy_saved_3uses=None,  target_subsidies=None):
        cost = cost_insulation.copy()
        idx = pd.IndexSlice
        if self.design:
            target_0 = certificate.isin(['E','D', 'C', 'B', 'A']).astype(bool)
            target_1 = energy_saved_3uses[energy_saved_3uses >= 0.35].fillna(0).astype(bool)
            target_global = target_0 & target_1
            cost_global = cost[target_global].fillna(0)
            cost_global[cost_global > 50000] = 50000 # Useless cause doesn't exist

            cost_isol = cost[~target_global].fillna(0)
            cost_isol[cost_isol.loc[:, idx[False, False, False, True]] > 7000] = 7000 #useless cause doesn't exist
            cost_isol[cost_isol.loc[:, [c for c in cost_isol.columns if (sum(idx[c]) == 1)]] > 15000] = 15000  # It's overlapping with the line just above but 15000>7000 so not a problem
            cost_isol[cost_isol.loc[:, [c for c in cost_isol.columns if (sum(idx[c]) == 2)]] > 25000] = 25000
            cost_isol[cost_isol.loc[:, [c for c in cost_isol.columns if (sum(idx[c]) > 2)]] > 30000] = 30000
            cost = cost_global + cost_isol
        else:
            if self.target is not None and target_subsidies is not None:
                cost = cost[target_subsidies].fillna(0)
        if self.cost_max is not None:
                cost_max = reindex_mi(self.cost_max, cost.index)
                cost_max = pd.concat([cost_max] * cost.shape[1], axis=1).set_axis(cost.columns, axis=1)
                cost[cost > cost_max] = cost_max
        if self.cost_min is not None:
            cost_min = reindex_mi(self.cost_min, cost.index)
            cost_min = pd.concat([cost_min] * cost.shape[1], axis=1).set_axis(
                cost.columns, axis=1)
            cost[cost < cost_min] = 0
        return cost


def read_stock(config):
    stock = pd.read_csv(config['building_stock'], index_col=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).squeeze('columns')
    year = config['start']

    stock = pd.concat([stock], keys=[True], names=['Existing'])

    idx_names = ['Existing', 'Occupancy status', 'Income owner', 'Income tenant', 'Housing type',
                 'Heating system', 'Wall', 'Floor', 'Roof', 'Windows']
    stock = stock.reset_index(level=['Heating energy', 'Heating system'])
    stock['Heating system'].replace({3: 'Heat pump'}, inplace=True)
    stock['Heating system'].replace({0.95: 'Performance boiler'}, inplace=True)
    stock['Heating system'].replace({0.85: 'Performance boiler'}, inplace=True)
    stock['Heating system'].replace({0.75: 'Standard boiler'}, inplace=True)

    stock['Heating system'] = stock['Heating energy'] + '-' + stock['Heating system']
    stock = stock.set_index(['Heating system'], append=True).loc[:, 'Stock buildings']
    stock = stock.reorder_levels(idx_names)

    return stock, year


def read_policies(config):
    def read_mpr(data):
        l = list()
        heater = pd.read_csv(data['heater'], index_col=[0, 1]).squeeze('columns').unstack('Heating system')
        insulation = pd.read_csv(data['insulation'], index_col=[0])

        if data['global_retrofit']:
            if isinstance(data['global_retrofit'], dict):
                global_retrofit = pd.read_csv(data['global_retrofit']['value'], index_col=[0]).squeeze('columns')
                l.append(PublicPolicy('mpr', data['global_retrofit']['start'], data['global_retrofit']['end'], global_retrofit, 'subsidy_non_cumulative',
                                  gest='insulation'))
            else:
                global_retrofit = pd.read_csv(data['global_retrofit'], index_col=[0]).squeeze('columns')
                l.append(PublicPolicy('mpr', data['start'], data['end'], global_retrofit, 'subsidy_non_cumulative',
                                  gest='insulation'))


        if data['mpr_serenite']:
            if isinstance(data['mpr_serenite'], dict):
                mpr_serenite = pd.read_csv(data['mpr_serenite']['value'], index_col=[0]).squeeze('columns')
                l.append(PublicPolicy('mpr', data['mpr_serenite']['start'], data['mpr_serenite']['end'], mpr_serenite, 'subsidy_non_cumulative',
                                  gest='insulation'))
            else:
                mpr_serenite = pd.read_csv(data['mpr_serenite'], index_col=[0]).squeeze('columns')
                l.append(PublicPolicy('mpr', data['start'], data['end'], mpr_serenite, 'subsidy_non_cumulative',
                                  gest='insulation'))

        if data['bonus']:
            if isinstance(data['bonus'], dict):
                bonus_best = pd.read_csv(data['bonus']['value'], index_col=[0]).squeeze('columns')
                bonus_worst = pd.read_csv(data['bonus']['value'], index_col=[0]).squeeze('columns')
                l.append(PublicPolicy('mpr', data['bonus']['start'], data['bonus']['end'], bonus_best, 'bonus_best', gest='insulation'))
                l.append(PublicPolicy('mpr', data['bonus']['start'], data['bonus']['end'], bonus_worst, 'bonus_worst', gest='insulation'))
            else:
                bonus_best = pd.read_csv(data['bonus'], index_col=[0]).squeeze('columns')
                bonus_worst = pd.read_csv(data['bonus'], index_col=[0]).squeeze('columns')
                l.append(PublicPolicy('mpr', data['start'], data['end'], bonus_best, 'bonus_best', gest='insulation'))
                l.append(PublicPolicy('mpr', data['start'], data['end'], bonus_worst, 'bonus_worst', gest='insulation'))

        l.append(PublicPolicy('mpr', data['start'], data['end'], heater, 'subsidy_target', gest='heater'))
        l.append(PublicPolicy('mpr', data['start'], data['end'], insulation, 'subsidy_target', gest='insulation'))

        return l

    def read_cee(data):
        l = list()
        heater = pd.read_csv(data['heater'], index_col=[0, 1]).squeeze('columns').unstack('Heating system')
        insulation = pd.read_csv(data['insulation'], index_col=[0])
        tax = pd.read_csv(data['tax'], index_col=[0])

        l.append(PublicPolicy('cee', data['start'], data['end'], tax.loc[data['start']:data['end']-1, :], 'tax'))
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
        return [PublicPolicy('carbon_tax', data['start'], data['end'], tax.loc[data['start']:data['end']-1, :], 'tax')]

    def read_cite(data):
        l = list()
        heater = pd.read_csv(data['heater'], index_col=[0, 1]).squeeze('columns').unstack('Heating system')
        l.append(PublicPolicy('cite', data['start'], data['end'], heater, 'subsidy_ad_volarem', gest='heater',
                              cap=data['cap']))
        l.append(
            PublicPolicy('cite', data['start'], data['end'], data['insulation'], 'subsidy_ad_volarem', gest='insulation',
                         cap=data['cap']))
        return l

    def read_zil(data):
        data_max = pd.read_csv(data['max'], index_col=[0]).squeeze()

        if data['ad_volarem']:
            return [
                PublicPolicy('zero_interest_loan', data['start'], data['end'], data['value'], 'subsidy_ad_volarem',
                             target=True, cost_min=data['min'], cost_max=data_max, gest='insulation', design=data['design2019'])]
        else:
            return [
                PublicPolicy('zero_interest_loan', data['start'], data['end'], data['value'], 'zero_interest_loan',
                             gest='insulation', target=True, cost_max=data_max, cost_min=data['min'], design=data['design2019'])]

    def read_reduced_tax(data):
        l = list()
        l.append(PublicPolicy('reduced_tax', data['start'], data['end'], data['value'], 'reduced_tax', gest='heater'))
        l.append(
            PublicPolicy('reduced_tax', data['start'], data['end'], data['value'], 'reduced_tax', gest='insulation'))
        return l

    def read_ad_volarem(data):
        l = list()
        value = data['value']
        l.append(PublicPolicy('sub_ad_volarem', data['start'], data['end'], value, 'subsidy_ad_volarem',
                              gest='heater'))
        l.append(PublicPolicy('sub_ad_volarem', data['start'], data['end'], value, 'subsidy_ad_volarem',
                              gest='insulation'))
        return l

    def read_oil_fuel_elimination(data):
        return [PublicPolicy('oil_fuel_elimination', data['start'], data['end'], data['value'],
                             'heater_regulation', gest='heater')]

    read = {'mpr': read_mpr, 'cee': read_cee, 'cap': read_cap, 'carbon_tax': read_carbon_tax,
            'cite': read_cite, 'reduced_tax': read_reduced_tax, 'zero_interest_loan': read_zil,
            'sub_ad_volarem': read_ad_volarem, 'oil_fuel_elimination': read_oil_fuel_elimination}

    list_policies = list()
    for key, item in config['policies'].items():
        if key in read.keys():
            list_policies += read[key](item)
        else:
            print('{} reading function is not implemented'.format(key))

    policies_heater = [p for p in list_policies if p.gest == 'heater']
    policies_insulation = [p for p in list_policies if p.gest == 'insulation']
    taxes = [p for p in list_policies if p.policy == 'tax']

    return policies_heater, policies_insulation, taxes


def read_exogenous(config):

    cost_heater = pd.read_csv(config['cost_heater'], index_col=[0]).squeeze('columns').rename(None) * config[
        'cost_factor']
    cost_insulation = pd.read_csv(config['cost_insulation'], index_col=[0]).squeeze('columns').rename(None) * config[
        'cost_factor']

    energy_prices = pd.read_csv(config['energy_prices'], index_col=[0])
    energy_prices.loc[range(config['start'] + 2, config['end']), :] *= config['prices_factor']

    energy_taxes = pd.read_csv(config['energy_taxes'], index_col=[0])
    energy_taxes.loc[range(config['start'] + 2, config['end']), :] *= config['prices_factor']

    return energy_prices, energy_taxes, cost_heater, cost_insulation


def read_revealed(config):
    efficiency = pd.read_csv(config['efficiency'], index_col=[0]).squeeze('columns')

    choice_insulation = {'Wall': [False, True], 'Floor': [False, True], 'Roof': [False, True], 'Windows': [False, True]}
    names = list(choice_insulation.keys())
    choice_insulation = list(product(*[i for i in choice_insulation.values()]))
    choice_insulation.remove((False, False, False, False))
    choice_insulation = pd.MultiIndex.from_tuples(choice_insulation, names=names)

    # ms_heater = pd.read_csv(config['ms_heater'], index_col=[0])

    #for housing type MS
    ms_heater = pd.read_csv(config['ms_heater'], index_col=[0, 1])
    ms_heater.columns.set_names('Heating system final', inplace=True)

    """
    #Values are over 0.01, but this allows to replace the 0 with Nan if there are any left
    restrict_heater = ms_heater < 0.01
    ms_heater[restrict_heater] = float('nan')
    # renormalizing (not really useful because input is normalized)
    ms_heater = (ms_heater.T / ms_heater.sum(axis=1)).T"""

    #ms_heater = ms_heater.dropna(axis=1, how='all')
    choice_heater = list(ms_heater.columns)

    df = pd.read_csv(config['renovation_rate_ini'])
    renovation_rate_ini = df.set_index(list(df.columns[:-1])).squeeze().rename(None).round(decimals=3)

    ms_intensive = pd.read_csv(config['ms_insulation'], index_col=[0, 1, 2, 3]).squeeze('columns').rename(None).round(
        decimals=3)

    return efficiency, choice_insulation, ms_heater, choice_heater, renovation_rate_ini, ms_intensive


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

    if config['pop_housing'] is None:
        param['stock_need'], param['pop_housing'] = stock_need(param['population'],
                                                               param['population'][config['start']] / stock.sum(),
                                                               param['pop_housing_min'],
                                                               config['start'], param['factor_pop_housing'])
    elif config['pop_housing'] == 'constant':
        pass
    else:
        param['pop_housing'] = pd.read_csv(config['pop_housing'], index_col=[0], header=None).squeeze()
        param['stock_need'] = param['population'] / param['pop_housing']

    if config['share_multi_family'] is None:
        param['share_multi_family'] = share_multi_family(param['stock_need'], param['factor_multi_family'])
    elif config['share_multi_family'] == 'constant':
        pass
    else:
        param['share_multi_family'] = pd.read_csv(config['share_multi_family'], index_col=[0], header=None).squeeze()

    idx = range(config['start'], config['end'])
    param['available_income'] = pd.Series(
        [param['available_income'] * (1 + config['income_rate']) ** (i - idx[0]) for i in idx], index=idx)
    param['available_income_pop'] = param['available_income'] / param['population_total']

    param['demolition_rate'] = config['demolition_rate']
    param['flow_demolition'] = pd.Series(param['demolition_rate'] * stock.sum(), index=idx[1:])

    param['flow_need'] = param['stock_need'] - param['stock_need'].shift(1)
    param['flow_construction'] = param['flow_need'] + param['flow_demolition']

    if config['surface_built'] == 'endogenous':
        surface_built = evolution_surface_built(param['surface'].xs(False, level='Existing'), param['surface_max'],
                                                param['surface_elasticity'], param['available_income_pop'])
    elif config['surface_built'] == 'surface_built':
        pass
    else:
        surface_built = pd.read_csv(config['surface_built'], index_col=[0]).squeeze().rename(None)

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

    ms_heater_built = pd.read_csv(config['ms_heater_built'], index_col=[0], header=[0])
    ms_heater_built.columns.set_names(['Heating system'], inplace=True)
    ms_heater_built.index.set_names(['Housing type'], inplace=True)

    ms_heater_built = reindex_mi(ms_heater_built, construction.index).stack()
    construction = (reindex_mi(construction, ms_heater_built.index).T * ms_heater_built).T
    construction = construction.loc[(construction != 0).any(axis=1)]

    performance_insulation = pd.concat([pd.Series(param['performance_insulation'])] * construction.shape[0], axis=1,
                                       keys=construction.index).T
    param['flow_built'] = pd.concat((construction, performance_insulation), axis=1).set_index(
        list(performance_insulation.keys()), append=True)

    param['flow_built'] = pd.concat([param['flow_built']], keys=[False], names=['Existing']).reorder_levels(
        stock.index.names)

    if not config['construction']:
        param['flow_built'][param['flow_built'] > 0] = 0

    df = pd.read_csv(config['health_cost'], index_col=[0, 1])
    param['health_expenditure'] = df['Health expenditure']
    param['mortality_cost'] = df['Social cost of mortality']
    param['loss_well_being'] = df['Loss of well-being']
    param['carbon_value'] = pd.read_csv(config['carbon_value'], index_col=[0]).squeeze('columns')
    param['carbon_emission'] = pd.read_csv(config['carbon_emission'], index_col=[0])
    param['carbon_value_kwh'] = (param['carbon_value'] * param['carbon_emission'].T).T.dropna() / 10**6

    param['data_ceren'] = pd.read_csv(config['data_ceren'], index_col=[0]).squeeze('columns')

    summary_param = dict()
    summary_param['Sizing factor (%)'] = pd.Series(param['sizing_factor'], index=param['population'].index)
    summary_param['Total population (Millions)'] = param['population'] / 10**6
    summary_param['Income (Billions euro)'] = param['available_income'] * param['sizing_factor'] / 10**9
    summary_param['Buildings stock (Millions)'] = param['stock_need'] / 10**6
    summary_param['Buildings additional (Thousands)'] = param['flow_need'] / 10**6
    summary_param['Buildings built (Thousands)'] = param['flow_construction'] / 10**3
    summary_param['Buildings demolished (Thousands)'] = param['flow_demolition'] / 10**3
    summary_param['Person by housing'] = param['pop_housing']
    summary_param['Share multi-family (%)'] = param['share_multi_family']

    temp = param['surface'].xs(True, level='Existing', drop_level=True)
    temp.index = temp.index.map(lambda x: 'Surface existing {} - {} (m2/dwelling)'.format(x[0], x[1]))
    summary_param.update(temp.T)

    footprint_built = pd.read_csv(config['footprint']['construction'], index_col=[0])
    carbon_footprint_built = footprint_built.loc[:, 'Carbon content (kgCO2/m2)']
    carbon_footprint_built = config['footprint']['Traditional material'] * carbon_footprint_built[
        'Traditional material'] + config['footprint']['Bio material'] * carbon_footprint_built['Bio material']

    embodied_energy_built = footprint_built.loc[:, 'Grey energy (kWh/m2)']
    embodied_energy_built = config['footprint']['Traditional material'] * embodied_energy_built[
        'Traditional material'] + config['footprint']['Bio material'] * embodied_energy_built['Bio material']

    footprint_renovation = pd.read_csv(config['footprint']['renovation'], index_col=[0, 1])
    carbon_footprint_renovation = footprint_renovation.xs('Carbon content (kgCO2/m2)', level='Content')
    param['carbon_footprint_renovation'] = carbon_footprint_renovation.loc['Traditional material', :] * config['footprint'][
        'Traditional material'] + carbon_footprint_renovation.loc['Bio material', :] * config['footprint'][
                                      'Bio material']
    embodied_energy_renovation = footprint_renovation.xs('Grey energy (kWh/m2)', level='Content')
    param['embodied_energy_renovation'] = embodied_energy_renovation.loc['Traditional material', :] * config['footprint'][
        'Traditional material'] + embodied_energy_renovation.loc['Bio material', :] * config['footprint'][
                                      'Bio material']

    temp = param['surface'].xs(False, level='Existing', drop_level=True)
    temp.index = temp.index.map(lambda x: 'Surface construction {} - {} (m2/dwelling)'.format(x[0], x[1]))
    summary_param.update(temp.T)

    temp = param['surface'].xs(False, level='Existing', drop_level=True)
    temp = (param['flow_built'].groupby(temp.index.names).sum() * temp).sum() / 10**6
    summary_param['Surface construction (Million m2)'] = temp

    summary_param['Carbon footprint construction (MtCO2)'] = (summary_param[
                                                                  'Surface construction (Million m2)'] * carbon_footprint_built) / 10**3
    param['Carbon footprint construction (MtCO2)'] = summary_param['Carbon footprint construction (MtCO2)']

    summary_param['Embodied energy construction (TWh PE)'] = (summary_param[
                                                                  'Surface construction (Million m2)'] * embodied_energy_built) / 10**3
    param['Embodied energy construction (TWh PE)'] = summary_param['Embodied energy construction (TWh PE)']

    param['Surface construction (Million m2)'] = summary_param['Surface construction (Million m2)']

    summary_param = pd.DataFrame(summary_param)
    # summary_param = summary_param.loc[config['start']:, :]
    return param, summary_param
