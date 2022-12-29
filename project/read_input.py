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
from pandas import Series, DataFrame, concat, MultiIndex, Index
from numpy.random import normal

import copy
import os

from project.utils import reindex_mi, get_pandas
from project.dynamic import stock_need, share_multi_family, evolution_surface_built, share_type_built
from project.input.param import generic_input


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
    def __init__(self, name, start, end, value, policy, gest=None, cap=None, target=None, cost_min=None, cost_max=None,
                 new=None, by='index', non_cumulative=None, frequency=None, intensive=None):
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
        self.new = new
        self.by = by
        self.non_cumulative = non_cumulative
        self.frequency = frequency
        self.intensive = intensive

    def cost_targeted(self, cost_insulation, cost_included=None, target_subsidies=None):
        """
        Gives the amount of the cost of a gesture for a segment over which the subvention applies.

        If self.new, cost global is the amount loaned for gestures which are considered as 'global renovations',
        and thus caped by the proper maximum zil amount taking the heater replacement into account.
        Also, cost_no_global are the amount loaned for unique or bunch renovations actions.


        Parameters
        ----------
        cost_insulation: pd.DataFrame
            Cost of an insulation gesture
        target_subsidies: pd.DataFrame
            Boolean values. If self.new it corresponds to the global renovations
        cost_included: pd.DataFrame
            After tax cost of a heater


        Returns
        -------
        cost: pd.DataFrame
            Each cell of the DataFrame corresponds to the cost after subventions of a specific gesture and segment


        """
        cost = cost_insulation.copy()
        idx = pd.IndexSlice
        target = None
        if self.target is not None and target_subsidies is not None:
            n = 'old'
            if self.new:
                n = 'new'
            n = '{}_{}'.format(self.name, n)
            target = target_subsidies[n]
            if not self.new:
                cost = cost[target].fillna(0)

            if self.new and self.name == 'zero_interest_loan':
                target_global = target_subsidies[n]
                cost_global = cost[target_global].fillna(0).copy()
                cost_included = reindex_mi(cost_included, cost_global.index)
                cost_included[cost_included.index.get_level_values("Heater replacement") == False] = 0
                cost_included = pd.concat([cost_included] * cost_global.shape[1], axis=1).set_axis(cost_global.columns, axis=1)
                cost_global[cost_global > 50000 - cost_included] = 50000 - cost_included

                cost_no_global = cost[~target_global].fillna(0).copy()
                # windows specific cap
                cost_no_global[cost_no_global.loc[:, idx[False, False, False, True]] > 7000] = 7000

                one_insulation = [c for c in cost_no_global.columns if (sum(idx[c]) == 1)]
                two_insulation = [c for c in cost_no_global.columns if (sum(idx[c]) == 2)]
                more_insulation = [c for c in cost_no_global.columns if (sum(idx[c]) > 2)]
                no_switch_idx = cost_no_global.xs(False, level='Heater replacement', drop_level=False).index

                cost_no_global[cost_no_global.loc[no_switch_idx, one_insulation] > 15000] = 15000 # count_cap_effect = 400
                cost_no_global[cost_no_global.loc[no_switch_idx, two_insulation] > 25000] = 25000 # count_cap_effect = 270
                cost_no_global[cost_no_global.loc[no_switch_idx, more_insulation] > 30000] = 30000 # count_cap_effect = 320
                cost_no_global[cost_no_global.loc[:, one_insulation] > 25000 - cost_included.loc[:, one_insulation]] = 25000 - cost_included # count_cap_effect = 1306
                cost_no_global[cost_no_global.loc[:, two_insulation] > 30000 - cost_included.loc[:, two_insulation]] = 30000 - cost_included # count_cap_effect = 2954

                cost = cost_global + cost_no_global
                #count_cap_effect = pd.DataFrame([cost_global > 50000 - cost_included][0], index=cost_global.index, columns=cost_global.columns).sum().sum()
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
    """Read initial building stock.

    Parameters
    ----------
    config: dict

    Returns
    -------
    pd.Series
        MultiIndex Series with building stock attributes as levels.
    """

    stock = get_pandas(config['building_stock'], lambda x: pd.read_csv(x, index_col=[0, 1, 2, 3, 4, 5, 6, 7, 8]).squeeze())

    stock = stock.reset_index('Heating system')
    stock['Heating system'] = stock['Heating system'].str.replace('Electricity-Heat pump', 'Electricity-Heat pump water')
    stock = stock.set_index('Heating system', append=True).squeeze()
    year = config['start']

    stock = pd.concat([stock], keys=[True], names=['Existing'])
    idx_names = ['Existing', 'Occupancy status', 'Income owner', 'Income tenant', 'Housing type',
                 'Heating system', 'Wall', 'Floor', 'Roof', 'Windows']

    stock = stock.reorder_levels(idx_names)
    return stock, year


def read_policies(config):
    def read_mpr(data):
        l = list()
        heater = get_pandas(data['heater'],
                            lambda x: pd.read_csv(x, index_col=[0, 1]).squeeze().unstack('Heating system'))
        insulation = get_pandas(data['insulation'],
                                lambda x: pd.read_csv(x, index_col=[0]))

        if data['global_retrofit']:
            if isinstance(data['global_retrofit'], dict):
                global_retrofit = get_pandas(data['global_retrofit']['value'],
                                             lambda x: pd.read_csv(x, index_col=[0]).squeeze())

                l.append(PublicPolicy('mpr', data['global_retrofit']['start'], data['global_retrofit']['end'], global_retrofit, 'subsidy_non_cumulative',
                                      gest='insulation'))
            else:
                global_retrofit = get_pandas(data['global_retrofit'],
                                             lambda x: pd.read_csv(x, index_col=[0]).squeeze())
                l.append(PublicPolicy('mpr', data['start'], data['end'], global_retrofit, 'subsidy_non_cumulative',
                                      gest='insulation'))

        if data['bonus']:
            if isinstance(data['bonus'], dict):
                bonus_best = get_pandas(data['bonus']['value'], lambda x: pd.read_csv(x, index_col=[0]).squeeze())
                bonus_worst = get_pandas(data['bonus']['value'], lambda x: pd.read_csv(x, index_col=[0]).squeeze())

                """bonus_best = pd.read_csv(data['bonus']['value'], index_col=[0]).squeeze('columns')
                bonus_worst = pd.read_csv(data['bonus']['value'], index_col=[0]).squeeze('columns')"""
                l.append(PublicPolicy('mpr', data['bonus']['start'], data['bonus']['end'], bonus_best, 'bonus_best', gest='insulation'))
                l.append(PublicPolicy('mpr', data['bonus']['start'], data['bonus']['end'], bonus_worst, 'bonus_worst', gest='insulation'))
            else:
                bonus_best = get_pandas(data['bonus'], lambda x: pd.read_csv(x, index_col=[0]).squeeze())
                bonus_worst = get_pandas(data['bonus'], lambda x: pd.read_csv(x, index_col=[0]).squeeze())

                """bonus_best = pd.read_csv(data['bonus'], index_col=[0]).squeeze('columns')
                bonus_worst = pd.read_csv(data['bonus'], index_col=[0]).squeeze('columns')"""
                l.append(PublicPolicy('mpr', data['start'], data['end'], bonus_best, 'bonus_best', gest='insulation'))
                l.append(PublicPolicy('mpr', data['start'], data['end'], bonus_worst, 'bonus_worst', gest='insulation'))

        l.append(PublicPolicy('mpr', data['start'], data['end'], heater, 'subsidy_target', gest='heater'))
        l.append(PublicPolicy('mpr', data['start'], data['end'], insulation, 'subsidy_target', gest='insulation'))

        return l

    def read_mpr_serenite(data):
        """Create MPR Serenite PublicPolicy instance.

        MaPrimeRénov' Sérénité (formerly Habiter Mieux Sérénité) for major energy renovation work in your home.
        To do so, your work must result in an energy gain of at least 35%.
        The amount of the bonus varies according to the amount of your resources.

        Parameters
        ----------
        data

        Returns
        -------
        list
        """
        l = list()
        mpr_serenite = get_pandas(data['insulation'],
                                  lambda x: pd.read_csv(x, index_col=[0]).squeeze())

        l.append(PublicPolicy('mpr_serenite', data['start'], data['end'], mpr_serenite, 'subsidy_non_cumulative',
                              gest='insulation', non_cumulative=['mpr', 'cite'], cap=data['cap']))
        return l

    def read_cee(data):
        l = list()
        heater = get_pandas(data['heater'], lambda x: pd.read_csv(x, index_col=[0, 1]).squeeze().unstack('Heating system'))
        insulation = get_pandas(data['insulation'], lambda x: pd.read_csv(x, index_col=[0]))
        tax = get_pandas(data['tax'], lambda x: pd.read_csv(x, index_col=[0]))

        l.append(PublicPolicy('cee', data['start'], data['end'], tax.loc[data['start']:data['end']-1, :], 'tax'))
        l.append(PublicPolicy('cee', data['start'], data['end'], heater, 'subsidy_target', gest='heater'))
        l.append(PublicPolicy('cee', data['start'], data['end'], insulation, 'subsidy_target', gest='insulation'))
        return l

    def read_cap(data):
        cap = get_pandas(data['insulation'], lambda x: pd.read_csv(x, index_col=[0]).squeeze())
        return [PublicPolicy('subsidies_cap', data['start'], data['end'], cap, 'subsidies_cap', gest='insulation')]

    def read_carbon_tax(data):
        tax = get_pandas(data['tax'], lambda x: pd.read_csv(x, index_col=[0]).squeeze())
        emission = get_pandas(data['emission'], lambda x: pd.read_csv(x, index_col=[0]).squeeze())
        tax = (tax * emission).fillna(0) / 10 ** 6
        tax = tax.loc[(tax != 0).any(axis=1)]
        return [PublicPolicy('carbon_tax', data['start'], data['end'], tax.loc[data['start']:data['end']-1, :], 'tax')]

    def read_cite(data):
        """Creates the income tax credit PublicPolicy instance.

        TODO: Cap set to 16,000€. (but seems to be 4,000€) ?
        TODO: Windows should be exempted.

        Oil fuel-Performant Boiler exempted.

        Parameters
        ----------
        data

        Returns
        -------

        """
        l = list()
        heater = get_pandas(data['heater'], lambda x: pd.read_csv(x, index_col=[0]).squeeze())
        l.append(PublicPolicy('cite', data['start'], data['end'], heater, 'subsidy_ad_volarem', gest='heater',
                              cap=data['cap'], by='columns'))
        insulation = get_pandas(data['insulation'], lambda x: pd.read_csv(x, index_col=[0]).squeeze())
        l.append(
            PublicPolicy('cite', data['start'], data['end'], insulation, 'subsidy_ad_volarem', gest='insulation',
                         cap=data['cap']))
        return l

    def read_zil(data):
        """Creates a zero_interest_loan PublicPolicy instance.

        "new" is a specific attribute of zero_interest_loan,
            if it is true the zil will be implemented by gestures and not by epc jumps requirements.
            Some of the gestures available to a segment are qualified by the zil program as 'global renovation'
            and have a higher loan cap (50 000). For a gesture to be a 'global renovation' it must reduce of 35%
            the conventional primary energy need and the resulting building must not be of G or F epc level.
            This is define in define_policy_target (in building.py).
            Other gesture are 'unique or bunch of actions' and have detailed caps:
            - 1 action on window: 7 000
            - 1 other action: 15  000
            - 2 actions: 25 000
            - 3 actions or more: 30 000
            For each of the gestures of a segment we will apply to its cost the proper cap to have the amount loaned,
             and carefully take into account the cost of heating replacement if need.

        "ad_valorem" means the policy will be considered as a subvention in the DCM,
            if it's False, the DCM will have another coefficient of preference associated to a dummy variable zil.

        Parameters
        ----------
        data: dict
            it's the config dictionary.

        Returns
        -------
        PublicPolicy instance with zero_interest_loan attributes
        """
        data_max = get_pandas(data['max'], lambda x: pd.read_csv(x, index_col=[0]).squeeze())
        return [
                PublicPolicy('zero_interest_loan', data['start'], data['end'], data['value'], 'subsidy_ad_volarem',
                             target=True, cost_min=data['min'], cost_max=data_max, gest='insulation', new=data['new'])]

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

    def read_obligation(data):
        l = list()
        banned_performance = get_pandas(data['value'], lambda x: pd.read_csv(x, index_col=[0], header=None).squeeze())
        l.append(PublicPolicy('obligation', data['start'], data['end'], banned_performance, 'obligation',
                              gest='insulation', frequency=data['frequency'], intensive=data['intensive']))
        return l

    def read_landlord(data):
        return [PublicPolicy('landlord', data['start'], data['end'], None, 'regulation', gest='insulation')]

    def read_multi_family(data):
        return [PublicPolicy('multi_family', data['start'], data['end'], None, 'regulation', gest='insulation')]

    read = {'mpr': read_mpr, 'mpr_serenite': read_mpr_serenite, 'cee': read_cee, 'cap': read_cap, 'carbon_tax': read_carbon_tax,
            'cite': read_cite, 'reduced_tax': read_reduced_tax, 'zero_interest_loan': read_zil,
            'sub_ad_volarem': read_ad_volarem, 'oil_fuel_elimination': read_oil_fuel_elimination,
            'obligation': read_obligation, 'landlord': read_landlord, 'multi_family': read_multi_family}

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


def read_prices(config):
    energy_prices = get_pandas(config['energy_prices'], lambda x: pd.read_csv(x, index_col=[0]).rename_axis('Year').rename_axis('Heating energy', axis=1))
    energy_prices = energy_prices.loc[:config['end']]

    return energy_prices


def read_inputs(config, other_inputs=generic_input):
    """Read all inputs in Python object and concatenate in one dict.

    Parameters
    ----------
    config: dict
        Configuration dictionary with path to data.
    other_inputs: dict
        Other inputs that are manually inserted in param.py

    Returns
    -------
    dict
    """

    inputs = dict()
    idx = range(config['start'], config['end'])

    inputs.update(other_inputs)

    cost_heater = get_pandas(config['cost_heater'], lambda x: pd.read_csv(x, index_col=[0]).squeeze().rename(None))
    inputs.update({'cost_heater': cost_heater})

    cost_insulation = get_pandas(config['cost_insulation'], lambda x: pd.read_csv(x, index_col=[0]).squeeze().rename(None))
    inputs.update({'cost_insulation': cost_insulation})

    energy_prices = get_pandas(config['energy_prices'], lambda x: pd.read_csv(x, index_col=[0]).rename_axis('Year').rename_axis('Heating energy', axis=1))
    inputs.update({'energy_prices': energy_prices.loc[:config['end'], :]})

    energy_taxes = get_pandas(config['energy_taxes'], lambda x: pd.read_csv(x, index_col=[0]).rename_axis('Year').rename_axis('Heating energy', axis=1))
    inputs.update({'energy_taxes': energy_taxes.loc[:config['end'], :]})

    efficiency = get_pandas(config['efficiency'], lambda x: pd.read_csv(x, index_col=[0]).squeeze())
    inputs.update({'efficiency': efficiency})

    ms_heater = get_pandas(config['ms_heater'], lambda x: pd.read_csv(x, index_col=[0, 1]))
    ms_heater.columns.set_names('Heating system final', inplace=True)
    inputs.update({'ms_heater': ms_heater})

    df = get_pandas(config['renovation_rate_ini'])
    renovation_rate_ini = df.set_index(list(df.columns[:-1])).squeeze().rename(None).round(decimals=3)
    inputs.update({'renovation_rate_ini': renovation_rate_ini})

    ms_intensive = get_pandas(config['ms_insulation'], lambda x: pd.read_csv(x, index_col=[0, 1, 2, 3]).squeeze().rename(None).round(decimals=3))
    inputs.update({'ms_intensive': ms_intensive})

    population = get_pandas(config['population'], lambda x: pd.read_csv(x, index_col=[0], header=None).squeeze())
    inputs.update({'population': population.loc[:config['end']]})

    inputs.update({'stock_ini': other_inputs['stock_ini']})

    if config['pop_housing'] is None:
        inputs.update({'pop_housing_min': other_inputs['pop_housing_min']})
        inputs.update({'factor_pop_housing': other_inputs['factor_pop_housing']})
    else:
        pop_housing = get_pandas(config['pop_housing'],
                                          lambda x: pd.read_csv(x, index_col=[0], header=None).squeeze())
        inputs.update({'pop_housing': pop_housing.loc[:config['end']]})

    if config['share_multi_family'] is None:
        inputs.update({'factor_multi_family': other_inputs['factor_multi_family']})
    else:
        share_multi_family = get_pandas(config['share_multi_family'],
                                                 lambda x: pd.read_csv(x, index_col=[0], header=None).squeeze())
        inputs.update({'share_multi_family': share_multi_family})

    inputs.update({'available_income': other_inputs['available_income']})
    inputs.update({'income_rate': config['income_rate']})
    inputs.update({'demolition_rate': config['demolition_rate']})

    if config['surface_built'] is None:
        inputs.update({'surface': other_inputs['surface']})
        inputs.update({'surface_max': other_inputs['surface_max']})
        inputs.update({'surface_elasticity': other_inputs['surface_elasticity']})
    else:
        surface_built = get_pandas(config['surface_built'], lambda x: pd.read_csv(x, index_col=[0]).squeeze().rename(None))
        inputs.update({'surface_built': surface_built})

    ms_heater_built = get_pandas(config['ms_heater_built'], lambda x: pd.read_csv(x, index_col=[0], header=[0]))
    ms_heater_built.columns.set_names(['Heating system'], inplace=True)
    ms_heater_built.index.set_names(['Housing type'], inplace=True)
    inputs.update({'ms_heater_built': ms_heater_built})

    inputs.update({'performance_insulation': other_inputs['performance_insulation']})

    df = get_pandas(config['health_cost'], lambda x: pd.read_csv(x, index_col=[0, 1]))
    inputs.update({'health_cost': df})

    carbon_value = get_pandas(config['carbon_value'], lambda x: pd.read_csv(x, index_col=[0]).squeeze())
    inputs.update({'carbon_value': carbon_value.loc[idx]})

    carbon_emission = get_pandas(config['carbon_emission'], lambda x: pd.read_csv(x, index_col=[0]).rename_axis('Year'))
    inputs.update({'carbon_emission': carbon_emission.loc[idx, :]})

    footprint_built = get_pandas(config['footprint']['construction'], lambda x: pd.read_csv(x, index_col=[0]))
    inputs.update({'footprint_built': footprint_built})
    footprint_renovation = get_pandas(config['footprint']['renovation'], lambda x: pd.read_csv(x, index_col=[0, 1]))
    inputs.update({'footprint_renovation': footprint_renovation})

    inputs.update({'traditional_material': config['footprint']['Traditional material']})
    inputs.update({'bio_material': config['footprint']['Bio material']})

    """levels_category = ['Housing type', 'Occupancy status', 'Income tenant', 'Income owner', 'Heating system']
    for key, item in inputs.items():
        if isinstance(item, (Series, DataFrame)):
            level = [i for i in item.index.names if i in levels_category]
            inputs[key] = item.reset_index(level).astype({i: 'category' for i in level}).set_index(
                level, append=True).squeeze()"""
    return inputs


def parse_inputs(inputs, taxes, config, stock):
    """Macro module : run exogenous dynamic parameters.

    Parameters
    ----------
    inputs: dict
        Raw inputs read as Python object.
    taxes: list
    config: dict
        Configuration file.
    stock: Series
        Building stock.

    Returns
    -------
    dict
        Parsed input
    """

    idx = range(config['start'], config['end'])

    parsed_inputs = copy.deepcopy(inputs)

    cost_factor = 1
    if 'cost_factor' in config.keys():
        cost_factor = config['cost_factor']

    prices_factor = 1
    if 'prices_factor' in config.keys():
        prices_factor = config['prices_factor']

    parsed_inputs['cost_heater'] *= cost_factor
    parsed_inputs['cost_insulation'] *= cost_factor
    parsed_inputs['energy_prices'].loc[range(config['start'] + 2, config['end']), :] *= prices_factor
    parsed_inputs['energy_taxes'].loc[range(config['start'] + 2, config['end']), :] *= prices_factor

    parsed_inputs['population_total'] = inputs['population']
    parsed_inputs['sizing_factor'] = stock.sum() / inputs['stock_ini']
    parsed_inputs['population'] = inputs['population'] * parsed_inputs['sizing_factor']

    if 'pop_housing' in parsed_inputs.keys():
        parsed_inputs['stock_need'] = parsed_inputs['population'] / parsed_inputs['pop_housing']
    else:
        parsed_inputs['stock_need'], parsed_inputs['pop_housing'] = stock_need(parsed_inputs['population'],
                                                                               parsed_inputs['population'][
                                                                                   config['start']] / stock.sum(),
                                                                               inputs['pop_housing_min'],
                                                                               config['start'],
                                                                               inputs['factor_pop_housing'])
    if 'share_multi_family' not in inputs.keys():
        parsed_inputs['share_multi_family'] = share_multi_family(parsed_inputs['stock_need'],
                                                                 inputs['factor_multi_family'])

    parsed_inputs['available_income'] = pd.Series(
        [inputs['available_income'] * (1 + config['income_rate']) ** (i - idx[0]) for i in idx], index=idx)

    parsed_inputs['available_income_pop'] = (parsed_inputs['available_income'] / parsed_inputs['population_total']).dropna()

    parsed_inputs['flow_demolition'] = pd.Series(inputs['demolition_rate'] * stock.sum(), index=idx[1:])
    parsed_inputs['flow_need'] = parsed_inputs['stock_need'] - parsed_inputs['stock_need'].shift(1)
    parsed_inputs['flow_construction'] = parsed_inputs['flow_need'] + parsed_inputs['flow_demolition']

    if 'surface_built' in inputs.keys():
        surface_built = inputs['surface_built']
    else:
        surface_built = evolution_surface_built(inputs['surface'].xs(False, level='Existing'), inputs['surface_max'],
                                                inputs['surface_elasticity'], parsed_inputs['available_income_pop'])

    surface_existing = pd.concat([parsed_inputs['surface'].xs(True, level='Existing')] * surface_built.shape[1], axis=1,
                                 keys=surface_built.columns)
    parsed_inputs['surface'] = pd.concat((surface_existing, surface_built), axis=0, keys=[True, False], names=['Existing'])

    type_built = share_type_built(parsed_inputs['stock_need'], parsed_inputs['share_multi_family'],
                                  parsed_inputs['flow_construction']) * parsed_inputs['flow_construction']

    share_decision_maker = stock.groupby(
        ['Occupancy status', 'Housing type', 'Income owner', 'Income tenant']).sum().unstack(
        ['Occupancy status', 'Income owner', 'Income tenant'])
    share_decision_maker = (share_decision_maker.T / share_decision_maker.sum(axis=1)).T
    share_decision_maker = pd.concat([share_decision_maker] * type_built.shape[1], keys=type_built.columns, axis=1)
    construction = (reindex_mi(type_built, share_decision_maker.columns, axis=1) * share_decision_maker).stack(
        ['Occupancy status', 'Income owner', 'Income tenant']).fillna(0)

    ms_heater_built = reindex_mi(inputs['ms_heater_built'], construction.index).stack()
    construction = (reindex_mi(construction, ms_heater_built.index).T * ms_heater_built).T
    construction = construction.loc[(construction != 0).any(axis=1)]

    performance_insulation = pd.concat([pd.Series(inputs['performance_insulation'])] * construction.shape[0], axis=1,
                                       keys=construction.index).T

    parsed_inputs['flow_built'] = pd.concat((construction, performance_insulation), axis=1).set_index(
        list(performance_insulation.keys()), append=True)

    parsed_inputs['flow_built'] = pd.concat([parsed_inputs['flow_built']], keys=[False],
                                            names=['Existing']).reorder_levels(stock.index.names)

    if not config['construction']:
        parsed_inputs['flow_built'][parsed_inputs['flow_built'] > 0] = 0

    df = inputs['health_cost']
    parsed_inputs['health_expenditure'] = df['Health expenditure']
    parsed_inputs['mortality_cost'] = df['Social cost of mortality']
    parsed_inputs['loss_well_being'] = df['Loss of well-being']
    parsed_inputs['carbon_value_kwh'] = (parsed_inputs['carbon_value'] * parsed_inputs['carbon_emission'].T).T.dropna() / 10**6

    carbon_footprint_built = inputs['footprint_built'].loc[:, 'Carbon content (kgCO2/m2)']
    carbon_footprint_built = inputs['traditional_material'] * carbon_footprint_built[
        'Traditional material'] + inputs['bio_material'] * carbon_footprint_built['Bio material']
    parsed_inputs['carbon_footprint_built'] = carbon_footprint_built

    embodied_energy_built = inputs['footprint_built'].loc[:, 'Grey energy (kWh/m2)']
    embodied_energy_built = inputs['traditional_material'] * embodied_energy_built[
        'Traditional material'] + inputs['bio_material'] * embodied_energy_built['Bio material']
    parsed_inputs['embodied_energy_built'] = embodied_energy_built

    carbon_footprint_renovation = inputs['footprint_renovation'].xs('Carbon content (kgCO2/m2)', level='Content')
    parsed_inputs['carbon_footprint_renovation'] = carbon_footprint_renovation.loc['Traditional material', :] * inputs[
        'traditional_material'] + carbon_footprint_renovation.loc['Bio material', :] * inputs['bio_material']
    embodied_energy_renovation = inputs['footprint_renovation'].xs('Grey energy (kWh/m2)', level='Content')
    parsed_inputs['embodied_energy_renovation'] = embodied_energy_renovation.loc['Traditional material', :] * inputs[
        'traditional_material'] + embodied_energy_renovation.loc['Bio material', :] * inputs['bio_material']

    temp = parsed_inputs['surface'].xs(False, level='Existing', drop_level=True)
    temp = (parsed_inputs['flow_built'].groupby(temp.index.names).sum() * temp).sum() / 10**6
    parsed_inputs['Surface construction (Million m2)'] = temp

    parsed_inputs['Carbon footprint construction (MtCO2)'] = (parsed_inputs['Surface construction (Million m2)'] * parsed_inputs['carbon_footprint_built']) / 10**3
    parsed_inputs['Embodied energy construction (TWh PE)'] = (parsed_inputs['Surface construction (Million m2)'] * parsed_inputs['embodied_energy_built']) / 10**3

    energy_prices = parsed_inputs['energy_prices'].copy()
    energy_taxes = parsed_inputs['energy_taxes'].copy()

    if config['prices_constant']:
        energy_prices = pd.concat([energy_prices.loc[config['start'], :]] * energy_prices.shape[0], keys=energy_prices.index,
                                  axis=1).T

    total_taxes = pd.DataFrame(0, index=energy_prices.index, columns=energy_prices.columns)
    for t in taxes:
        total_taxes = total_taxes.add(t.value, fill_value=0)

    if energy_taxes is not None:
        total_taxes = total_taxes.add(energy_taxes, fill_value=0)
        taxes += [PublicPolicy('energy_taxes', energy_taxes.index[0], energy_taxes.index[-1], energy_taxes, 'tax')]

    if config['taxes_constant']:
        total_taxes = pd.concat([total_taxes.loc[config['start'], :]] * total_taxes.shape[0], keys=total_taxes.index,
                                axis=1).T

    energy_vta = energy_prices * inputs['vta_energy_prices']
    taxes += [PublicPolicy('energy_vta', energy_vta.index[0], energy_vta.index[-1], energy_vta, 'tax')]
    total_taxes += energy_vta
    parsed_inputs['taxes'] = taxes
    parsed_inputs['total_taxes'] = total_taxes

    energy_prices = energy_prices.add(total_taxes, fill_value=0)
    parsed_inputs['energy_prices'] = energy_prices

    if config.get('remove_market_failures'):
        parsed_inputs['remove_market_failures'] = config['remove_market_failures']
    else:
        parsed_inputs['remove_market_failures'] = None

    return parsed_inputs


def dump_inputs(parsed_inputs, path):
    """Create summary input DataFrame.

    Parameters
    ----------
    parsed_inputs: dict

    Returns
    -------
    DataFrame
    """
    
    summary_input = dict()
    summary_input['Sizing factor (%)'] = pd.Series(parsed_inputs['sizing_factor'], index=parsed_inputs['population'].index)
    summary_input['Total population (Millions)'] = parsed_inputs['population'] / 10**6
    summary_input['Income (Billions euro)'] = parsed_inputs['available_income'] * parsed_inputs['sizing_factor'] / 10**9
    summary_input['Buildings stock (Millions)'] = parsed_inputs['stock_need'] / 10**6
    summary_input['Buildings additional (Thousands)'] = parsed_inputs['flow_need'] / 10**3
    summary_input['Buildings built (Thousands)'] = parsed_inputs['flow_construction'] / 10**3
    summary_input['Buildings demolished (Thousands)'] = parsed_inputs['flow_demolition'] / 10**3
    summary_input['Person by housing'] = parsed_inputs['pop_housing']
    summary_input['Share multi-family (%)'] = parsed_inputs['share_multi_family']

    temp = parsed_inputs['surface'].xs(True, level='Existing', drop_level=True)
    temp.index = temp.index.map(lambda x: 'Surface existing {} - {} (m2/dwelling)'.format(x[0], x[1]))
    summary_input.update(temp.T)

    temp = parsed_inputs['surface'].xs(False, level='Existing', drop_level=True)
    temp.index = temp.index.map(lambda x: 'Surface construction {} - {} (m2/dwelling)'.format(x[0], x[1]))
    summary_input.update(temp.T)

    summary_input['Surface construction (Million m2)'] = parsed_inputs['Surface construction (Million m2)']
    summary_input['Carbon footprint construction (MtCO2)'] = parsed_inputs['Carbon footprint construction (MtCO2)']
    summary_input['Embodied energy construction (TWh PE)'] = parsed_inputs['Embodied energy construction (TWh PE)']

    summary_input = pd.DataFrame(summary_input)

    t = parsed_inputs['total_taxes'].copy()
    t.columns = t.columns.map(lambda x: 'Taxes {} (euro/kWh)'.format(x))
    temp = parsed_inputs['energy_prices'].copy()
    temp.columns = temp.columns.map(lambda x: 'Prices {} (euro/kWh)'.format(x))
    pd.concat((summary_input, t, temp), axis=1).to_csv(os.path.join(path, 'input.csv'))

    return summary_input


def dict2data_inputs(inputs):
    """Grouped all inputs in the same DataFrame.

    Process is useful to implement a global sensitivity analysis.

    Returns
    -------
    DataFrame
    """

    data = DataFrame(columns=['variables', 'index', 'value'])
    metadata = DataFrame(columns=['variables', 'type', 'name', 'index', 'columns'])
    for key, item in inputs.items():
        i = True

        if isinstance(item, dict):
            metadata = concat((metadata.T, Series({'variables': key, 'type': type(item).__name__})), axis=1).T
            i = False
            item = Series(item)

        if isinstance(item, (float, int)):
            data = concat((data.T, Series({'variables': key, 'value': item})), axis=1).T
            metadata = concat((metadata.T, Series({'variables': key, 'type': type(item).__name__})), axis=1).T

        if isinstance(item, DataFrame):
            metadata = concat((metadata.T, Series({'variables': key, 'type': type(item).__name__,
                                                   'index': item.index.names.copy(),
                                                   'columns': item.columns.names.copy()})), axis=1).T
            i = False
            item = item.stack(item.columns.names)

        if isinstance(item, Series):
            if i:
                metadata = concat((metadata.T, Series({'variables': key, 'type': type(item).__name__, 'name': item.name,
                                                       'index': item.index.names.copy()})), axis=1).T

            if isinstance(item.index, MultiIndex):
                item.index = item.index.to_flat_index()

            item.index = item.index.rename('index')
            df = concat([item.rename('value').reset_index()], keys=[key], names=['variables']).reset_index('variables')
            data = concat((data, df), axis=0)

    data = data.astype({'variables': 'string', 'value': 'float64'})
    data.reset_index(drop=True, inplace=True)
    return data


def data2dict_inputs(data, metadata):
    """Parse aggregate data pandas and return dict fill with several inputs.

    Parameters
    ----------
    data: DataFrame
        Model data input.
    metadata: DataFrame
        Additional information to find out how to parse data.

    Returns
    -------
    dict
    """

    def parse_index(n, index_values):
        if len(n) == 1:
            idx = Index(index_values, name=n[0])
        else:
            idx = MultiIndex.from_tuples(index_values)
            idx.names = n
        return idx

    parsed_input = dict()
    for variables, df in data.groupby('variables'):
        meta = metadata[metadata['variables'] == variables]
        if meta['type'].iloc[0] == 'int':
            parsed_input.update({variables: int(df['value'].iloc[0])})
        elif meta['type'].iloc[0] == 'float':
            parsed_input.update({variables: float(df['value'].iloc[0])})
        elif meta['type'].iloc[0] == 'Series':
            idx = parse_index(meta['index'].iloc[0], df['index'].values)
            parsed_input.update({variables: Series(df['value'].values, name=str(meta['name'].iloc[0]), index=idx)})
        elif meta['type'].iloc[0] == 'DataFrame':
            idx = parse_index(meta['index'].iloc[0] + meta['columns'].iloc[0], df['index'].values)
            parsed_input.update({variables: Series(df['value'].values, name=str(meta['name'].iloc[0]), index=idx).unstack(
                meta['columns'].iloc[0])})

        elif meta['type'].iloc[0] == 'dict':
            parsed_input.update({variables: Series(df['value'].values, index=df['index'].values).to_dict()})

    return parsed_input


def generate_price_scenarios(energy_prices, year_2=2020, year_1=2019, year_0=2018, nb_draws=3, path=None):
    """Generate energy prices scenarios.

    Les prix suivent un processus auto regressif
    $$P_t= λ_1 . P{t-1} + λ_2 . P_{t-2} + α + ϵ_t$$

    On calibre en fixant :
        - P{t-1}= prix de 2019
        - P_{t-2}= prix de 2018
        - Plusieurs combinaisons sont fixés pour λ_1 et λ_2 en croisant λ_1∈[0.6,0.65,0.7,0.75],
         et λ_1+λ_2∈[0.85,0.9,0.95,0.97] soit 16 combinaisons.
        - α est fixé de sorte que $P_{2020} - λ_1 . P{2019} + λ_2 . P_{2018} =α$
        - Enfin ϵ_t∼N(0,P_2012⁄10). Pour chaque combinaisons 10 tirages sont faites soit 160 simulations au total.

    Le premier prix que l’on cherchera à simuler sera le prix de 2013.
    On fait cette opération pour l’électricité, le gaz, le fioul et le bois.

    Parameters
    ----------
    energy_prices: DataFrame
        Energy prices. Year as index, and energy as columns.
    year_2: int
    year_1: int
    year_0: int
        First year of iteration.
    nb_draws: int, default 10

    Returns
    -------

    """

    # lambda_1_values = [0.6, 0.65, 0.7, 0.75]
    # lambda_sum_values = [0.85, 0.9, 0.95, 0.97]
    lambda_1_values = [0.7]
    lambda_sum_values = [0.97]

    result = dict()
    prices = dict()
    prices.update({year_0: concat([energy_prices.loc[year_0, :]] * nb_draws, axis=1).set_axis(range(0, nb_draws), axis=1),
                   year_1: concat([energy_prices.loc[year_1, :]] * nb_draws, axis=1).set_axis(range(0, nb_draws), axis=1)
                   })
    for lambda_1 in lambda_1_values:
        lambda_2_values = [i - lambda_1 for i in lambda_sum_values]
        for lambda_2 in lambda_2_values:
            alpha = energy_prices.loc[year_2, :] - lambda_1 * energy_prices.loc[year_1, :] - lambda_2 * energy_prices.loc[year_0, :]
            epsilon = concat([Series(normal(loc=0, scale=energy_prices.loc[year_2, :] / 10), index=energy_prices.columns) for _ in range(nb_draws)], axis=1)
            for year in range(year_2, energy_prices.index.max() + 1):
                prices[year] = ((epsilon+ lambda_1 * prices[year - 1] + lambda_2 * prices[year - 2]).T + alpha).T
            for i in range(0, nb_draws):
                n = '{}_{}_{}'.format(round(lambda_1, 1), round(lambda_2, 1), i)
                df = concat([prices[year].loc[:, i].rename(year) for year in prices.keys()], axis=1).T
                result.update({n: df})

    result = {k: df for k, df in result.items() if (df > 0).all().all()}
    if path is not None:
        for name, df in result.items():
            df.to_csv(os.path.join(path, 'energy_prices_{}.csv'.format(name)))
        return {name: 'energy_prices_{}.csv'.format(name) for name in result.keys()}
    else:
        return result

