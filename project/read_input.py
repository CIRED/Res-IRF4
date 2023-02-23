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
from numpy.testing import assert_almost_equal

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
    value: float
    policy : {'energy_taxes', 'subsidies'}

    """
    def __init__(self, name, start, end, value, policy, gest=None, cap=None, target=None, cost_min=None, cost_max=None,
                 new=None, by='index', non_cumulative=None, frequency=None, intensive=None, min_performance=None,
                 bonus=False, social_housing=False):
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
        self.min_performance = min_performance
        self.bonus = bonus
        self.social_housing = social_housing

    def cost_targeted(self, cost_insulation, target_subsidies=None):
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


        Returns
        -------
        cost: pd.DataFrame
            Each cell of the DataFrame corresponds to the cost after subventions of a specific gesture and segment
        """
        cost = cost_insulation.copy()
        if self.target is not None and target_subsidies is not None:
            cost = cost[target_subsidies.astype(bool)].fillna(0)
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

    stock = get_pandas(config['building_stock'], lambda x: pd.read_csv(x, index_col=[0, 1, 2, 3, 4, 5, 6, 7, 8]).squeeze()).rename('Stock buildings')
    stock_sum = stock.sum()

    stock = stock.reset_index('Heating system')

    # specify heat-pump
    repartition = 0.8
    idx = stock['Heating system'] == 'Electricity-Heat pump'
    hp_water = stock.loc[idx, :].copy()
    hp_water['Stock buildings'] *= repartition
    hp_water['Heating system'] = hp_water['Heating system'].str.replace('Electricity-Heat pump', 'Electricity-Heat pump water')

    hp_air = stock.loc[idx, :].copy()
    hp_air['Stock buildings'] *= (1 - repartition)
    hp_air['Heating system'] = hp_air['Heating system'].str.replace('Electricity-Heat pump', 'Electricity-Heat pump air')

    stock = stock.loc[~idx, :]
    stock = pd.concat((stock, hp_water, hp_air), axis=0)

    multi_family = stock.index.get_level_values('Housing type') == 'Multi-family'
    oil_fuel = stock['Heating system'].isin(['Oil fuel-Performance boiler', 'Oil fuel-Standard boiler'])
    stock.loc[multi_family & oil_fuel, 'Heating system'] = 'Oil fuel-Collective boiler'

    repartition = 0.5
    idx_gas = stock['Heating system'].isin(['Natural gas-Performance boiler', 'Natural gas-Standard boiler']) & multi_family
    collective_gas = stock.loc[idx_gas, :].copy()
    collective_gas['Stock buildings'] *= repartition
    collective_gas['Heating system'] = collective_gas['Heating system'].str.replace('Natural gas-Performance boiler', 'Natural gas-Collective boiler')
    collective_gas['Heating system'] = collective_gas['Heating system'].str.replace('Natural gas-Standard boiler', 'Natural gas-Collective boiler')
    individual_gas = stock.loc[idx_gas, :].copy()
    individual_gas['Stock buildings'] *= (1 - repartition)
    stock = stock.loc[~idx_gas, :]
    stock = pd.concat((stock, collective_gas, individual_gas), axis=0)

    stock = stock.set_index('Heating system', append=True).squeeze()
    stock = stock.groupby(stock.index.names).sum()
    stock = pd.concat([stock], keys=[True], names=['Existing'])
    idx_names = ['Existing', 'Occupancy status', 'Income owner', 'Income tenant', 'Housing type',
                 'Heating system', 'Wall', 'Floor', 'Roof', 'Windows']

    stock = stock.reorder_levels(idx_names)
    assert_almost_equal(stock.sum(), stock_sum)

    return stock


def read_policies(config):
    def read_mpr(data):
        l = list()
        heater = get_pandas(data['heater'],
                            lambda x: pd.read_csv(x, index_col=[0, 1]).squeeze().unstack('Heating system'))
        if data.get('growth_heater'):
            growth_heater = get_pandas(data['growth_heater'], lambda x: pd.read_csv(x, index_col=[0], header=None).squeeze())
            heater = {k: i * heater for k, i in growth_heater.items()}

        insulation = get_pandas(data['insulation'], lambda x: pd.read_csv(x, index_col=[0, 1]))
        if data.get('growth_insulation'):
            growth_insulation = get_pandas(data['growth_insulation'], lambda x: pd.read_csv(x, index_col=[0], header=None).squeeze())
            insulation = {k: i * insulation for k, i in growth_insulation.items()}

        if data.get('global_renovation'):
            global_renovation = get_pandas(data['global_renovation'],
                                           lambda x: pd.read_csv(x, index_col=[0]).squeeze())
            l.append(PublicPolicy('mpr', data['start'], data['end'], global_renovation, 'subsidy_target',
                                  gest='insulation', target='global_renovation'))

        if data['bonus']:
            bonus_best = get_pandas(data['bonus'], lambda x: pd.read_csv(x, index_col=[0]).squeeze())
            bonus_worst = get_pandas(data['bonus'], lambda x: pd.read_csv(x, index_col=[0]).squeeze())

            l.append(PublicPolicy('mpr', data['start'], data['end'], bonus_best, 'bonus', gest='insulation',
                                  target='bonus_best'))
            l.append(PublicPolicy('mpr', data['start'], data['end'], bonus_worst, 'bonus', gest='insulation',
                                  target='bonus_worst'))

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

        value = get_pandas(data['insulation'], lambda x: pd.read_csv(x, index_col=[0, 1]).squeeze())
        cap = get_pandas(data['cap'], lambda x: pd.read_csv(x, index_col=[0, 1]).squeeze())

        if data.get('growth_insulation'):
            growth_insulation = get_pandas(data['growth_insulation'], lambda x: pd.read_csv(x, index_col=[0], header=None).squeeze())
            value = {k: i * value for k, i in growth_insulation.items()}
            cap = {k: i * cap for k, i in growth_insulation.items()}

        l.append(PublicPolicy(data['name'], data['start'], data['end'], value, data['policy'],
                              target=data.get('target'), gest='insulation', non_cumulative=data.get('non_cumulative'),
                              cap=cap))
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
        return [PublicPolicy('subsidies_cap', data['start'], data['end'], cap, 'subsidies_cap', gest='insulation',
                             target=data.get('target'))]

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
        l.append(PublicPolicy('cite', data['start'], data['end'], heater, 'subsidy_ad_valorem', gest='heater',
                              cap=data['cap'], by='columns'))
        insulation = get_pandas(data['insulation'], lambda x: pd.read_csv(x, index_col=[0]).squeeze())
        l.append(
            PublicPolicy('cite', data['start'], data['end'], insulation, 'subsidy_ad_valorem', gest='insulation',
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
                PublicPolicy('zero_interest_loan', data['start'], data['end'], data['value'], 'subsidy_ad_valorem',
                             target=True, cost_min=data['min'], cost_max=data_max, gest='insulation', new=data['new'])]

    def read_reduced_vta(data):
        l = list()
        l.append(PublicPolicy('reduced_vta', data['start'], data['end'], data['value'], 'reduced_vta', gest='heater',
                              social_housing=True))
        l.append(
            PublicPolicy('reduced_vta', data['start'], data['end'], data['value'], 'reduced_vta', gest='insulation',
                         social_housing=True))
        return l

    def read_ad_valorem(data):
        l = list()
        value = data['value']
        if data.get('index') is not None:
            mask = get_pandas(data['index'], lambda x: pd.read_csv(x, index_col=[0]).squeeze())
            value *= mask

        by = 'index'
        if data.get('columns') is not None:
            mask = get_pandas(data['columns'], lambda x: pd.read_csv(x, index_col=[0]).squeeze())
            value *= mask
            by = 'columns'

        name = 'sub_ad_valorem'
        if data.get('name') is not None:
            name = data['name']

        l.append(PublicPolicy(name, data['start'], data['end'], value, 'subsidy_ad_valorem',
                              gest=data['gest'], by=by, target=data.get('target')))
        return l

    def restriction_energy(data):
        # value = pd.Series(data['value']).rename_axis('Energy')
        return [PublicPolicy(data['name'], data['start'], data['end'], data['value'],
                             'restriction_energy', gest='heater', target=data.get('target'))]

    def restriction_heater(data):
        # value = pd.Series(data['value']).rename_axis('Heating system')
        return [PublicPolicy(data['name'], data['start'], data['end'], data['value'],
                             'restriction_heater', gest='heater', target=data.get('target'))]

    def premature_heater(data):
        return [PublicPolicy(data['name'], data['start'], data['end'], data['value'],
                             'premature_heater', gest='heater', target=data.get('target'))]

    def read_obligation(data):
        l = list()
        banned_performance = get_pandas(data['value'], lambda x: pd.read_csv(x, index_col=[0], header=None).squeeze()).dropna()
        start = min(banned_performance.index)
        if data['start'] > start:
            start = data['start']
        frequency = data['frequency']
        if frequency is not None:
            frequency = pd.Series(frequency["value"], index=pd.Index(frequency["index"], name=frequency["name"]))

        l.append(PublicPolicy(data['name'], start, data['end'], banned_performance, 'obligation',
                              gest='insulation', frequency=frequency, intensive=data['intensive'],
                              min_performance=data['minimum_performance']))

        if data.get('sub_obligation') is not None:
            value = get_pandas(data['sub_obligation'], lambda x: pd.read_csv(x, index_col=[0]).squeeze())
            l.append(PublicPolicy('sub_obligation', start, data['end'], value, 'subsidy_ad_valorem',
                                  gest='insulation'))
        return l

    def read_landlord(data):
        return [PublicPolicy('landlord', data['start'], data['end'], None, 'regulation', gest='insulation')]

    def read_multi_family(data):
        return [PublicPolicy('multi_family', data['start'], data['end'], None, 'regulation', gest='insulation')]

    read = {'mpr': read_mpr,
            'mpr_serenite_high_income': read_mpr_serenite,
            'mpr_serenite_low_income': read_mpr_serenite,
            'mpr_multifamily': read_mpr_serenite,
            'cee': read_cee, 'cap': read_cap, 'carbon_tax': read_carbon_tax,
            'cite': read_cite, 'reduced_vta': read_reduced_vta, 'zero_interest_loan': read_zil,
            'landlord': read_landlord, 'multi_family': read_multi_family}

    list_policies = list()
    for key, item in config['policies'].items():
        item['name'] = key
        if key in read.keys():
            list_policies += read[key](item)
        else:
            if item.get('policy') == 'subsidy_ad_valorem':
                list_policies += read_ad_valorem(item)
            elif item.get('policy') == 'premature_heater':
                list_policies += premature_heater(item)
            elif item.get('policy') == 'restriction_energy':
                list_policies += restriction_energy(item)
            elif item.get('policy') == 'restriction_heater':
                list_policies += restriction_heater(item)
            elif item.get('policy') == 'obligation':
                list_policies += read_obligation(item)
            else:
                print('{} reading function is not implemented'.format(key))

    policies_heater = [p for p in list_policies if p.gest == 'heater']
    policies_insulation = [p for p in list_policies if p.gest == 'insulation']
    taxes = [p for p in list_policies if p.policy == 'tax']

    return policies_heater, policies_insulation, taxes


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
    energy_prices = get_pandas(config['macro']['energy_prices'], lambda x: pd.read_csv(x, index_col=[0]).rename_axis('Year').rename_axis('Heating energy', axis=1))
    inputs.update({'energy_prices': energy_prices.loc[:config['end'], :]})

    energy_taxes = get_pandas(config['macro']['energy_taxes'], lambda x: pd.read_csv(x, index_col=[0]).rename_axis('Year').rename_axis('Heating energy', axis=1))
    inputs.update({'energy_taxes': energy_taxes.loc[:config['end'], :]})

    cost_heater = get_pandas(config['technical']['cost_heater'], lambda x: pd.read_csv(x, index_col=[0]).squeeze().rename(None))
    inputs.update({'cost_heater': cost_heater})

    cost_insulation = get_pandas(config['technical']['cost_insulation'], lambda x: pd.read_csv(x, index_col=[0]).squeeze().rename(None))
    inputs.update({'cost_insulation': cost_insulation})

    efficiency = get_pandas(config['technical']['efficiency'], lambda x: pd.read_csv(x, index_col=[0], header=None).squeeze())
    inputs.update({'efficiency': efficiency})

    lifetime_heater = get_pandas(config['technical']['lifetime_heater'], lambda x: pd.read_csv(x, index_col=[0]).squeeze()).rename(None)
    inputs.update({'lifetime_heater': lifetime_heater})

    ms_heater = get_pandas(config['ms_heater'], lambda x: pd.read_csv(x, index_col=[0, 1]))
    ms_heater.columns.set_names('Heating system final', inplace=True)
    inputs.update({'ms_heater': ms_heater})

    if 'district_heating' in config.keys():
        district_heating = get_pandas(config['district_heating'], lambda x: pd.read_csv(x, index_col=[0, 1]).squeeze())
        inputs.update({'district_heating': district_heating})

    calibration_renovation = None
    if config['renovation']['endogenous']:
        df = get_pandas(config['renovation']['renovation_rate_ini'])
        renovation_rate_ini = df.set_index(list(df.columns[:-1])).squeeze().rename(None).round(decimals=3)
        scale_calibration = config['renovation']['scale']
        calibration_renovation = {'renovation_rate_ini': renovation_rate_ini, 'scale': scale_calibration,
                                  'threshold_indicator': config['renovation'].get('threshold')}
    inputs.update({'calibration_renovation': calibration_renovation})

    calibration_intensive = None
    if config['ms_insulation']['endogenous']:
        ms_insulation_ini = get_pandas(config['ms_insulation']['ms_insulation_ini'], lambda x: pd.read_csv(x, index_col=[0, 1, 2, 3]).squeeze().rename(None).round(decimals=3))
        minimum_performance = config['ms_insulation']['minimum_performance']
        calibration_intensive = {'ms_insulation_ini': ms_insulation_ini, 'minimum_performance': minimum_performance}
    inputs.update({'calibration_intensive': calibration_intensive})

    population = get_pandas(config['macro']['population'], lambda x: pd.read_csv(x, index_col=[0], header=None).squeeze())
    inputs.update({'population': population.loc[:config['end']]})

    inputs.update({'stock_ini': config['macro']['stock_ini']})

    if config['macro']['pop_housing'] is None:
        inputs.update({'pop_housing_min': other_inputs['pop_housing_min']})
        inputs.update({'factor_pop_housing': other_inputs['factor_pop_housing']})
    else:
        pop_housing = get_pandas(config['macro']['pop_housing'],
                                          lambda x: pd.read_csv(x, index_col=[0], header=None).squeeze())
        inputs.update({'pop_housing': pop_housing.loc[:config['end']]})

    if config['macro']['share_single_family_construction'] is not None:
        temp = get_pandas(config['macro']['share_single_family_construction'],
                          lambda x: pd.read_csv(x, index_col=[0], header=None).squeeze())
        inputs.update({'share_single_family_construction': temp})

    else:
        if config['macro']['share_multi_family'] is None:
            inputs.update({'factor_multi_family': other_inputs['factor_multi_family']})
        else:
            _share_multi_family = get_pandas(config['macro']['share_multi_family'],
                                                     lambda x: pd.read_csv(x, index_col=[0], header=None).squeeze())
            inputs.update({'share_multi_family': _share_multi_family})

    inputs.update({'available_income': config['macro']['available_income']})
    inputs.update({'income_rate': config['macro']['income_rate']})
    income = get_pandas(config['macro']['income'], lambda x: pd.read_csv(x, index_col=[0])).squeeze().rename_axis('Income').rename(None)
    inputs.update({'income': income})

    if isinstance(config['macro']['demolition_rate'], (float, int)):
        demolition_rate = config['macro']['demolition_rate']
    else:
        demolition_rate = get_pandas(config['macro']['demolition_rate'],
                                     lambda x: pd.read_csv(x, index_col=[0], header=None)).squeeze().rename(None)

    inputs.update({'demolition_rate': demolition_rate})
    rotation_rate = get_pandas(config['macro']['rotation_rate'], lambda x: pd.read_csv(x, index_col=[0])).squeeze().rename(None)
    inputs.update({'rotation_rate': rotation_rate})

    surface = get_pandas(config['technical']['surface'], lambda x: pd.read_csv(x, index_col=[0, 1, 2]).squeeze().rename(None))
    inputs.update({'surface': surface})

    ratio_surface = get_pandas(config['technical']['ratio_surface'], lambda x: pd.read_csv(x, index_col=[0]))
    inputs.update({'ratio_surface': ratio_surface})

    if config['macro']['surface_built'] is None:
        inputs.update({'surface_max': other_inputs['surface_max']})
        inputs.update({'surface_elasticity': other_inputs['surface_elasticity']})
    else:
        surface_built = get_pandas(config['macro']['surface_built'], lambda x: pd.read_csv(x, index_col=[0]).squeeze().rename(None))
        inputs.update({'surface_built': surface_built})

    if config['macro'].get('flow_construction') is not None:
        flow_construction = get_pandas(config['macro']['flow_construction'],
                                        lambda x: pd.read_csv(x, index_col=[0], header=None).squeeze())
        inputs.update({'flow_construction': flow_construction})

    ms_heater_built = get_pandas(config['ms_heater_built'], lambda x: pd.read_csv(x, index_col=[0], header=[0]))
    ms_heater_built.columns.set_names(['Heating system'], inplace=True)
    ms_heater_built.index.set_names(['Housing type'], inplace=True)
    inputs.update({'ms_heater_built': ms_heater_built})

    inputs.update({'performance_insulation': other_inputs['performance_insulation']})

    df = get_pandas(config['health_cost'], lambda x: pd.read_csv(x, index_col=[0, 1]))
    inputs.update({'health_cost': df})

    carbon_value = get_pandas(config['carbon_value'], lambda x: pd.read_csv(x, index_col=[0]).squeeze())
    inputs.update({'carbon_value': carbon_value.loc[idx]})

    carbon_emission = get_pandas(config['technical']['carbon_emission'], lambda x: pd.read_csv(x, index_col=[0]).rename_axis('Year'))
    inputs.update({'carbon_emission': carbon_emission.loc[idx, :]})

    footprint_built = get_pandas(config['technical']['footprint']['construction'], lambda x: pd.read_csv(x, index_col=[0]))
    inputs.update({'footprint_built': footprint_built})
    footprint_renovation = get_pandas(config['technical']['footprint']['renovation'], lambda x: pd.read_csv(x, index_col=[0, 1]))
    inputs.update({'footprint_renovation': footprint_renovation})

    inputs.update({'traditional_material': config['technical']['footprint']['Traditional material']})
    inputs.update({'bio_material': config['technical']['footprint']['Bio material']})

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

    if config['technical'].get('technical_progress') is not None:
        if 'insulation' in config['technical']['technical_progress'].keys():
            if config['technical']['technical_progress']['insulation']['activated']:
                value = config['technical']['technical_progress']['insulation']['value_end']
                start = config['technical']['technical_progress']['insulation']['start']
                end = config['technical']['technical_progress']['insulation']['end']
                value = round((1 + value) ** (1 / (end - start + 1)) - 1, 5)
                parsed_inputs['technical_progress'] = dict()
                parsed_inputs['technical_progress']['insulation'] = Series(value, index=range(start, end + 1)).reindex(idx).fillna(0)
        if 'heater' in config['technical']['technical_progress'].keys():
            if config['technical']['technical_progress']['heater']['activated']:
                value = config['technical']['technical_progress']['heater']['value_end']
                start = config['technical']['technical_progress']['heater']['start']
                end = config['technical']['technical_progress']['heater']['end']
                value = round((1 + value) ** (1 / (end - start + 1)) - 1, 5)
                parsed_inputs['technical_progress'] = dict()
                parsed_inputs['technical_progress']['heater'] = Series(value, index=range(start, end + 1)).reindex(idx).fillna(0)

    parsed_inputs['cost_heater'] *= cost_factor
    parsed_inputs['cost_insulation'] *= cost_factor
    parsed_inputs['energy_prices'].loc[range(config['start'] + 2, config['end']), :] *= prices_factor
    parsed_inputs['energy_taxes'].loc[range(config['start'] + 2, config['end']), :] *= prices_factor

    if isinstance(inputs['demolition_rate'], (float, int)):
        inputs['demolition_rate'] = pd.Series(inputs['demolition_rate'], index=idx[1:])

    parsed_inputs['flow_demolition'] = inputs['demolition_rate'] * stock.sum()

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
    parsed_inputs['flow_need'] = parsed_inputs['stock_need'] - parsed_inputs['stock_need'].shift(1)
    if 'flow_construction' not in parsed_inputs.keys():
        parsed_inputs['flow_construction'] = parsed_inputs['flow_need'] + parsed_inputs['flow_demolition']

    parsed_inputs['available_income'] = pd.Series(
        [inputs['available_income'] * (1 + config['macro']['income_rate']) ** (i - idx[0]) for i in idx], index=idx)

    if False:
        available_income_pop = (parsed_inputs['available_income'] / parsed_inputs['population_total']).dropna()

        surface_built = evolution_surface_built(inputs['surface'].xs(False, level='Existing'), inputs['surface_max'],
                                                inputs['surface_elasticity'], available_income_pop)

        surface_existing = pd.concat([parsed_inputs['surface'].xs(True, level='Existing')] * surface_built.shape[1], axis=1,
                                     keys=surface_built.columns)
        parsed_inputs['surface'] = pd.concat((surface_existing, surface_built), axis=0, keys=[True, False], names=['Existing'])

    parsed_inputs['surface'] = pd.concat([parsed_inputs['surface']] * len(idx), axis=1, keys=idx)

    if 'share_single_family_construction' in inputs.keys():
        temp = pd.concat((inputs['share_single_family_construction'], (1 - inputs['share_single_family_construction'])),
                         axis=1, keys=['Single-family', 'Multi-family'], names=['Housing type'])
        type_built = parsed_inputs['flow_construction'] * temp.T
    else:
        if 'share_multi_family' not in inputs.keys():
            parsed_inputs['share_multi_family'] = share_multi_family(parsed_inputs['stock_need'],
                                                                     inputs['factor_multi_family'])
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

    if not config['macro']['construction']:
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

    if config['simple']['prices_constant']:
        energy_prices = pd.concat([energy_prices.loc[config['start'], :]] * energy_prices.shape[0], keys=energy_prices.index,
                                  axis=1).T

    total_taxes = pd.DataFrame(0, index=energy_prices.index, columns=energy_prices.columns)
    for t in taxes:
        total_taxes = total_taxes.add(t.value, fill_value=0)

    if energy_taxes is not None:
        total_taxes = total_taxes.add(energy_taxes, fill_value=0)
        taxes += [PublicPolicy('energy_taxes', energy_taxes.index[0], energy_taxes.index[-1], energy_taxes, 'tax')]

    if config['simple']['taxes_constant']:
        total_taxes = pd.concat([total_taxes.loc[config['start'], :]] * total_taxes.shape[0], keys=total_taxes.index,
                                axis=1).T

    energy_vta = energy_prices * inputs['vta_energy_prices']
    taxes += [PublicPolicy('energy_vta', energy_vta.index[0], energy_vta.index[-1], energy_vta, 'tax')]
    total_taxes += energy_vta
    parsed_inputs['taxes'] = taxes
    parsed_inputs['total_taxes'] = total_taxes

    energy_prices = energy_prices.add(total_taxes, fill_value=0)
    parsed_inputs['energy_prices'] = energy_prices

    parsed_inputs['input_financing'].update(config['financing_cost'])

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

    lambda_1_values = [0.6, 0.65, 0.7, 0.75]
    lambda_sum_values = [0.85, 0.9, 0.95, 0.97]

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
                n = '{}_{}_{}'.format(round(lambda_1, 3), round(lambda_2, 3), i)
                df = concat([prices[year].loc[:, i].rename(year) for year in prices.keys()], axis=1).T
                result.update({n: df})

    result = {k: df for k, df in result.items() if (df > 0).all().all()}
    if path is not None:
        for name, df in result.items():
            df.to_csv(os.path.join(path, 'energy_prices_{}.csv'.format(name.replace('.', ''))))
        return {name.replace('.', ''): 'energy_prices_{}.csv'.format(name.replace('.', '')) for name in result.keys()}
    else:
        return result


def create_simple_policy(start, end, value=0.3, gest='insulation'):
    return PublicPolicy('sub_ad_valorem', start, end, value, 'subsidy_ad_valorem',
                        gest=gest)
