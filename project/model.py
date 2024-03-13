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


"""Main module of the model.

This module contains the main functions to run the model. The main function is res_irf that runs the model for a given
scenario.


Functions
---------
res_irf
    Res-IRF model.
calibration_res_irf
    Calibrate Res-IRF and returns calibrated parameters.
get_inputs
    Initialize thermal buildings object based on input dictionary.
config2inputs
    Create main Python object from configuration file.
initialize
    Create main Python objects read by model.
stock_turnover
    Update stock vintage due to renovation, demolition and construction.
select_post_inputs
    Inputs used during post-treatment but not used during the iteration.
get_config
    Get the configuration file.
prepare_config
    Prepare the configuration file.
"""


import os
import pandas as pd
from time import time
import json
from importlib import resources
from pickle import load, dump
from copy import deepcopy
import logging
# import psutil

from project.building import AgentBuildings
from project.read_input import read_stock, read_policies, read_inputs, parse_inputs, dump_inputs, create_simple_policy
from project.write_output import plot_scenario, compare_results
from project.utils import reindex_mi, deciles2quintiles, get_json, create_logger, make_policies_tables, subplots_attributes, plot_thermal_insulation, parse_policies
from project.utils import memory_object, get_size, size_dict
from project.input.resources import resources_data


def get_config() -> dict:
    with resources.path('project.config', 'config.json') as f:
        with open(f) as file:
            return json.load(file)['Reference']


def prepare_config(config):
    # read parameters for master file
    if 'file' in config.keys():
        temp = deepcopy(config)
        path = config['file']
        config = get_json(path)
        # if temp of keys in config replace values with temp
        for k, v in temp.items():
            if k in config.keys():
                if isinstance(v, dict):
                    config[k].update(v)
                else:
                    config[k] = v

        # config.update(temp)
        # del config['file']

    # read policies
    parse_policies(config)
    return config


def config2inputs(config=None):
    """Create main Python object from configuration file.

    Parameters
    ----------
    config: dict

    Returns
    -------

    """

    if config is None:
        config = get_config()

    config = prepare_config(config)

    year = config['start']
    stock = read_stock(config)
    inputs = read_inputs(config)

    # parse_policies(config)

    if config['simple'].get('heating_system'):
        replace = config['simple']['heating_system']
        shape_ini = stock.shape[0]
        stock = stock.reset_index('Heating system')
        stock['Heating system'] = stock['Heating system'].replace(replace)
        stock = stock.dropna()
        stock = stock.set_index('Heating system', append=True).squeeze()
        stock = stock.groupby(stock.index.names).sum()
        shape_out = stock.shape[0]
        print('From {} to {}'.format(shape_ini, shape_out))
        to_drop = []
        if 'Heating-District heating' in config['simple']['heating_system'].keys():
            if config['simple']['heating_system']['Heating-District heating'] is None:
                inputs['flow_district_heating'] = None
                to_drop += ['Heating-District heating']
        """to_replace = [k for k, i in replace.items() if i not in inputs['ms_heater'].columns and i is not None]
        if to_replace:
            raise NotImplemented"""

        list_heater = list(stock.index.get_level_values('Heating system').unique())
        list_heater.extend([i if i not in replace.keys() else replace[i] for i in inputs['calibration_heater']['ms_heater'].columns])
        list_heater = list(set(list_heater))
        list_heater = [i for i in list_heater if i is not None]

        if isinstance(inputs['calibration_heater']['ms_heater'], pd.DataFrame):
            inputs['calibration_heater']['ms_heater'].drop([i for i in inputs['calibration_heater']['ms_heater'].columns if i not in list_heater], axis=1, inplace=True)
            inputs['calibration_heater']['ms_heater'].drop([i for i in inputs['calibration_heater']['ms_heater'].index.get_level_values('Heating system') if i not in list_heater], axis=0, inplace=True, level='Heating system')
            inputs['calibration_heater']['ms_heater'] = (inputs['calibration_heater']['ms_heater'].T / inputs['calibration_heater']['ms_heater'].sum(axis=1)).T

        if isinstance(inputs['ms_heater_built'], pd.DataFrame):
            # Replace index that are in Heating system level with replace
            inputs['ms_heater_built'] = inputs['ms_heater_built'].rename(index=replace).groupby(
                inputs['ms_heater_built'].index.names).sum()

        inputs['cost_heater'].drop([i for i in inputs['cost_heater'].index if i not in list_heater], inplace=True)

    if config['simple'].get('insulation'):
        pass

    if config['simple'].get('no_heating_switch'):
        inputs['lifetime_heater'].loc[:] = float('inf')

    if config['simple']['surface']:
        surface = (reindex_mi(inputs['surface'], stock.index) * stock).sum() / stock.sum()
        inputs['surface'] = pd.Series(round(surface, 0), index=inputs['surface'].index)

    if config['simple']['ratio_surface']:
        ratio_surface = (reindex_mi(inputs['ratio_surface'], stock.index).T * stock).T.sum() / stock.sum()
        for idx in inputs['ratio_surface'].index:
            inputs['ratio_surface'].loc[idx, :] = ratio_surface.round(1)

    if config['simple'].get('emission_constant'):
        inputs['carbon_emission'] = pd.concat([inputs['carbon_emission'].loc[config['start'], :]] * inputs['carbon_emission'].shape[0], axis=1,
                                              keys=inputs['carbon_emission'].index).T

    if config['simple'].get('no_natural_replacement'):
        inputs['demolition_rate'] = None
        config['macro']['construction'] = False

    if config['simple'].get('no_policy'):
        for name, policy in config['policies'].items():
            if policy['end'] > policy['start']:
                policy['end'] = config['start'] + 2
                config['policies'][name] = policy

    if config['simple'].get('current_policies'):
        for name, policy in config['policies'].items():
            # remove policies that start after calibration
            if policy['start'] > config['start'] + 1:
                policy['end'] = policy['start']

            if policy['end'] > policy['start']:
                policy['end'] = config['end']

            if policy.get('growth_insulation'):
                policy['growth_insulation'] = None

            if policy.get('growth_heater'):
                policy['growth_heater'] = None

            if policy.get('growth'):
                policy['growth'] = None

            for k in policy.keys():
                if isinstance(policy[k], dict):
                    if policy[k].get('start') is not None:
                        if policy[k]['start'] > config['start'] + 1:
                            policy[k]['end'] = policy[k]['start']
                        if policy[k]['end'] > policy[k]['start']:
                            policy[k]['end'] = config['end']

            config['policies'][name] = policy

    if config.get('policies') is not None:
        policies_heater, policies_insulation, taxes = read_policies(config)
    else:
        policies_insulation, policies_heater, taxes = [], [], []

    if config['simple'].get('current_policies'):
        for t in taxes:
            t.value = pd.concat([t.value.iloc[0, :]] * t.value.shape[0], axis=1, keys=t.value.index).T

    if config['simple'].get('income_constant'):
        inputs['income_rate'] = 0

    policies_insulation = [p for p in policies_insulation if p.end > p.start]
    policies_heater = [p for p in policies_heater if p.end > p.start]

    if config['simple']['quintiles']:
        stock, policies_heater, policies_insulation, inputs = deciles2quintiles(stock, policies_heater,
                                                                                policies_insulation, inputs)

    if config['simple'].get('detailed_output') is None:
        config['simple']['detailed_output'] = True

    return config, inputs, stock, year, policies_heater, policies_insulation, taxes


def select_post_inputs(parsed_inputs):
    """Inputs used during post-treatment but not used during the iteration.

    Parameters
    ----------
    parsed_inputs: dict

    Returns
    -------
    dict
    """

    vars = ['carbon_emission', 'renewable_gas',
            'population', 'surface', 'embodied_energy_renovation', 'carbon_footprint_renovation',
            'Carbon footprint construction (MtCO2)', 'Embodied energy construction (TWh PE)',
            'health_cost_dpe', 'health_cost_income', 'carbon_value_kwh', 'carbon_value',
            'use_subsidies', 'implicit_discount_rate', 'energy_prices_wt']

    return {key: item for key, item in parsed_inputs.items() if key in vars}


def get_inputs(path=None, config=None, variables=None):
    """Initialize thermal buildings object based on input dictionary.

    Parameters
    ----------
    path: str, optional
        If None do not write output.
    config: dict, optional
        If config is None use configuration file of Reference scenario
    variables: list, optional
        'buildings', 'energy_prices', 'cost_insulation', 'carbon_emission', 'carbon_value_kwh', 'health_cost'

    Returns
    -------
    dict
    """
    if variables is None:
        variables = ['buildings', 'energy_prices', 'cost_insulation', 'carbon_emission', 'carbon_value_kwh',
                     'health_cost', 'income', 'cost_heater', 'health_cost_dpe']

    config, inputs, stock, year, policies_heater, policies_insulation, taxes = config2inputs(config)
    inputs_dynamics = initialize(inputs, stock, year, taxes, path=path, config=config)
    output = {'buildings': inputs_dynamics['buildings'],
              'income': inputs['income'],
              'energy_prices': inputs_dynamics['energy_prices'],
              'cost_insulation': inputs_dynamics['cost_insulation'],
              'cost_heater': inputs_dynamics['cost_heater'],
              'carbon_emission': inputs_dynamics['post_inputs']['carbon_emission'],
              'carbon_value_kwh': inputs_dynamics['post_inputs']['carbon_value_kwh'],
              'efficiency': inputs['efficiency'],
              'health_cost_dpe': inputs_dynamics['health_cost_dpe'],
              'implicit_discount_rate': inputs_dynamics['post_inputs']['implicit_discount_rate']
              }
    output = {k: item for k, item in output.items() if k in variables}

    return output


def initialize(inputs, stock, year, taxes, path=None, config=None, logger=None, level_logger='DEBUG'):
    """Create main Python objects read by model.

    Parameters
    ----------
    inputs
    stock
    year
    taxes
    config
    path
    logger

    Returns
    -------

    """

    if config is None:
        config = get_config()

    parsed_inputs = parse_inputs(inputs, taxes, config, stock)
    if path is not None and config['simple']['detailed_output'] and config.get('figures') is not False:
        dump_inputs(parsed_inputs, path)
    post_inputs = select_post_inputs(parsed_inputs)
    if logger is None:
        logger = create_logger(path=path, level=level_logger)
    logger.info('Creating AgentBuildings object')

    if path is not None and config['simple']['detailed_output']:
        with open(os.path.join(path, 'config.json'), 'w') as fp:
            json.dump(config, fp)

    buildings = AgentBuildings(stock, parsed_inputs['surface'], parsed_inputs['ratio_surface'], parsed_inputs['efficiency'],
                               parsed_inputs['income'], parsed_inputs['preferences'],
                               parsed_inputs['performance_insulation_renovation'],
                               lifetime_heater=parsed_inputs['lifetime_heater'],
                               path=path,
                               year=year,
                               endogenous=config['renovation']['endogenous'],
                               exogenous=config['renovation']['exogenous'],
                               logger=logger,
                               quintiles=config['simple']['quintiles'],
                               rational_behavior_insulation=parsed_inputs['rational_behavior_insulation'],
                               rational_behavior_heater=parsed_inputs['rational_behavior_heater'],
                               resources_data=resources_data,
                               detailed_output=config['simple'].get('detailed_output'),
                               figures=config['simple'].get('figures'),
                               method_health_cost=config.get('method_health_cost'),
                               residual_rate=config['technical'].get('residual_rate'),
                               constraint_heat_pumps=config['technical'].get('constraint_heat_pumps', True),
                               variable_size_heater=config['technical'].get('variable_size_heater', True),
                               temp_sink=parsed_inputs['temp_sink'])

    technical_progress = None
    if 'technical_progress' in parsed_inputs.keys():
        technical_progress = parsed_inputs['technical_progress']

    inputs_dynamic = {
        'buildings': buildings,
        'energy_prices': parsed_inputs['energy_prices'],
        'energy_prices_wt': parsed_inputs['energy_prices_wt'],
        'taxes': parsed_inputs['taxes'],
        'post_inputs': post_inputs,
        'cost_heater': parsed_inputs['cost_heater'],
        'lifetime_heater': parsed_inputs['lifetime_heater'],
        'calibration_heater': parsed_inputs['calibration_heater'],
        'flow_district_heating': parsed_inputs['flow_district_heating'],
        'cost_insulation': parsed_inputs['cost_insulation'],
        'lifetime_insulation': parsed_inputs['lifetime_insulation'],
        'calibration_renovation': parsed_inputs['calibration_renovation'],
        'demolition_rate': parsed_inputs['demolition_rate'],
        'flow_built': parsed_inputs['flow_built'],
        'financing_cost': parsed_inputs.get('input_financing'),
        'technical_progress': technical_progress,
        'consumption_ini': parsed_inputs['consumption_ini'],
        'supply': parsed_inputs['supply'],
        'premature_replacement': parsed_inputs['premature_replacement'],
        'health_cost_dpe': parsed_inputs['health_cost_dpe'],
        'health_cost_income': parsed_inputs['health_cost_income'],
        'output': config['output'],
        'hourly_profile': parsed_inputs.get('hourly_profile')
    }
    return inputs_dynamic


def stock_turnover(buildings, prices, taxes, cost_heater, cost_insulation, lifetime_insulation,
                   p_heater, p_insulation, flow_built, year,
                   post_inputs,  calib_heater=None, calib_renovation=None, financing_cost=None,
                   prices_before=None, climate=None, district_heating=None, step=1, demolition_rate=None, memory=False,
                   exogenous_social=None, output_options='full', premature_replacement=None, supply=None,
                   carbon_content=None, carbon_content_before=None):
    """Update stock vintage due to renovation, demolition and construction.


    Returns
    -------
    buildings : AgentBuildings
        Updated AgentBuildings object.
    stock: : Series
    output : Series
    """

    buildings.logger.info('Run {}'.format(year))

    if prices_before is None:
        prices_before = prices
    if carbon_content_before is None:
        carbon_content_before = carbon_content

    buildings.year = year

    # bill rebate - taxes revenues recycling
    bill_rebate, bill_rebate_before = 0, 0
    for t in [t for t in taxes if t.recycling is not None]:

        recycling_revenue = None
        if t.recycling_ini is not None:
            recycling_revenue = t.recycling_ini
        elif year - 1 in buildings.taxes_revenues.keys():
            if t.name in buildings.taxes_revenues[year - 1].index:
                recycling_revenue = buildings.taxes_revenues[year - 1][t.name] * 10**9
        else:
            continue

        if isinstance(t.recycling, pd.Series):
            target = (buildings.stock * reindex_mi(t.recycling, buildings.stock.index)).sum().round(0)
            factor = t.recycling / target
        else:
            factor = 1 / buildings.stock.sum()

        bill_rebate = factor * recycling_revenue
        buildings.bill_rebate.update({year: bill_rebate})

    if year - 1 in buildings.bill_rebate.keys():
        bill_rebate_before = buildings.bill_rebate[year - 1]

    heater_demolition = None
    if demolition_rate is not None:
        temp = buildings.flow_demolition(demolition_rate, step=step)
        heater_demolition = temp.groupby('Heating system').sum()
        buildings.add_flows([- temp])
    buildings.logger.info('Calculation retrofit')
    if output_options == 'full':
        buildings.consumption_before_retrofit = buildings.store_consumption(prices_before,
                                                                            carbon_content_before,
                                                                            bill_rebate=bill_rebate_before)

    flow_retrofit = buildings.flow_retrofit(prices, cost_heater, cost_insulation, lifetime_insulation,
                                            financing_cost=financing_cost,
                                            policies_heater=p_heater,
                                            policies_insulation=p_insulation,
                                            calib_renovation=calib_renovation,
                                            calib_heater=calib_heater,
                                            district_heating=district_heating,
                                            exogenous_social=exogenous_social,
                                            carbon_value_kwh=post_inputs['carbon_value_kwh'].loc[year, :],
                                            carbon_value=post_inputs['carbon_value'].loc[year],
                                            carbon_content=carbon_content,
                                            bill_rebate=bill_rebate)

    """if memory:
        memory_dict = {'Memory': '{:.1f} MiB'.format(psutil.Process().memory_info().rss / (1024 * 1024)),
                       'AgentBuildings': '{:.1f} MiB'.format(get_size(buildings) / 10 ** 6)}
        memory_dict.update(size_dict(memory_object(buildings), n=50, display=False))
        buildings.memory.update({year: memory_dict})"""

    buildings.add_flows([flow_retrofit])

    flows_obligation = buildings.flow_obligation(p_insulation, prices, cost_insulation,
                                                 financing_cost=financing_cost)
    if flows_obligation is not None:
        buildings.add_flows(flows_obligation)

    new_heating = None
    if flow_built is not None:
        buildings.add_flows([flow_built])
        new_heating = flow_built.groupby('Heating system').sum()

    buildings.logger.info('Writing output')
    if output_options == 'full':
        buildings.logger.debug('Full output')
        stock, output = buildings.parse_output_run(prices, post_inputs, climate=climate, step=step, taxes=taxes,
                                                   bill_rebate=bill_rebate)
    elif output_options == 'cost_benefit':
        buildings.logger.debug('Cost-benefit output')
        stock = buildings.simplified_stock().rename(year)
        output = buildings.parse_output_run_cba(prices, post_inputs, step=step, taxes=taxes, bill_rebate=bill_rebate)
    elif output_options == 'consumption':
        buildings.logger.debug('Consumption output')
        stock = buildings.simplified_stock().rename(year)
        output = buildings.parse_output_consumption(prices, bill_rebate=bill_rebate)

    else:
        raise NotImplemented('output_options should be full, cost_benefit or consumption')

    # updating heating-system vintage
    if True:
        switch_heating = buildings._heater_store['replacement'].groupby('Heating system').sum().sum()
        if new_heating is not None:
            pass
            # switch_heating = switch_heating.add(new_heating, fill_value=0)
        temp = buildings.heater_vintage.loc[:, 1] - buildings._heater_store['replacement'].groupby('Heating system').sum().sum(axis=1)

        # update heater_vintage with switch_heating for year == lifetime heater
        # reduce column of 1 to account for vintage
        buildings.heater_vintage.columns = buildings.heater_vintage.columns - 1
        buildings.heater_vintage.drop(0, axis=1, inplace=True)

        if heater_demolition is not None:
            heater_demolition = heater_demolition - temp
            heater_demolition.dropna(inplace=True)
            temp = dict()
            # demolition target old heating-system ?
            for i in heater_demolition.index:
                x = buildings.heater_vintage.loc[i, :int(buildings.lifetime_heater[i] / 2)]
                temp.update({i: heater_demolition.loc[i] * x / x.sum()})
            temp = pd.DataFrame(temp).T.rename_axis('Heating system', axis=0).rename_axis('Year', axis=1)
            buildings.heater_vintage = buildings.heater_vintage.sub(temp, fill_value=0)
            # test not negative values

        for i in switch_heating.index:
            buildings.heater_vintage.loc[i, buildings.lifetime_heater.loc[i]] = switch_heating.loc[i]

        # test
        heating = buildings.stock.groupby('Heating system').sum()
        temp = pd.concat((heating, buildings.heater_vintage.sum(axis=1)), axis=1)

    return buildings, stock, output


def res_irf(config, path, level_logger='DEBUG'):
    """Res-IRF model.

    Parameters
    ----------
    config: dict
        Scenario-specific input
    path: str
        Scenario-specific output path
    level_logger: str

    Returns
    -------
    str
        Scenario name
    pd.DataFrame
        Detailed results
    """
    start_run = time()

    os.mkdir(path)
    if config['simple'].get('level_logger') is not None:
        level_logger = config['simple']['level_logger']

    logger = create_logger(path=path, level=level_logger)
    try:
        logger.info('Reading input')
        config, inputs, stock, year, policies_heater, policies_insulation, taxes = config2inputs(config)

        if policies_heater + policies_insulation and config['simple']['detailed_output'] and \
                config.get('figures') is not False:
            make_policies_tables(policies_heater + policies_insulation, os.path.join(path, 'policy_scenario.csv'),
                                 plot=True)

        inputs_dynamics = initialize(inputs, stock, year, taxes, path=path, config=config, logger=logger)
        buildings, energy_prices = inputs_dynamics['buildings'], inputs_dynamics['energy_prices']
        technical_progress = inputs_dynamics['technical_progress']

        output, stock = pd.DataFrame(), pd.DataFrame()
        buildings.logger.info('Calibration energy consumption {}'.format(buildings.first_year))

        if config['output'] == 'full' and buildings.path_ini is not None:
            plot_thermal_insulation(buildings.stock, save=os.path.join(buildings.path_ini, 'thermal_insulation.png'))
            _stock = buildings.simplified_stock(energy_level=True)
            _stock = _stock.groupby(
                ['Occupancy status', 'Income owner', 'Income tenant', 'Housing type', 'Energy', 'Performance']).sum()
            subplots_attributes(_stock, dict_order=resources_data['index'],
                                dict_color=resources_data['colors'],
                                sharey=True, save=os.path.join(buildings.path_ini, 'stock.png'))

        if config.get('calibration'):
            with open(config['calibration'], "rb") as file:
                calibration = load(file)
                buildings.calibration_exogenous(**calibration)
        else:
            buildings.calibration_consumption(energy_prices.loc[buildings.first_year, :],
                                              inputs_dynamics['consumption_ini'],
                                              inputs_dynamics['health_cost_income'],
                                              inputs_dynamics['health_cost_dpe'])

        s, o = buildings.parse_output_run(energy_prices.loc[buildings.first_year, :], inputs_dynamics['post_inputs'],
                                          taxes=taxes)
        stock = pd.concat((stock, s), axis=1)
        output = pd.concat((output, o), axis=1)

        timestep = 1
        if config.get('step'):
            timestep = config['step']
        years = [config['start'] + 1]
        years += range(config['start'] + 2, config['end'], timestep)
        if config['end'] - 1 not in years:
            years.append(config['end'] - 1)

        if inputs_dynamics['supply']['insulation'] is not None:
            inputs_dynamics['cost_insulation'] /= inputs_dynamics['supply']['insulation']['markup_insulation']

        if config['simple'].get('no_policy_insulation'):
            for p in policies_insulation:
                p.end = config['start'] + 2

        if config['simple'].get('no_policy_heater'):
            for p in policies_heater:
                if p.variable:
                    p.end = config['start'] + 2

        for k, year in enumerate(years):
            start = time()

            if year == config['end'] - 1:
                yrs = [year]
            else:
                yrs = range(year, years[k + 1])
            step = len(yrs)

            prices = energy_prices.loc[year, :]
            carbon_content = inputs_dynamics['post_inputs']['carbon_emission'].loc[year, :]

            if year > config['start']:
                prices_before = energy_prices.loc[year - 1, :]
                if 'Emission content (gCO2/kWh)' in buildings.store_over_years[year - 1].keys():
                    carbon_content_before = buildings.store_over_years[year - 1]['Emission content (gCO2/kWh)']
                else:
                    carbon_content_before = inputs_dynamics['post_inputs']['carbon_emission'].loc[year - 1, :]
            else:
                prices_before = prices
                carbon_content_before = carbon_content

            p_heater = [p for p in policies_heater if (year >= p.start) and (year < p.end)]
            p_insulation = [p for p in policies_insulation if (year >= p.start) and (year < p.end)]
            f_built = inputs_dynamics['flow_built'].loc[:, yrs]
            if isinstance(f_built, pd.DataFrame):
                f_built = f_built.sum(axis=1).rename(year)
            f_built = f_built.dropna()

            flow_district_heating = None
            if inputs_dynamics['flow_district_heating'] is not None:
                flow_district_heating = inputs_dynamics['flow_district_heating'].loc[year]

            if technical_progress is not None:
                if technical_progress.get('insulation') is not None:
                    inputs_dynamics['cost_insulation'] *= (1 + technical_progress['insulation'].loc[year])**step
                if technical_progress.get('heater') is not None:
                    heat_pump = [i for i in resources_data['index']['Heat pumps'] if i in inputs_dynamics['cost_heater'].index]
                    inputs_dynamics['cost_heater'].loc[heat_pump] *= (1 + technical_progress['heater'].loc[year])**step

            buildings, s, o = stock_turnover(buildings, prices, taxes,
                                             inputs_dynamics['cost_heater'],
                                             inputs_dynamics['cost_insulation'],
                                             inputs_dynamics['lifetime_insulation'],
                                             p_heater, p_insulation, f_built, year,
                                             inputs_dynamics['post_inputs'],
                                             calib_renovation=inputs_dynamics['calibration_renovation'],
                                             calib_heater=inputs_dynamics['calibration_heater'],
                                             premature_replacement=inputs_dynamics['premature_replacement'],
                                             financing_cost=inputs_dynamics['financing_cost'],
                                             supply=inputs_dynamics['supply'],
                                             district_heating=flow_district_heating,
                                             demolition_rate=inputs_dynamics['demolition_rate'],
                                             exogenous_social=inputs.get('exogenous_social'),
                                             output_options=config['output'],
                                             climate=config.get('climate'),
                                             prices_before=prices_before,
                                             carbon_content=carbon_content,
                                             carbon_content_before=carbon_content_before,
                                             step=step)

            stock = pd.concat((stock, s), axis=1)
            stock.index.names = s.index.names
            output = pd.concat((output, o), axis=1)
            buildings.logger.info('Run time {}: {:,.0f} seconds.'.format(year, round(time() - start, 2)))
            if year == buildings.first_year + 1 and config['output'] == 'full' and buildings.path_ini is not None:
                compare_results(o, buildings.path)
                # inputs_dynamics['post_inputs']['implicit_discount_rate']
                buildings.make_static_analysis(inputs_dynamics['cost_insulation'], inputs_dynamics['cost_heater'],
                                               prices, 0.05, 0.05, inputs_dynamics['post_inputs']['health_cost_dpe'],
                                               inputs_dynamics['post_inputs']['carbon_emission'].loc[year, :],
                                               carbon_value=50)

                with open(os.path.join(buildings.path_calibration, 'calibration.pkl'), 'wb') as file:
                    dump({
                        'coefficient_global': buildings.coefficient_global,
                        'coefficient_heater': buildings.coefficient_heater,
                        'constant_insulation_extensive': buildings.constant_insulation_extensive,
                        'constant_insulation_intensive': buildings.constant_insulation_intensive,
                        'constant_heater': buildings.constant_heater,
                        'scale_insulation': buildings.scale_insulation,
                        'scale_heater': buildings.scale_heater
                    }, file)

        if path is not None:
            buildings.logger.info('Writing output in {}'.format(path))

            if buildings.memory:
                pd.DataFrame(buildings.memory).to_csv(os.path.join(path, 'memory.csv'))

            output.round(3).to_csv(os.path.join(path, 'output.csv'))
            buildings.logger.info('Dumping output in {}'.format(os.path.join(path, 'output.csv')))

            if config['output'] == 'full':
                stock.round(2).to_csv(os.path.join(path, 'stock.csv'))
            if buildings.path_ini is not None:
                buildings.logger.info('Creating standard figures')
                plot_scenario(output, stock, buildings)

        buildings.logger.info('Run time Res-IRF: {:,.1f} minutes.'.format(round((time() - start_run) / 60, 2)))

        return os.path.basename(os.path.normpath(path)), output, stock

    except Exception as e:
        logger.exception(e)
        raise e


def calibration_res_irf(path, config=None, level_logger='DEBUG'):
    """Calibrate Res-IRF and returns calibrated parameters.
    Function is useful for running multiple scenarios with the same calibration.
    Typical example is for sensitivity analysis or elasticity calculation.
    Parameters
    ----------
    config
    path
    level_logger
    Returns
    -------
    """
    if not os.path.isdir(path):
        os.mkdir(path)
    logger = create_logger(path=path, level=level_logger)
    try:
        logger.info('Reading input')
        config, inputs, stock, year, policies_heater, policies_insulation, taxes = config2inputs(config)
        inputs_dynamics = initialize(inputs, stock, year, taxes, path=path, config=config, logger=logger)
        buildings, energy_prices = inputs_dynamics['buildings'], inputs_dynamics['energy_prices']

        buildings.logger.info('Calibration energy consumption {}'.format(buildings.first_year))
        buildings.calibration_consumption(energy_prices.loc[buildings.first_year, :],
                                          inputs_dynamics['consumption_ini'],
                                          inputs_dynamics['health_cost_income'],
                                          inputs_dynamics['health_cost_dpe']
                                          )

        output = pd.DataFrame()
        _, o = buildings.parse_output_run(energy_prices.loc[buildings.first_year, :], inputs_dynamics['post_inputs'])
        output = pd.concat((output, o), axis=1)

        year = buildings.first_year + 1
        prices = energy_prices.loc[year, :]
        p_heater = [p for p in policies_heater if (year >= p.start) and (year < p.end)]
        p_insulation = [p for p in policies_insulation if (year >= p.start) and (year < p.end)]
        f_built = inputs_dynamics['flow_built'].loc[:, year].dropna()
        flow_district_heating = None
        if inputs_dynamics['flow_district_heating'] is not None:
            flow_district_heating = inputs_dynamics['flow_district_heating'].loc[year]
        carbon_content = inputs_dynamics['post_inputs']['carbon_emission'].loc[year, :]

        buildings, s, o = stock_turnover(buildings, prices, taxes,
                                         inputs_dynamics['cost_heater'],
                                         inputs_dynamics['cost_insulation'], inputs_dynamics['lifetime_insulation'],
                                         p_heater, p_insulation, f_built, year, inputs_dynamics['post_inputs'],
                                         district_heating=flow_district_heating,
                                         calib_renovation=inputs_dynamics['calibration_renovation'],
                                         calib_heater=inputs_dynamics['calibration_heater'],
                                         financing_cost=inputs_dynamics['financing_cost'],
                                         demolition_rate=inputs_dynamics['demolition_rate'],
                                         supply=inputs_dynamics['supply'],
                                         premature_replacement=inputs_dynamics['premature_replacement'],
                                         carbon_content=carbon_content
                                         )

        output = pd.concat((output, o), axis=1)
        output.to_csv(os.path.join(buildings.path, 'output_calibration.csv'))

        if year == buildings.first_year + 1:
            compare_results(o, buildings.path)

        calibration = {
            'coefficient_global': buildings.coefficient_global,
            'coefficient_heater': buildings.coefficient_heater,
            'constant_heater': buildings.constant_heater,
            'scale_heater': buildings.scale_heater,
            'constant_insulation_intensive': buildings.constant_insulation_intensive,
            'constant_insulation_extensive': buildings.constant_insulation_extensive,
            'scale_insulation': buildings.scale_insulation,
            'number_firms_insulation': buildings.number_firms_insulation,
            'number_firms_heater': buildings.number_firms_heater,
            'rational_hidden_cost': buildings.rational_hidden_cost,
            'hi_threshold': buildings.hi_threshold
        }

        return calibration

    except Exception as e:
        logger.exception(e)
        raise e
