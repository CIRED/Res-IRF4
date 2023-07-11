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
import copy
import logging
import json
from time import time
from multiprocessing import get_context, Pool
from datetime import datetime
import re
import argparse

from project.write_output import plot_compare_scenarios, indicator_policies, make_summary, plot_compare_scenarios_simple
from project.model import res_irf, prepare_config
from project.utils import get_json, parse_policies

LOG_FORMATTER = '%(asctime)s - %(process)s - %(name)s - %(levelname)s - %(message)s'


def run(path=None, folder=None):
    start = time()

    output_compare = 'full'

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=os.path.join('project', 'config', 'test', 'config.json'), help='path config file')

    parser.add_argument('-d', '--directory', default='project/config/policies', help='path config directory')
    parser.add_argument('-a', '--assessment', default=None, help='path config file with assessmnet')
    parser.add_argument('-y', '--year', default=None, help='end year')

    args = parser.parse_args()

    if not os.path.isdir(os.path.join('project', 'output')):
        os.mkdir(os.path.join('project', 'output'))

    if path is None:
        path = args.config

    if isinstance(path, str):
        with open(path) as file:
            configuration = json.load(file)
    else:
        configuration = path

    for key in [k for k in configuration.keys() if k not in ['assessment', 'scenarios', 'sensitivity',
                                                             'policies_scenarios']]:
        configuration[key] = prepare_config(configuration[key])

    policy_name = None
    prefix = ''
    if 'policies_scenarios' in configuration.keys():
        if configuration['policies_scenarios']['activated']:
            output_compare = 'simple'
            prefix = 'policies_scenarios'
            config_policies = get_json(configuration['policies_scenarios']['file'])
            for key, item in config_policies.items():
                configuration[key] = copy.deepcopy(configuration['Reference'])
                configuration[key]['simple']['no_policy'] = False
                configuration[key]['simple']['current_policies'] = False
                configuration[key]['policies'] = copy.deepcopy(item)

        del configuration['policies_scenarios']

    if 'assessment' in configuration.keys():
        if configuration['assessment']['activated']:
            config_policies = configuration['assessment']
            config_policies = {key: item for key, item in config_policies.items() if item is not None}
            policy_name = configuration['assessment']['Policy name']
            end = configuration['Reference']['policies'][policy_name]['end']
            configuration['Reference']['end'] = end

            prefix = policy_name

            if config_policies['AP-1']:
                configuration['AP-1'] = copy.deepcopy(configuration['Reference'])
                configuration['AP-1']['policies'][config_policies['Policy name']]['end'] = configuration['Reference']['start'] + 2

            if config_policies['ZP']:
                configuration['ZP'] = copy.deepcopy(configuration['Reference'])
                for name, policy in configuration['ZP']['policies'].items():
                    policy['end'] = configuration['Reference']['start'] + 2
                    configuration['ZP']['policies'][name] = policy

            if config_policies['ZP'] and config_policies['ZP+1']:
                configuration['ZP+1'] = copy.deepcopy(configuration['ZP'])
                configuration['ZP+1']['policies'][config_policies['Policy name']]['end'] = configuration['Reference']['end']

            list_years = [int(re.search('20[0-9][0-9]', key)[0]) for key in config_policies.keys() if
                          re.search('20[0-9][0-9]', key)]
            for year in list_years:
                if config_policies['AP-{}'.format(year)] and year < configuration['Reference']['end'] and year > configuration['Reference']['policies'][policy_name]['start'] and year < configuration['Reference']['policies'][policy_name]['end']:
                    configuration['AP-{}'.format(year)] = copy.deepcopy(configuration['Reference'])
                    configuration['AP-{}'.format(year)]['policies'][config_policies['Policy name']]['end'] = year
                    configuration['AP-{}'.format(year)]['end'] = year + 1

        del configuration['assessment']

    if 'scenarios' in configuration.keys():
        if configuration['scenarios']['activated']:
            prefix = 'scenarios'
            config_scenarios = configuration['scenarios']

            if config_scenarios.get('no_policy') is not None:
                configuration['NoPolicy'] = copy.deepcopy(configuration['Reference'])
                configuration['NoPolicy']['simple']['no_policy'] = True
                configuration['NoPolicy']['simple']['current_policies'] = False

            if config_scenarios.get('current_policies') is not None:
                configuration['CurrentPolicies'] = copy.deepcopy(configuration['Reference'])
                configuration['CurrentPolicies']['simple']['current_policies'] = True

            if config_scenarios.get('policies') is not None:
                for key, item in config_scenarios['policies'].items():
                    configuration[key] = copy.deepcopy(configuration['Reference'])
                    temp = {'policies': item}
                    parse_policies(temp)
                    configuration[key]['policies'] = temp['policies']
                    # do not copy Reference simple
                    configuration[key]['simple']['no_policy'] = False
                    configuration[key]['simple']['current_policies'] = False

            if config_scenarios.get('remove_policies') is not None:
                for key in config_scenarios['remove_policies']:
                    s = 'No{}'.format(key.capitalize().replace('_', ''))
                    configuration[s] = copy.deepcopy(configuration['Reference'])
                    if key in configuration[s]['policies'].keys():
                        configuration[s]['policies'][key]['end'] = configuration[s]['start'] + 2
                    # do not copy Reference simple
                    configuration[s]['simple']['no_policy'] = False
                    configuration[s]['simple']['current_policies'] = False

            if config_scenarios.get('add_policies') is not None:
                for key, item in config_scenarios['add_policies'].items():
                    s = 'Add{}'.format(key.capitalize().replace('_', ''))
                    configuration[s] = copy.deepcopy(configuration['Reference'])
                    configuration[s]['policies'].update({key: item})
                    # do not copy Reference simple
                    configuration[s]['simple']['no_policy'] = False
                    configuration[s]['simple']['current_policies'] = False

            if config_scenarios.get('prices_constant') is not None:
                configuration['PriceConstant'] = copy.deepcopy(configuration['Reference'])
                configuration['PriceConstant']['simple']['prices_constant'] = True

            if config_scenarios.get('no_natural_replacement') is not None:
                configuration['NoNaturalReplacement'] = copy.deepcopy(configuration['Reference'])
                configuration['NoNaturalReplacement']['simple']['no_natural_replacement'] = True

            if config_scenarios.get('constant') is not None:
                configuration['Constant'] = copy.deepcopy(configuration['Reference'])
                configuration['Constant']['simple']['prices_constant'] = True
                configuration['Constant']['simple']['current_policies'] = True
                configuration['Constant']['simple']['no_natural_replacement'] = True
                configuration['Constant']['simple']['emission_constant'] = True

            if config_scenarios.get('exogenous') is not None:
                values = config_scenarios['exogenous']
                if values:
                    if isinstance(values, float):
                        values = [values]
                    for v in values:
                        configuration['Exogenous{}'.format(v)] = copy.deepcopy(configuration['Reference'])
                        configuration['Exogenous{}'.format(v)]['renovation']['endogenous'] = False
                        configuration['Exogenous{}'.format(v)]['renovation']['exogenous']['number'] = v

        del configuration['scenarios']

    if 'sensitivity' in configuration.keys():
        if configuration['sensitivity']['activated']:
            prefix = 'sensitivity'
            config_sensitivity = configuration['sensitivity']

            if config_sensitivity.get('energy_prices') is not None:
                if isinstance(config_sensitivity['energy_prices'], str):
                    config_sensitivity['energy_prices'] = get_json(config_sensitivity['energy_prices'])

                for key, item in config_sensitivity['energy_prices'].items():
                    configuration[key] = copy.deepcopy(configuration['Reference'])
                    configuration[key]['energy']['energy_prices'] = copy.deepcopy(item)

            if config_sensitivity.get('building_stock') is not None:
                values = config_sensitivity['building_stock']
                if values:
                    if isinstance(values, str):
                        values = [values]
                    for v in values:
                        name = v.split('/')[-1].split('.')[0].replace('_', '')
                        configuration[name] = copy.deepcopy(configuration['Reference'])
                        configuration[name]['building_stock'] = v

            if config_sensitivity.get('lifetime_insulation') is not None:
                values = config_sensitivity['lifetime_insulation']
                if isinstance(values, (float, int)):
                    values = [values]
                for v in values:
                    configuration['Lifetime{}'.format(v)] = copy.deepcopy(configuration['Reference'])
                    configuration['Lifetime{}'.format(v)]['technical']['lifetime_insulation'] = v

            if config_sensitivity.get('scale_renovation') is not None:
                values = config_sensitivity['scale_renovation']
                if isinstance(values, dict):
                    values = [values]
                for v in values:
                    name = str(v['target']).replace('.', '')
                    configuration['ScaleInsulation{}'.format(name)] = copy.deepcopy(configuration['Reference'])
                    configuration['ScaleInsulation{}'.format(name)]['renovation']['scale'] = v

            if config_sensitivity.get('scale_heater') is not None:
                values = config_sensitivity['scale_heater']
                if isinstance(values, dict):
                    values = [values]
                for v in values:
                    name = str(v['target']).replace('.', '')
                    configuration['ScaleHeater{}'.format(name)] = copy.deepcopy(configuration['Reference'])
                    configuration['ScaleHeater{}'.format(name)]['switch_heater']['scale'] = v

            if config_sensitivity.get('step') is not None:
                values = config_sensitivity['step']
                for v in values:
                    name = 'Step{}'.format(v)
                    configuration[name] = copy.deepcopy(configuration['Reference'])
                    configuration[name]['step'] = int(v)

        del configuration['sensitivity']

    if 'uncertainty' in configuration.keys():
        if configuration['uncertainty']['activated']:
            prefix = 'uncertainty'
            config_uncertainty = {k: i for k, i in configuration['uncertainty'].items() if k != 'activated'}
            keys, values = zip(*config_uncertainty.items())
            from itertools import product
            permutations_scenarios = [dict(zip(keys, v)) for v in product(*values)]
            for k, scenarios in enumerate(permutations_scenarios):
                scenario_name = 'S{}'.format(k)
                configuration[scenario_name] = copy.deepcopy(configuration['Reference'])
                for scenario_input, value in scenarios.items():
                    if scenario_input == 'energy_prices_factor':
                        configuration[scenario_name]['energy']['energy_prices']['factor'] = value
                    if scenario_input == 'carbon_emission':
                        configuration[scenario_name]['energy']['carbon_emission'] = value
                    if scenario_input == 'scale_renovation':
                        configuration[scenario_name]['renovation']['scale'] = value

        del configuration['uncertainty']

    t = datetime.today().strftime('%Y%m%d_%H%M%S')

    if folder is None or not prefix:
        folder = os.path.join('project', 'output')
        if not prefix:
            folder = os.path.join(folder, '{}'.format(t))
        else:
            folder = os.path.join(folder, '{}_{}'.format(prefix, t))
    else:
        folder = os.path.join(folder, '{}'.format(prefix))

    os.mkdir(folder)

    logger = logging.getLogger('log_{}'.format(t))
    logger.setLevel('DEBUG')
    logger.propagate = False
    # file handler
    file_handler = logging.FileHandler(os.path.join(folder, 'root_log.log'))
    file_handler.setFormatter(logging.Formatter(LOG_FORMATTER))
    logger.addHandler(file_handler)

    if args.year:
        for key in configuration.keys():
            configuration[key]['end'] = int(args.year)

    for s in configuration.keys():
        if s != 'Reference':
            # to change after tests
            configuration[s]['figures'] = False

    logger.debug('Scenarios: {}'.format(', '.join(configuration.keys())))
    try:
        logger.debug('Launching processes')
        with Pool(6) as pool:
            results = pool.starmap(res_irf,
                                   zip(configuration.values(), [os.path.join(folder, n) for n in configuration.keys()]))
        result = {i[0]: i[1] for i in results}
        # stocks = {i[0]: i[2] for i in results}

        logger.debug('Parsing results')
        config_policies = get_json('project/input/policies/cba_inputs.json')
        if configuration.get('Reference').get('output') == 'full' and output_compare == 'full':
            plot_compare_scenarios(result, folder, quintiles=configuration.get('Reference').get('simple').get('quintiles'))
            if 'Reference' in result.keys() and len(result.keys()) > 1 and config_policies is not None:
                indicator_policies(result, folder, config_policies, policy_name=policy_name)
            make_summary(folder, option='comparison')
        elif output_compare == 'simple':
            plot_compare_scenarios_simple(result, folder, quintiles=configuration.get('Reference').get('simple').get('quintiles'))

        logger.debug('Run time: {:,.0f} minutes.'.format((time() - start) / 60))
    except Exception as e:
        logger.exception(e)
        raise e


if __name__ == '__main__':

    logging.basicConfig()
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('matplotlib.axes').disabled = True

    run()
