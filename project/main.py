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
    parser.add_argument('-a', '--assessment', default=None, help='path config file with assessment')
    parser.add_argument('-y', '--year', default=None, help='end year')
    parser.add_argument('-cpu', '--cpu', default=6, help='number of cpu')

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

    for key in [k for k in configuration.keys() if k not in ['assessment', 'assessment_test', 'scenarios', 'sensitivity',
                                                             'policies_scenarios']]:
        configuration[key] = prepare_config(configuration[key])

    policies_name = None
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

    if 'assessment_test' in configuration.keys():
        if configuration['assessment_test']['activated']:
            policy_name = 'policy_test'
            prefix = 'assessment_test'
            policies = configuration['assessment_test']
            if 'file' in policies.keys():
                temp = get_json(policies['file'])
                policies.update(temp)
                del policies['file']

            configuration['Reference']['simple']['no_policy'] = False
            configuration['Reference']['simple']['current_policies'] = False
            configuration['ZP'] = copy.deepcopy(configuration['Reference'])
            del policies['activated']
            for key, item in policies.items():
                configuration['ZP+{}'.format(key)] = copy.deepcopy(configuration['Reference'])
                configuration['ZP+{}'.format(key)]['policies'].update({'policy_test': item})

        del configuration['assessment_test']
        if 'scenarios' in configuration.keys():
            del configuration['scenarios']

    if 'assessment' in configuration.keys():
        if configuration['assessment']['activated']:
            config_policies = configuration['assessment']
            config_policies = {key: item for key, item in config_policies.items() if item is not None}
            if not isinstance(configuration['assessment']['Policy name'], list):
                policies_name = [configuration['assessment']['Policy name']]
            else:
                policies_name = configuration['assessment']['Policy name']

            prefix = '-'.join(policies_name)

            for policy_name in policies_name:
                if policy_name not in configuration['Reference']['policies'].keys():
                    raise ValueError('Policy name not in Reference policies')
                if len(policies_name) == 1:
                    end = configuration['Reference']['policies'][policy_name]['end']
                    configuration['Reference']['end'] = end

                if config_policies.get('AP-1'):
                    if 'AP-1' not in configuration.keys():
                        configuration['AP-1'] = copy.deepcopy(configuration['Reference'])

                    temp = max(configuration['Reference']['policies'][policy_name]['start'], configuration['Reference']['start'] + 2)
                    configuration['AP-1']['policies'][policy_name]['years_stop'] = range(temp, configuration['Reference']['policies'][policy_name]['end'])

                if config_policies.get('ZP'):
                    if 'ZP' not in configuration.keys():
                        configuration['ZP'] = copy.deepcopy(configuration['Reference'])
                        for name, policy in configuration['ZP']['policies'].items():
                            policy['end'] = configuration['Reference']['start'] + 2
                            configuration['ZP']['policies'][name] = policy

                    temp = max(configuration['Reference']['policies'][policy_name]['start'], configuration['Reference']['start'] + 2)
                    configuration['ZP']['policies'][policy_name]['years_stop'] = range(temp, configuration['Reference']['policies'][policy_name]['end'])
                    configuration['ZP']['policies'][policy_name]['end'] = configuration['Reference']['policies'][policy_name]['end']

                if config_policies.get('ZP') and config_policies.get('ZP+1'):
                    if 'ZP+1' not in configuration.keys():
                        configuration['ZP+1'] = copy.deepcopy(configuration['ZP'])
                    configuration['ZP+1']['policies'][policy_name] = copy.deepcopy(configuration['Reference']['policies'][policy_name])

                if configuration['Reference']['policies'][policy_name]['policy'] not in ['obligation',
                                                                                         'restriction_energy',
                                                                                         'subsidy_cap']:
                    list_years = [int(re.search('20[0-9][0-9]', key)[0]) for key in config_policies.keys() if
                                  re.search('20[0-9][0-9]', key)]
                    list_years = list(set(list_years))
                    for year in list_years:
                        if config_policies['AP-{}'.format(year)] and year < configuration['Reference']['end'] and year >= configuration['Reference']['policies'][policy_name]['start'] and year < configuration['Reference']['policies'][policy_name]['end']:
                            if 'AP-{}'.format(year) not in configuration.keys():
                                configuration['AP-{}'.format(year)] = copy.deepcopy(configuration['Reference'])
                            configuration['AP-{}'.format(year)]['policies'][policy_name]['year_stop'] = year
                            configuration['AP-{}'.format(year)]['end'] = year + 1
                        if config_policies['ZP+{}'.format(year)] and year < configuration['Reference']['end'] and year >= configuration['Reference']['policies'][policy_name]['start'] and year < configuration['Reference']['policies'][policy_name]['end']:
                            if 'ZP+{}'.format(year) not in configuration.keys():
                                configuration['ZP+{}'.format(year)] = copy.deepcopy(configuration['ZP'])
                            configuration['ZP+{}'.format(year)]['policies'][policy_name] = copy.deepcopy(
                                configuration['Reference']['policies'][policy_name])
                            temp = max(configuration['Reference']['start'] + 2, configuration['Reference']['policies'][policy_name]['start'])
                            years_stop = [i for i in range(temp, year + 1) if i != year]
                            configuration['ZP+{}'.format(year)]['policies'][policy_name]['years_stop'] = years_stop
                            configuration['ZP+{}'.format(year)]['policies'][policy_name]['end'] = year + 1
                            configuration['ZP+{}'.format(year)]['end'] = year + 1
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

            if config_sensitivity.get('financing_cost') is not None:
                for key, item in config_sensitivity['financing_cost'].items():
                    configuration['financing_cost_{}'.format(key)] = copy.deepcopy(configuration['Reference'])
                    configuration['financing_cost_{}'.format(key)]['financing_cost'] = copy.deepcopy(item)

            if config_sensitivity.get('debt_income_ratio') is not None:
                for key, item in config_sensitivity['debt_income_ratio'].items():
                    configuration['debt_income_ratio_{}'.format(key)] = copy.deepcopy(configuration['Reference'])
                    configuration['debt_income_ratio_{}'.format(key)]['financing_cost']['debt_income_ratio'] = item

            if config_sensitivity.get('constraint_heat_pumps') is not None:
                for key, item in config_sensitivity['constraint_heat_pumps'].items():
                    configuration['constraint_heat_pumps_{}'.format(key)] = copy.deepcopy(configuration['Reference'])
                    configuration['constraint_heat_pumps_{}'.format(key)]['technical']['constraint_heat_pumps'] = item

            if config_sensitivity.get('variable_size_heater') is not None:
                for key, item in config_sensitivity['variable_size_heater'].items():
                    configuration['variable_size_heater_{}'.format(key)] = copy.deepcopy(configuration['Reference'])
                    configuration['variable_size_heater_{}'.format(key)]['technical']['variable_size_heater'] = item

            if config_sensitivity.get('carbon_emission') is not None:
                for key, item in config_sensitivity['carbon_emission'].items():
                    configuration['carbon_emission_{}'.format(key)] = copy.deepcopy(configuration['Reference'])
                    configuration['carbon_emission_{}'.format(key)]['energy']['carbon_emission'] = item

            if config_sensitivity.get('residual_rate') is not None:
                for key, item in config_sensitivity['residual_rate'].items():
                    configuration['residual_rate_{}'.format(key)] = copy.deepcopy(configuration['Reference'])
                    configuration['residual_rate_{}'.format(key)]['technical']['residual_rate'] = item

            if config_sensitivity.get('carbon_emission') is not None:
                for key, item in config_sensitivity['carbon_emission'].items():
                    configuration['carbon_emission_{}'.format(key)] = copy.deepcopy(configuration['Reference'])
                    configuration['carbon_emission_{}'.format(key)]['energy']['carbon_emission'] = item

            if config_sensitivity.get('income_rate') is not None:
                for key, item in config_sensitivity['income_rate'].items():
                    configuration['income_rate_{}'.format(key)] = copy.deepcopy(configuration['Reference'])
                    configuration['income_rate_{}'.format(key)]['macro']['income_rate'] = item

            if config_sensitivity.get('factor_energy_prices') is not None:
                for key, item in config_sensitivity['factor_energy_prices'].items():
                    configuration['factor_prices_{}'.format(key)] = copy.deepcopy(configuration['Reference'])
                    configuration['factor_prices_{}'.format(key)]['energy']['energy_prices']['factor'] = item

            if config_sensitivity.get('technical_progress_insulation') is not None:
                for key, item in config_sensitivity['technical_progress_insulation'].items():
                    configuration['progress_insulation_{}'.format(key)] = copy.deepcopy(configuration['Reference'])
                    configuration['progress_insulation_{}'.format(key)]['technical']['technical_progress']['insulation'] = copy.deepcopy(item)

            if config_sensitivity.get('technical_progress_heater') is not None:
                for key, item in config_sensitivity['technical_progress_heater'].items():
                    configuration['progress_heater_{}'.format(key)] = copy.deepcopy(configuration['Reference'])
                    configuration['progress_heater_{}'.format(key)]['technical']['technical_progress']['heater'] = copy.deepcopy(item)

            if config_sensitivity.get('district_heating') is not None:
                for key, item in config_sensitivity['district_heating'].items():
                    configuration['district_heating_{}'.format(key)] = copy.deepcopy(configuration['Reference'])
                    configuration['district_heating_{}'.format(key)]['switch_heater']['district_heating'] = item

            if config_sensitivity.get('renewable_gas') is not None:
                for key, item in config_sensitivity['renewable_gas'].items():
                    configuration['renewable_gas_{}'.format(key)] = copy.deepcopy(configuration['Reference'])
                    configuration['renewable_gas_{}'.format(key)]['energy']['renewable_gas'] = item

            if config_sensitivity.get('ms_heater_built') is not None:
                for key, item in config_sensitivity['ms_heater_built'].items():
                    configuration['heater_construction_{}'.format(key)] = copy.deepcopy(configuration['Reference'])
                    configuration['heater_construction_{}'.format(key)]['switch_heater']['ms_heater_built'] = item

            if config_sensitivity.get('turnover') is not None:
                for key, item in config_sensitivity['turnover'].items():
                    configuration['turnover_{}'.format(key)] = copy.deepcopy(configuration['Reference'])
                    configuration['turnover_{}'.format(key)]['macro']['flow_construction'] = item['flow_construction']
                    configuration['turnover_{}'.format(key)]['macro']['demolition_rate'] = item['demolition_rate']

            if config_sensitivity.get('scale_heater') is not None:
                for key, item in config_sensitivity['scale_heater'].items():
                    configuration['scale_heater_{}'.format(key)] = copy.deepcopy(configuration['Reference'])
                    configuration['scale_heater_{}'.format(key)]['switch_heater']['scale'] = copy.deepcopy(item)

            if config_sensitivity.get('scale_renovation') is not None:
                for key, item in config_sensitivity['scale_renovation'].items():
                    configuration['scale_renovation_{}'.format(key)] = copy.deepcopy(configuration['Reference'])
                    configuration['scale_renovation_{}'.format(key)]['renovation']['scale'] = copy.deepcopy(item)

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
                    configuration['lifetimeinsulation_{}'.format(v)] = copy.deepcopy(configuration['Reference'])
                    configuration['lifetimeinsulation_{}'.format(v)]['technical']['lifetime_insulation'] = v

            if config_sensitivity.get('step') is not None:
                values = config_sensitivity['step']
                for v in values:
                    name = 'Step{}'.format(v)
                    configuration[name] = copy.deepcopy(configuration['Reference'])
                    configuration[name]['step'] = int(v)

        del configuration['sensitivity']

    if 'uncertainty' in configuration.keys():
        if configuration['uncertainty']['activated']:
            output_compare = 'simple'
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
            configuration[s]['simple']['detailed_output'] = False
            configuration[s]['simple']['figures'] = False

    if len(configuration.keys()) > 10 and policies_name is None:
        output_compare = 'none'

    logger.debug('Scenarios: {}'.format(', '.join(configuration.keys())))
    try:
        logger.debug('Launching processes')
        with Pool(int(args.cpu)) as pool:
            results = pool.starmap(res_irf,
                                   zip(configuration.values(), [os.path.join(folder, n) for n in configuration.keys()]))
        result = {i[0]: i[1] for i in results}
        # stocks = {i[0]: i[2] for i in results}

        logger.debug('Parsing results')
        config_policies = get_json('project/input/policies/cba_inputs.json')
        if configuration.get('Reference').get('output') == 'full' and output_compare == 'full':
            if 'Reference' in result.keys() and len(result.keys()) > 1 and config_policies is not None:
                _, indicator = indicator_policies(result, folder, config_policies, policy_name=policies_name)
                # add NPV to result
                for scenario in result.keys():
                    temp = 0
                    if scenario in indicator.columns:
                        if 'NPV' in indicator.index:
                            temp = indicator.loc['NPV', scenario]
                    result[scenario].loc['NPV (Billion Euro)', result[scenario].columns[-1]] = temp

            if policies_name is None:
                plot_compare_scenarios(result, folder, quintiles=configuration.get('Reference').get('simple').get('quintiles'))
                make_summary(folder, option='comparison')
        elif output_compare == 'simple':
            plot_compare_scenarios_simple(result, folder, quintiles=configuration.get('Reference').get('simple').get('quintiles'))
        else:
            pass

        logger.debug('Run time: {:,.0f} minutes.'.format((time() - start) / 60))
    except Exception as e:
        logger.exception(e)
        raise e


if __name__ == '__main__':

    logging.basicConfig()
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('matplotlib.axes').disabled = True

    run()
