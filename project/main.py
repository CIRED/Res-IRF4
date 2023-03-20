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

from project.write_output import plot_compare_scenarios, indicator_policies, make_summary
from project.model import res_irf
from project.utils import get_json

LOG_FORMATTER = '%(asctime)s - %(process)s - %(name)s - %(levelname)s - %(message)s'


def run(path=None):
    start = time()

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=os.path.join('project', 'config', 'test', 'config.json'), help='path config file')

    parser.add_argument('-d', '--directory', default='project/config/policies', help='path config directory')
    parser.add_argument('-y', '--year', default=None, help='end year')
    parser.add_argument('-s', '--sensitivity', default=True, help='sensitivity')

    args = parser.parse_args()

    if not os.path.isdir(os.path.join('project', 'output')):
        os.mkdir(os.path.join('project', 'output'))

    if path is None:
        path = args.config
    with open(path) as file:
        configuration = json.load(file)

    policy_name = None
    prefix = ''
    if 'assessment' in configuration.keys():
        if configuration['assessment']['activated']:
            config_policies = configuration['assessment']
            config_policies = {key: item for key, item in config_policies.items() if item is not None}
            policy_name = configuration['assessment']['Policy name']
            prefix = policy_name

            if config_policies['AP-1']:
                configuration['AP-1'] = copy.deepcopy(configuration['Reference'])
                if config_policies['Policy name'] in ['global_retrofit', 'bonus', 'mpr_serenite']:
                    configuration['AP-1']['policies']['mpr'][config_policies['Policy name']]['end'] = \
                    configuration['Reference'][
                        'start'] + 2
                else:
                    configuration['AP-1']['policies'][config_policies['Policy name']]['end'] = configuration['Reference'][
                                                                                               'start'] + 2
                if config_policies['Policy name'] == 'mpr':
                    for options in ['global_retrofit', 'mpr_serenite', 'bonus']:
                        if isinstance(configuration['AP-1']['policies']['mpr'][options], dict):
                            configuration['AP-1']['policies']['mpr'][options]['end'] = configuration['Reference'][
                                                                                               'start'] + 2

            if config_policies['ZP']:
                configuration['ZP'] = copy.deepcopy(configuration['Reference'])
                for name, policy in configuration['ZP']['policies'].items():
                    policy['end'] = configuration['Reference']['start'] + 2
                    configuration['ZP']['policies'][name] = policy

                    if name == 'mpr':
                        for options in ['global_retrofit', 'bonus']:
                            if isinstance(configuration['ZP']['policies']['mpr'][options], dict):
                                configuration['ZP']['policies']['mpr'][options]['end'] = configuration['Reference'][
                                                                                               'start'] + 2

            if config_policies['ZP'] and config_policies['ZP+1']:
                configuration['ZP+1'] = copy.deepcopy(configuration['ZP'])
                if config_policies['Policy name'] in ['global_retrofit', 'bonus', 'mpr_serenite']:
                    configuration['ZP+1']['policies']['mpr'][config_policies['Policy name']]['end'] = \
                    configuration['Reference']['end']
                else:
                    configuration['ZP+1']['policies'][config_policies['Policy name']]['end'] = configuration['Reference'][
                    'end']

                if config_policies['Policy name'] == 'mpr':
                    for options in ['global_retrofit', 'mpr_serenite', 'bonus']:
                        if isinstance(configuration['ZP']['policies']['mpr'][options], dict):
                            configuration['ZP+1']['policies']['mpr'][options]['end'] = configuration['Reference'][
                    'end']

            list_years = [int(re.search('20[0-9][0-9]', key)[0]) for key in config_policies.keys() if
                          re.search('20[0-9][0-9]', key)]
            for year in list_years:
                if config_policies['AP-{}'.format(year)] and year < configuration['Reference']['end']:
                    configuration['AP-{}'.format(year)] = copy.deepcopy(configuration['Reference'])
                    if config_policies['Policy name'] in ['global_retrofit', 'bonus', 'mpr_serenite']:
                        configuration['AP-{}'.format(year)]['policies']['mpr'][config_policies['Policy name']]['end'] = year
                    else:
                        configuration['AP-{}'.format(year)]['policies'][config_policies['Policy name']]['end'] = year
                    if config_policies['Policy name'] == 'mpr':
                        for options in ['global_retrofit', 'mpr_serenite', 'bonus']:
                            if isinstance(configuration['ZP']['policies']['mpr'][options], dict):
                                configuration['AP-{}'.format(year)]['policies']['mpr'][options]['end'] = year

                    configuration['AP-{}'.format(year)]['end'] = year + 1

        del configuration['assessment']

    config_sensitivity = None
    if 'sensitivity' in configuration.keys():
        if configuration['sensitivity']['activated'] * args.sensitivity:
            prefix = 'sensitivity'
            config_sensitivity = configuration['sensitivity']
            if 'no_policy' in config_sensitivity.keys():
                if config_sensitivity['no_policy']:
                    configuration['NoPolicy'] = copy.deepcopy(configuration['Reference'])
                    configuration['NoPolicy']['simple']['no_policy'] = True
                    configuration['NoPolicy']['simple']['current_policies'] = False

            if 'current_policies' in config_sensitivity.keys():
                if config_sensitivity['current_policies']:
                    configuration['CurrentPolicies'] = copy.deepcopy(configuration['Reference'])
                    configuration['CurrentPolicies']['simple']['current_policies'] = True
            if 'prices_constant' in config_sensitivity.keys():
                if config_sensitivity['prices_constant']:
                    configuration['PriceConstant'] = copy.deepcopy(configuration['Reference'])
                    configuration['PriceConstant']['simple']['prices_constant'] = True
            if 'constant' in config_sensitivity.keys():
                if config_sensitivity['constant']:
                    configuration['Constant'] = copy.deepcopy(configuration['Reference'])
                    configuration['Constant']['simple']['prices_constant'] = True
                    configuration['Constant']['simple']['current_policies'] = True
                    configuration['Constant']['simple']['no_natural_replacement'] = True
                    configuration['Constant']['simple']['emission_constant'] = True

            if 'step' in config_sensitivity.keys():
                values = config_sensitivity['step']
                for v in values:
                    name = 'Step{}'.format(v)
                    configuration[name] = copy.deepcopy(configuration['Reference'])
                    configuration[name]['step'] = int(v)

            if 'building_stock' in config_sensitivity.keys():
                values = config_sensitivity['building_stock']
                if values:
                    if isinstance(values, str):
                        values = [values]
                    for v in values:
                        name = v.split('/')[-1].split('.')[0].replace('_', '')
                        configuration[name] = copy.deepcopy(configuration['Reference'])
                        configuration[name]['building_stock'] = v

            if 'deviation' in config_sensitivity.keys():
                values = config_sensitivity['deviation']
                if values:
                    if isinstance(values, (float, int)):
                        values = [values]
                    for v in values:
                        configuration['Deviation{:.0f}'.format(v * 100)] = copy.deepcopy(configuration['Reference'])
                        configuration['Deviation{:.0f}'.format(v * 100)]['renovation']['scale']['deviation'] = v

            if 'prices_factor' in config_sensitivity.keys():
                values = config_sensitivity['prices_factor']
                if values:
                    if isinstance(values, (float, int)):
                        values = [values]
                    for v in values:
                        configuration['PriceFactor{:.0f}'.format(v * 100)] = copy.deepcopy(configuration['Reference'])
                        configuration['PriceFactor{:.0f}'.format(v * 100)]['prices_factor'] = v
            if 'cost_factor' in config_sensitivity.keys():
                values = config_sensitivity['cost_factor']
                if values:
                    if isinstance(values, float):
                        values = [values]
                    for v in values:
                        configuration['CostFactor{:.0f}'.format(v * 100)] = copy.deepcopy(configuration['Reference'])
                        configuration['CostFactor{:.0f}'.format(v * 100)]['cost_factor'] = v

            if 'exogenous' in config_sensitivity.keys():
                values = config_sensitivity['exogenous']
                if values:
                    if isinstance(values, float):
                        values = [values]
                    for v in values:
                        configuration['Exogenous{}'.format(v)] = copy.deepcopy(configuration['Reference'])
                        configuration['Exogenous{}'.format(v)]['renovation']['endogenous'] = False
                        configuration['Exogenous{}'.format(v)]['renovation']['exogenous']['number'] = v

        del configuration['sensitivity']

    t = datetime.today().strftime('%Y%m%d_%H%M%S')
    if prefix == '':
        folder = os.path.join(os.path.join('project', 'output'), '{}'.format(t))
    else:
        folder = os.path.join(os.path.join('project', 'output'), '{}_{}'.format(prefix, t))
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

    logger.debug('Scenarios: {}'.format(', '.join(configuration.keys())))
    try:
        logger.debug('Launching processes')
        with Pool() as pool:
            results = pool.starmap(res_irf,
                                   zip(configuration.values(), [os.path.join(folder, n) for n in configuration.keys()]))
        result = {i[0]: i[1] for i in results}
        # stocks = {i[0]: i[2] for i in results}

        logger.debug('Parsing results')
        if configuration.get('Reference').get('full_output'):
            plot_compare_scenarios(result, folder, quintiles=configuration.get('Reference').get('simple').get('quintiles'))
            config_policies = get_json('project/input/policies/cba_inputs.json')
            if 'Reference' in result.keys() and len(result.keys()) > 1 and config_policies is not None:
                indicator_policies(result, folder, config_policies, policy_name=policy_name)
            make_summary(folder)

        logger.debug('Run time: {:,.0f} minutes.'.format((time() - start) / 60))
    except Exception as e:
        logger.exception(e)
        raise e


if __name__ == '__main__':

    logging.basicConfig()
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('matplotlib.axes').disabled = True

    run()
