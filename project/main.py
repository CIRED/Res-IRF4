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

from project.write_output import grouped_output
from project.model import res_irf
from project.read_input import generate_price_scenarios, read_prices

LOG_FORMATTER = '%(asctime)s - %(process)s - %(name)s - %(levelname)s - %(message)s'


def run(path=None):
    start = time()

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=os.path.join('project', 'input', 'config.json'), help='path config file')

    parser.add_argument('-d', '--directory', default='project/input/config/policies', help='path config directory')
    parser.add_argument('-y', '--year', default=None, help='end year')
    parser.add_argument('-s', '--sensitivity', default=True, help='sensitivity')

    args = parser.parse_args()

    if not os.path.isdir(os.path.join('project', 'output')):
        os.mkdir(os.path.join('project', 'output'))

    if path is None:
        path = args.config
    with open(path) as file:
        configuration = json.load(file)

    config_policies = None
    name_policy = ''
    if 'assessment' in configuration.keys():
        if configuration['assessment']['activated']:
            config_policies = configuration['assessment']
            config_policies = {key: item for key, item in config_policies.items() if item is not None}
            name_policy = configuration['assessment']['Policy name'] + '_'

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
            name_policy = 'sensitivity_'
            config_sensitivity = configuration['sensitivity']
            if 'ZP' in config_sensitivity.keys():
                if config_sensitivity['ZP']:
                    configuration['ZP'] = copy.deepcopy(configuration['Reference'])
                    for name, policy in configuration['ZP']['policies'].items():
                        policy['end'] = configuration['Reference']['start'] + 2
                        configuration['ZP']['policies'][name] = policy
            if 'prices_constant' in config_sensitivity.keys():
                if config_sensitivity['prices_constant']:
                    configuration['PriceConstant'] = copy.deepcopy(configuration['Reference'])
                    configuration['PriceConstant']['prices_constant'] = True
            if 'prices_factor' in config_sensitivity.keys():
                values = config_sensitivity['prices_factor']
                if values:
                    if isinstance(values, float):
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

            if 'renovation_rate_ini' in config_sensitivity.keys():
                if config_sensitivity['renovation_rate_ini']:
                    configuration['RetrofitIni'] = copy.deepcopy(configuration['Reference'])
                    configuration['RetrofitIni']['renovation_rate_ini'] = config_sensitivity['renovation_rate_ini']
            if 'target_freeriders' in config_sensitivity.keys():
                values = config_sensitivity['target_freeriders']
                if values:
                    if isinstance(values, float):
                        values = [values]
                    for v in values:
                        configuration['FreeridersIni{:.0f}'.format(v * 100)] = copy.deepcopy(configuration['Reference'])
                        configuration['FreeridersIni{:.0f}'.format(v * 100)]['target_freeriders'] = v

            if 'preferences_zeros' in config_sensitivity.keys():
                if config_sensitivity['preferences_zeros']:
                    configuration['PreferencesZeros'] = copy.deepcopy(configuration['Reference'])
                    configuration['PreferencesZeros']['preferences_zeros'] = config_sensitivity['preferences_zeros']

            if 'calib_scale' in config_sensitivity.keys():
                if config_sensitivity['calib_scale']:
                    configuration['CalibScale'] = copy.deepcopy(configuration['Reference'])
                    configuration['CalibScale']['calib_scale'] = config_sensitivity['calib_scale']

            if 'mpr_global_retrofit' in config_sensitivity.keys():
                if config_sensitivity['mpr_global_retrofit']:
                    configuration['MprGlobalRetrofit'] = copy.deepcopy(configuration['Reference'])
                    configuration['MprGlobalRetrofit']['policies']['mpr'][
                        'global_retrofit'] = "project/input/policies/mpr_global_retrofit.csv"

            if 'mpr_no_serenite' in config_sensitivity.keys():
                if config_sensitivity['mpr_no_serenite']:
                    configuration['MprNoSerenite'] = copy.deepcopy(configuration['Reference'])
                    configuration['MprNoSerenite']['policies']['mpr'][
                        'mpr_serenite'] = None

            if 'mpr_no_bonus' in config_sensitivity.keys():
                if config_sensitivity['mpr_no_bonus']:
                    configuration['MprNoBonus'] = copy.deepcopy(configuration['Reference'])
                    configuration['MprNoBonus']['policies']['mpr']['bonus'] = None

        del configuration['sensitivity']

    t = datetime.today().strftime('%Y%m%d_%H%M%S')
    folder = os.path.join(os.path.join('project', 'output'), '{}{}'.format(name_policy, t))
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
        stocks = {i[0]: i[2] for i in results}

        logger.debug('Parsing results')
        if configuration.get('Reference').get('full_output'):
            grouped_output(result, folder, config_policies, config_sensitivity,
                           quintiles=configuration.get('Reference').get('quintiles'))

        logger.debug('Run time: {:,.0f} minutes.'.format((time() - start) / 60))
    except Exception as e:
        logger.exception(e)
        raise e


if __name__ == '__main__':

    logging.basicConfig()
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('matplotlib.axes').disabled = True

    run()
