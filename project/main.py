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

from write_output import grouped_output
from model import res_irf

LOG_FORMATTER = '%(asctime)s - %(process)s - %(name)s - %(levelname)s - %(message)s'

if __name__ == '__main__':
    start = time()

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='project/input/config.json', help='path config file')
    parser.add_argument('-d', '--directory', default='project/input/config/policies', help='path config directory')
    parser.add_argument('-y', '--year', default=None, help='end year')
    parser.add_argument('-s', '--sensitivity', default=True, help='sensitivity')

    args = parser.parse_args()

    if not os.path.isdir('project/output'):
        os.mkdir('project/output')

    with open(args.config) as file:
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
                configuration['AP-1']['policies'][config_policies['Policy name']]['end'] = configuration['Reference'][
                                                                                               'start'] + 2

            if config_policies['ZP']:
                configuration['ZP'] = copy.deepcopy(configuration['Reference'])
                for name, policy in configuration['ZP']['policies'].items():
                    policy['end'] = configuration['Reference']['start'] + 2
                    configuration['ZP']['policies'][name] = policy

            if config_policies['ZP'] and config_policies['ZP+1']:
                configuration['ZP+1'] = copy.deepcopy(configuration['ZP'])
                configuration['ZP+1']['policies'][config_policies['Policy name']]['end'] = configuration['Reference'][
                    'end']

            list_years = [int(re.search('20[0-9][0-9]', key)[0]) for key in config_policies.keys() if
                          re.search('20[0-9][0-9]', key)]
            for year in list_years:
                if config_policies['AP-{}'.format(year)] and year < configuration['Reference']['end']:
                    configuration['AP-{}'.format(year)] = copy.deepcopy(configuration['Reference'])
                    configuration['AP-{}'.format(year)]['policies'][config_policies['Policy name']]['end'] = year
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
            if 'mpr_global_retrofit' in config_sensitivity.keys():
                if config_sensitivity['mpr_global_retrofit']:
                    configuration['MprGlobalRetrofit'] = copy.deepcopy(configuration['Reference'])
                    configuration['MprGlobalRetrofit']['policies']['mpr'][
                        'global_retrofit'] = "project/input/policies/mpr_global_retrofit.csv"
            if 'mpr_no_bonus' in config_sensitivity.keys():
                if config_sensitivity['mpr_no_bonus']:
                    configuration['MprNoBonus'] = copy.deepcopy(configuration['Reference'])
                    configuration['MprNoBonus']['policies']['mpr']['bonus'] = None
        del configuration['sensitivity']

    folder = os.path.join('project/output', '{}{}'.format(name_policy, datetime.today().strftime('%Y%m%d_%H%M%S')))
    os.mkdir(folder)

    logging.basicConfig(filename=os.path.join(folder, 'root_log.log'),
                        level=logging.DEBUG,
                        format=LOG_FORMATTER)
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('matplotlib.axes').disabled = True

    if args.year:
        for key in configuration.keys():
            configuration[key]['end'] = int(args.year)

    logging.debug('Scenarios: {}'.format(', '.join(configuration.keys())))
    try:
        logging.debug('Launching processes')
        with Pool() as pool:
            results = pool.starmap(res_irf,
                                   zip(configuration.values(), [os.path.join(folder, n) for n in configuration.keys()]))
        result = {i[0]: i[1] for i in results}
        stocks = {i[0]: i[2] for i in results}

        logging.debug('Parsing results')
        grouped_output(result, folder, config_policies, config_sensitivity)

        logging.debug('Run time: {:,.0f} minutes.'.format((time() - start) / 60))
    except Exception as e:
        logging.exception(e)
        raise e
