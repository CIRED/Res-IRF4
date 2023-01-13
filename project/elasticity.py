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
from project.model import res_irf, calibration_res_irf
from project.read_input import generate_price_scenarios, read_prices

LOG_FORMATTER = '%(asctime)s - %(process)s - %(name)s - %(levelname)s - %(message)s'


def run_elasticity(path=None):

    if path is None:
        path = os.path.join('project', 'input', 'config_elasticity.json')

    start = time()

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=os.path.join('project', 'input', 'config.json'), help='path config file')

    args = parser.parse_args()

    if not os.path.isdir(os.path.join('project', 'output')):
        os.mkdir(os.path.join('project', 'output'))

    if path is None:
        path = args.config
    with open(path) as file:
        configuration = json.load(file)

    t = datetime.today().strftime('%Y%m%d_%H%M%S')
    folder = os.path.join(os.path.join('project', 'output'), '{}'.format(t))
    os.mkdir(folder)

    logger = logging.getLogger('log_{}'.format(t))
    logger.setLevel('DEBUG')
    logger.propagate = False
    # file handler
    file_handler = logging.FileHandler(os.path.join(folder, 'root_log.log'))
    file_handler.setFormatter(logging.Formatter(LOG_FORMATTER))
    logger.addHandler(file_handler)

    end = configuration['Reference']['end']
    configuration['Reference']['end'] = 2020
    for name, policy in configuration['Reference']['policies'].items():
        policy['end'] = policy['start']
        configuration['Reference'][name] = policy

    calibration = calibration_res_irf(configuration['Reference'], os.path.join(folder, 'Reference'))

    configuration['Reference']['end'] = end
    # already calibrated
    configuration['Reference']['start'] += 2

    configuration['Reference']['full_output'] = False
    energy_prices = read_prices(configuration['Reference'])
    folder_prices = os.path.join('project', 'input', 'prices')
    scenarios = generate_price_scenarios(energy_prices, path=folder_prices, year_2=2020, year_1=2019, year_0=2018,
                                         nb_draws=10)
    for key, path in scenarios.items():
        configuration[key] = copy.deepcopy(configuration['Reference'])
        configuration[key]['energy_prices'] = os.path.join(folder_prices, path)

    del configuration['Reference']

    logger.debug('Scenarios: {}'.format(', '.join(configuration.keys())))
    try:
        logger.debug('Launching processes')
        with Pool() as pool:
            results = pool.starmap(res_irf,
                                   zip(configuration.values(), [os.path.join(folder, n) for n in configuration.keys()],
                                       [calibration] * len(configuration.keys())))
        result = {i[0]: i[1] for i in results}

        logger.debug('Parsing results')

        logger.debug('Run time: {:,.0f} minutes.'.format((time() - start) / 60))
    except Exception as e:
        logger.exception(e)
        raise e


if __name__ == '__main__':

    logging.basicConfig()
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('matplotlib.axes').disabled = True

    run_elasticity()
