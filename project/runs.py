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
import argparse
import logging

from project.main import run
from json import load, dumps
from datetime import datetime
from copy import deepcopy
from project.utils import get_json

if __name__ == '__main__':

    logging.basicConfig()
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('matplotlib.axes').disabled = True

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', default=None, help='path config directory')
    parser.add_argument('-a', '--assessment', default=None, help='path config file with assessment')
    parser.add_argument('-cpu', '--cpu', default=6, help='number of cpu')

    args = parser.parse_args()

    configs, folder = [], None

    if args.directory is not None:
        configs = [os.path.join(args.directory, c) for c in os.listdir(args.directory) if c.split('.')[1] == 'json']
        configs = sorted(configs)

    if args.assessment is not None:
        configs = []

        with open(args.assessment) as file:
            configuration = load(file)

        policies = configuration['scenarios']['assessment']

        t = datetime.today().strftime('%Y%m%d_%H%M%S')
        folder = os.path.join(os.path.join('project', 'output'), 'assessment_{}'.format(t))
        os.mkdir(folder)

        for policy in policies:
            _config = deepcopy(configuration)
            _config['assessment'] = {
                "activated": True,
                "Policy name": policy,
                "AP": "Reference",
                "AP-1": True,
                "ZP": True,
                "ZP+1": True,
                "ZP+2019": True,
                "ZP+2020": True,
                "ZP+2021": True,
                "ZP+2022": True,
                "ZP+2023": True,
                "ZP+2024": True,
                "ZP+2025": True,
                "ZP+2030": True,
                "ZP+2035": True,
                "ZP+2040": True,
                "ZP+2045": True,
                "ZP+2050": True,
                "AP-2019": True,
                "AP-2020": True,
                "AP-2021": True,
                "AP-2022": True,
                "AP-2023": True,
                "AP-2024": True,
                "AP-2025": True,
                "AP-2030": True,
                "AP-2035": True,
                "AP-2040": True,
                "AP-2045": True,
                "AP-2050": True
            }

            del _config['scenarios']
            if 'sensitivity' in _config.keys():
                del _config['sensitivity']
            configs += [_config]

    for c in configs:
        # add try/except to continue if one config fail
        run(path=c, folder=folder)

