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
import json
from datetime import datetime
from copy import deepcopy

if __name__ == '__main__':

    logging.basicConfig()
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('matplotlib.axes').disabled = True

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', default=None, help='path config directory')
    parser.add_argument('-a', '--assessment', default=None, help='path config file with assessmnet')

    args = parser.parse_args()

    configs = []
    if args.directory is not None:
        configs = [os.path.join(args.directory, c) for c in os.listdir(args.directory) if c.split('.')[1] == 'json']

    if args.assessment is not None:
        with open(args.assessment) as file:
            configuration = json.load(file)

        policies = configuration['sensitivity']['assessment']

        t = datetime.today().strftime('%Y%m%d_%H%M%S')
        folder = os.path.join(os.path.join('project', 'output'), 'assessment_{}'.format(t))
        os.mkdir(folder)
        folder_config = os.path.join(folder, 'config')
        os.mkdir(folder_config)

        for policy in policies:

            _config = deepcopy(configuration)
            _config['assessment'] = {
                "activated": True,
                "Policy name": policy,
                "AP": "Reference",
                "AP-1": True,
                "ZP": True,
                "ZP+1": True,
                "AP-2020": True
            },
            del _config['sensitivity']



    for config in configs:
        # add try/except to continue if one config fail
        run(path=config)
