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

from main import run


if __name__ == '__main__':
    """configs = ['project/input/config_zil.json',
               'project/input/config_cite.json',
               'project/input/config_cee.json',
               'project/input/config_reduced_tax.json',
               'project/input/config_carbon_tax.json']"""

    configs = ['project/input/config.json', 'project/input/config°bis.json']
    for config in configs:
        run(path=config)
