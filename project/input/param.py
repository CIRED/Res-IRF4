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

import pandas as pd
from project.utils import get_pandas

generic_input = dict()
generic_input['pop_housing_min'] = 2
generic_input['factor_pop_housing'] = -0.007
generic_input['factor_multi_family'] = 0.87
generic_input['price_index'] = 1

input_financing = dict()
input_financing['share_debt'] = (0.139, 1.15 * 10**-5)

input_financing['factor_saving_rate'] = pd.Series([2, 2, 2, 2, 1.6, 1.6, 1.3, 1.3, 1, 1],
                                                  index=pd.Index(
                                                      ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10'],
                                                      name='Income owner'))
generic_input['input_financing'] = input_financing


surface_elasticity = pd.Series({('Single-family', 'Owner-occupied'): 0.2,
                                ('Single-family', 'Privately rented'): 0.2,
                                ('Single-family', 'Social-housing'): 0.01,
                                ('Multi-family', 'Owner-occupied'): 0.01,
                                ('Multi-family', 'Privately rented'): 0.01,
                                ('Multi-family', 'Social-housing'): 0.01})
surface_elasticity.index.set_names(['Housing type', 'Occupancy status'], inplace=True)
generic_input['surface_elasticity'] = surface_elasticity

surface_max = pd.Series({('Single-family', 'Owner-occupied'): 160,
                         ('Single-family', 'Privately rented'): 101,
                         ('Single-family', 'Social-housing'): 90,
                         ('Multi-family', 'Owner-occupied'): 89,
                         ('Multi-family', 'Privately rented'): 76,
                         ('Multi-family', 'Social-housing'): 76})
surface_max.index.set_names(['Housing type', 'Occupancy status'], inplace=True)
generic_input['surface_max'] = surface_max


