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
import pickle

generic_input = dict()
generic_input['pop_housing_min'] = 2
generic_input['factor_pop_housing'] = -0.007
generic_input['factor_multi_family'] = 0.87
generic_input['available_income'] = 1421 * 10 ** 9
generic_input['price_index'] = 1

generic_input['income'] = pd.Series([13628, 20391, 24194, 27426, 31139, 35178, 39888, 45400, 54309, 92735],
                                    index=pd.Index(['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']))

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

generic_input['rotation_rate'] = pd.Series([0.121, 0.021, 0.052],
                                           index=pd.Index(['Owner-occupied', 'Privately rented', 'Social-housing'],
                                                          name='Occupancy status'))

generic_input['consumption_ini'] = pd.Series([33, 117, 36, 79],
                                             index=pd.Index(['Electricity', 'Natural gas', 'Oil fuel', 'Wood fuel'],
                                                            name='Heating energy'))

generic_input['stock_ini'] = 29037000

with open('project/input/parameters_thermal_module.pkl', 'rb') as f:
    generic_input['thermal_parameters'] = pickle.load(f)

investment_preferences = pd.Series([-0.0964, -0.152],
                                   index=pd.Index(['Single-family', 'Multi-family'], name='Housing type'),
                                   name='Housing type')
subsidy_preferences = 0.167
subsidy_loan_preferences = 0.473
bill_saving_preferences = pd.read_csv('project/input/bill_saving_preferences.csv', index_col=[0, 1]).squeeze('columns')
inertia = 0.8299

generic_input['preferences'] = {'investment': investment_preferences, 'subsidy': subsidy_preferences,
                                'bill_saved': bill_saving_preferences, 'inertia': inertia}

generic_input['performance_insulation'] = {'Wall': round(1 / 3.7, 1), 'Floor': round(1 / 3, 1), 'Roof': round(1 / 6, 1),
                                           'Windows': 1.5}

generic_input['surface'] = pd.read_csv('project/input/surface.csv', index_col=[0, 1, 2]).squeeze('columns').rename(None)

generic_input['index'] = {'Income tenant': ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10'],
                          'Income owner': ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10'],
                          'Occupation status': ['Owner-occupied', 'Privately rented', 'Social-housing'],
                          'Housing type': ['Single-family', 'Multi-family'],
                          'Performance': ['G', 'F', 'E', 'D', 'C', 'B', 'A'],
                          'Heating energy': ['Electricity', 'Natural gas', 'Oil fuel', 'Wood fuel'],
                          'Decision maker': ['Single-family - Owner-occupied', 'Single-family - Privately rented',
                                    'Single-family - Social-housing', 'Multi-family - Owner-occupied',
                                    'Multi-family - Privately rented', 'Multi-family - Social-housing',
                                    ]
                          }
