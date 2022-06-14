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
generic_input['available_income'] = 1421 * 10**9
generic_input['price_index'] = 1

generic_input['income'] = pd.Series([10030, 15910, 19730, 23680, 28150, 33320, 39260, 46450, 57230, 102880],
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

generic_input['consumption_ini'] = pd.Series([40, 132, 42, 79],
                                             index=pd.Index(['Electricity', 'Natural gas', 'Oil fuel', 'Wood fuel'],
                                                            name='Heating energy'))
consumption_hist = pd.read_csv('project/input/revealed_data/hist_consumption.csv', index_col=[0],
                                                header=[0])
generic_input['consumption_hist'] = {k: pd.Series(item, name='Historic') for k, item in
                                     consumption_hist.to_dict().items()}

"""generic_input['consumption_hist'] = {k: pd.Series(item).set_axis(pd.Series(item).index.astype(int)).rename('Historic')
                                     for k, item in generic_input['consumption_hist'].T.to_dict().items()}"""

generic_input['consumption_total_hist'] = pd.read_csv('project/input/revealed_data/hist_consumption_total.csv',
                                                      index_col=[0], header=None).squeeze().rename('Historic')

generic_input['consumption_total_objectives'] = pd.Series([214, 181, 151], index=[2023, 2030, 2050], name='Objectives')

generic_input['emissions_total_objectives'] = pd.Series([25.5, 0], index=[2030, 2050], name='Objectives')

generic_input['low_eff_objectives'] = pd.Series([0], index=[2050], name='Objectives')

generic_input['retrofit_hist'] = pd.read_csv('project/input/revealed_data/hist_retrofit.csv', index_col=[0],
                                             header=[0])
generic_input['retrofit_hist'] = {k: pd.DataFrame({2019: item}).T / 10**3 for k, item in
                                  generic_input['retrofit_hist'].T.to_dict().items()}

generic_input['stock_ini'] = 29037000

"""with open('project/input/parameters_thermal_module.pkl', 'rb') as f:
    generic_input['thermal_parameters'] = pickle.load(f)"""

param = {}
param['certificate_bounds'] = {'A': (0, 39.59394228117988),
                               'B': (39.59394228117988, 71.8798550383082),
                               'C': (71.8798550383082, 119.92500210928785),
                               'D': (119.92500210928785, 183.81313518374748),
                               'E': (183.81313518374748, 264.83029117086653),
                               'F': (264.83029117086653, 394.25890705145184),
                               'G': (394.25890705145184, 1000)}
param['ratio_surface'] = pd.DataFrame(
    [[0.975002, 0.748213, 0.762826, 0.162041], [0.905083, 0.708620, 0.748979, 0.151731]],
    index=pd.Index(['Single-family', 'Multi-family'], name='Housing type'),
    columns=['Wall', 'Floor', 'Roof', 'Windows'])

param['coefficient'] = pd.Series([0.891000, 0.791331], index=pd.Index(['Single-family', 'Multi-family'], name='Housing type'))
generic_input['thermal_parameters'] = param


investment_preferences_heater = pd.Series([-0.0964, -0.152],
                                          index=pd.Index(['Single-family', 'Multi-family'], name='Housing type'),
                                          name='Housing type')

investment_preferences_insulation = pd.Series([-0.0646, -0.151],
                                              index=pd.Index(['Single-family', 'Multi-family'], name='Housing type'),
                                              name='Housing type')

subsidy_preferences_heater = 0.167

subsidy_preferences_insulation = pd.Series([0.0486, 0.236],
                                           index=pd.Index(['Single-family', 'Multi-family'], name='Housing type'),
                                           name='Housing type')


subsidy_loan_preferences_heater = 0.473
subsidy_loan_preferences_insulation = 0.343
bill_saving_preferences = pd.read_csv('project/input/bill_saving_preferences.csv', index_col=[0, 1])
inertia = 0.8299


generic_input['preferences'] = {}
generic_input['preferences']['heater'] = {'investment': investment_preferences_heater,
                                          'subsidy': subsidy_preferences_heater,
                                          'bill_saved': bill_saving_preferences.loc[:, 'Heater'],
                                          'inertia': inertia}


generic_input['preferences']['insulation'] = {'investment': investment_preferences_insulation,
                                              'subsidy': subsidy_preferences_insulation,
                                              'bill_saved': bill_saving_preferences.loc[:, 'Insulation'],
                                              'zero_interest_loan': subsidy_loan_preferences_insulation
                                              }

generic_input['performance_insulation'] = {'Wall': round(1 / 3.7, 1), 'Floor': round(1 / 3, 1), 'Roof': round(1 / 6, 1),
                                           'Windows': 1.5}

generic_input['surface'] = pd.read_csv('project/input/surface.csv', index_col=[0, 1, 2]).squeeze('columns').rename(None)

generic_input['index'] = {'Income tenant': ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10'],
                          'Income owner': ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10'],
                          'Occupation status': ['Owner-occupied', 'Privately rented', 'Social-housing'],
                          'Housing type': ['Single-family', 'Multi-family'],
                          'Performance': ['G', 'F', 'E', 'D', 'C', 'B', 'A'],
                          'Heating energy': ['Electricity', 'Natural gas', 'Oil fuel', 'Wood fuel'],
                          'Decision maker': ['Single-family - Owner-occupied', 'Multi-family - Owner-occupied',
                                             'Single-family - Privately rented', 'Multi-family - Privately rented',
                                             'Single-family - Social-housing', 'Multi-family - Social-housing'
                                             ],
                          'Insulation': ['Wall', 'Floor', 'Roof', 'Windows']
                          }

colors = {
    "Owner-occupied": "lightcoral",
    "Privately rented": "chocolate",
    "Social-housing": "orange",
    "Single-family": "brown",
    "Multi-family": "darkolivegreen",
    "G": "black",
    "F": "dimgrey",
    "E": "grey",
    "D": "darkgrey",
    "C": "darkgreen",
    "B": "forestgreen",
    "A": "limegreen",
    "D1": "black",
    "D2": "maroon",
    "D3": "darkred",
    "D4": "brown",
    "D5": "firebrick",
    "D6": "orangered",
    "D7": "tomato",
    "D8": "lightcoral",
    "D9": "lightsalmon",
    "D10": "darksalmon",
    "Electricity": "darkorange",
    "Natural gas": "slategrey",
    "Oil fuel": "black",
    "Wood fuel": "saddlebrown",
    "VTA": "grey",
    "Taxes expenditure": "darkorange",
    "Subsidies heater": "orangered",
    "Subsidies insulation": "darksalmon",
    "Reduced tax": "darkolivegreen",
    "Cee": "tomato",
    "Cite": "blue",
    "Zero interest loan": "darkred",
    "Over cap": "grey",
    "Carbon tax": "rebeccapurple",
    "Mpr": "darkmagenta",
    'Sub ad volarem': "darkorange",
    "Sub merit": "slategrey",
    "Existing": "tomato",
    "Construction": "grey"
}

generic_input['colors'] = colors
