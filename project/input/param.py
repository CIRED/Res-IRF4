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
generic_input['available_income'] = 1421 * 10**9
generic_input['price_index'] = 1
generic_input['income'] = pd.Series([10030, 15910, 19730, 23680, 28150, 33320, 39260, 46450, 57230, 102880],
                                    index=pd.Index(['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10'],
                                                   name='Income'))
input_financing = dict()
input_financing['share_debt'] = (0.139, 1.15 * 10**-5)

input_financing['factor_saving_rate'] = pd.Series([2, 2, 2, 2, 1.6, 1.6, 1.3, 1.3, 1, 1],
                                                  index=pd.Index(
                                                      ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10'],
                                                      name='Income owner'))
input_financing['interest_rate'] = 0.04
input_financing['saving_rate'] = 0.025
generic_input['input_financing'] = input_financing
input_financing['duration'] = 10


generic_input['stock_ini'] = 29037000

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

generic_input['consumption_ini'] = pd.Series([39, 129, 40, 76],
                                             index=pd.Index(['Electricity', 'Natural gas', 'Oil fuel', 'Wood fuel'],
                                                            name='Heating energy'))
generic_input['vta_energy_prices'] = pd.Series(
    {'Electricity': 0.15, 'Natural gas': 0.15, 'Oil fuel': 0.2, 'Wood fuel': 0.1})


generic_input['ratio_surface'] = pd.DataFrame(
    [[1.42, 0.75, 0.77, 0.17], [0.78, 0.28, 0.29, 0.19]],
    index=pd.Index(['Single-family', 'Multi-family'], name='Housing type'),
    columns=['Wall', 'Floor', 'Roof', 'Windows'])

subsidy_preferences_heater = 0.167
bill_saving_preferences = get_pandas('project/input/bill_saving_preferences.csv', lambda x: pd.read_csv(x, index_col=[0, 1]))
# subsidy_loan_preferences_heater, subsidy_loan_preferences_insulation = 0.473, 0.343

inertia = 0.8299

preferences_by_housing_type = False
if preferences_by_housing_type:
    investment_preferences_heater = pd.Series([-0.0964, -0.152],
                                              index=pd.Index(['Single-family', 'Multi-family'], name='Housing type'),
                                              name='Housing type')

    investment_preferences_insulation = pd.Series([-0.0646, -0.151],
                                                  index=pd.Index(['Single-family', 'Multi-family'], name='Housing type'),
                                                  name='Housing type')

    subsidy_preferences_insulation = pd.Series([0.0486, 0.236],
                                               index=pd.Index(['Single-family', 'Multi-family'], name='Housing type'),
                                               name='Housing type')
else:
    investment_preferences_heater = -0.0964
    investment_preferences_insulation = -0.0646
    subsidy_preferences_insulation = 0.0486
    bill_saving_preferences = bill_saving_preferences.xs('Single-family', level='Housing type')

generic_input['preferences'] = {}
generic_input['preferences']['heater'] = {'cost': investment_preferences_heater,
                                          'subsidy': subsidy_preferences_heater,
                                          'bill_saved': bill_saving_preferences.loc[:, 'Heater'],
                                          'inertia': inertia}

"""
generic_input['preferences']['insulation'] = {'investment': investment_preferences_insulation,
                                              'subsidy': subsidy_preferences_insulation,
                                              'bill_saved': bill_saving_preferences.loc[:, 'Insulation'],
                                              'zero_interest_loan': subsidy_loan_preferences_insulation
                                              }
"""

generic_input['preferences']['insulation'] = {'cost': investment_preferences_heater,
                                              'subsidy': subsidy_preferences_heater,
                                              'bill_saved': bill_saving_preferences.loc[:, 'Heater'],
                                              }

generic_input['performance_insulation'] = {'Wall': 0.1,
                                           'Floor': round(1 / 3.6, 1),
                                           'Roof': round(1 / 8.6, 1),
                                           'Windows': 1.3}

generic_input['surface'] = get_pandas('project/input/surface.csv', lambda x: pd.read_csv(x, index_col=[0, 1, 2]).squeeze().rename(None))



"""generic_input['supply'] = {
    'factor_max': 30,
    'factor_min': 0.8,
    'factor_norm': 1,
    'utilization_norm': 0.8
}"""



