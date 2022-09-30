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

from pandas import DataFrame, Series, read_csv
from project.utils import get_pandas

resources_data = dict()

consumption_hist = get_pandas('project/input/resources_dir/hist_consumption.csv', lambda x: read_csv(x, index_col=[0], header=[0]))
resources_data['consumption_hist'] = {k: Series(item, name='Historic') for k, item in consumption_hist.to_dict().items()}

consumption_total_hist = get_pandas('project/input/resources_dir/hist_consumption_total.csv', lambda x: read_csv(x, index_col=[0], header=None).squeeze().rename('Historic'))
resources_data['consumption_total_hist'] = consumption_total_hist

resources_data['consumption_total_objectives'] = Series([207, 176, 146], index=[2028, 2030, 2050], name='Objectives')

resources_data['emissions_total_objectives'] = Series([23, 0], index=[2030, 2050], name='Objectives')

resources_data['low_eff_objectives'] = Series([0], index=[2050], name='Objectives')


resources_data['retrofit_hist'] = get_pandas('project/input/resources_dir/hist_retrofit.csv', lambda x: read_csv(x, index_col=[0], header=[0]))
resources_data['retrofit_hist'] = {k: DataFrame({2019: item}).T / 10 ** 3 for k, item in
                                   resources_data['retrofit_hist'].T.to_dict().items()}

retrofit_comparison = get_pandas('project/input/resources_dir/retrofit_comparison_resirf3.csv', lambda x: read_csv(x, index_col=[0], header=[0]))
resources_data['retrofit_comparison'] = retrofit_comparison

resources_data['public_policies_2019'] = DataFrame([1.88, 1.05, 0, 1.32, 0.56],
                                                   index=['Cee', 'Cite', 'Mpr', 'Reduced tax', 'Zero interest loan'],
                                                   columns=[2019])

calibration_data = get_pandas('project/input/resources_dir/data_ceren.csv', lambda x: read_csv(x, index_col=[0]).squeeze())
resources_data['data_calibration'] = calibration_data

resources_data['investment_per_renovating_houshold_decision_maker'] = {k: DataFrame([9100], index=['TREMI 2019'],
                                                                                      columns=[2019]).T for k in [
    'Single-family - {}'.format(i) for i in ['Owner-occupied', 'Privately rented', 'Social-housing']] + [
    'Multi-family - {}'.format(i) for i in ['Owner-occupied', 'Privately rented', 'Social-housing']]}

resources_data['investment_per_renovating_houshold_income_owner'] = {k:  DataFrame([9100], index=['TREMI 2019'], columns=[2019]).T for k in
                                  ['D{}'.format(i) for i in range(1, 11)]}

# CategoricalIndex(['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10'], ordered=True)
resources_data['index'] = {'Income tenant': ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10'],
                           'Income owner': ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10'],
                           'Occupation status': ['Owner-occupied', 'Privately rented', 'Social-housing'],
                           'Housing type': ['Single-family', 'Multi-family'],
                           'Performance': ['G', 'F', 'E', 'D', 'C', 'B', 'A'],
                           'Energy': ['Electricity', 'Natural gas', 'Oil fuel', 'Wood fuel'],
                           'Decision maker': ['Single-family - Owner-occupied', 'Multi-family - Owner-occupied',
                                              'Single-family - Privately rented', 'Multi-family - Privately rented',
                                              'Single-family - Social-housing', 'Multi-family - Social-housing'
                                              ],
                           'Insulation': ['Wall', 'Floor', 'Roof', 'Windows'],
                           'Heating system': ['Electricity-Heat pump', 'Electricity-Performance boiler',
                                              'Natural gas-Performance boiler', 'Oil fuel-Performance boiler',
                                              'Wood fuel-Performance boiler'],
                           'Count': [1, 2, 3, 4, 5]
                           }

colors = {
    "Owner-occupied": "lightcoral",
    "Privately rented": "chocolate",
    "Social-housing": "orange",
    "Single-family": "brown",
    "Multi-family": "darkolivegreen",
    "Single-family - Owner-occupied": "firebrick",
    "Multi-family - Owner-occupied": "salmon",
    "Single-family - Privately rented": "darkgreen",
    "Multi-family - Privately rented": "mediumseagreen",
    "Single-family - Social-housing": "darkorange",
    "Multi-family - Social-housing": "chocolate",
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
    "Electricity-Heat pump": "sandybrown",
    "Electricity-Performance boiler": "darkorange",
    "Natural gas-Performance boiler": "slategrey",
    "Natural gas-Standard boiler": "grey",
    "Oil fuel-Performance boiler": "black",
    "Oil fuel-Standard boiler": "black",
    "Wood fuel-Performance boiler": "saddlebrown",
    "Wood fuel-Standard boiler": "saddlebrown",
    "VTA": "grey",
    "Energy taxes": "blue",
    "Energy vta": "red",
    "Taxes expenditure": "darkorange",
    "Subsidies heater": "orangered",
    "Subsidies insulation": "darksalmon",
    "Reduced tax": "darkolivegreen",
    "Cee": "tomato",
    "Cee tax": "red",
    "Cite": "blue",
    "Zero interest loan": "darkred",
    "Over cap": "grey",
    "Carbon tax": "rebeccapurple",
    "Mpr": "darkmagenta",
    'Sub ad volarem': "darkorange",
    "Sub merit": "slategrey",
    "Existing": "tomato",
    "New": "lightgrey",
    "Renovation": "brown",
    "Construction": "dimgrey",
    'Investment': 'firebrick',
    'Embodied emission additional': 'darkgreen',
    'Cofp': 'grey',
    'Energy saving': 'darkorange',
    'Emission saving': 'forestgreen',
    'Well-being benefit': 'royalblue',
    'Health savings': 'blue',
    'Mortality reduction benefit': 'lightblue',
    'Social NPV': 'black',
    'Windows': 'royalblue',
    'Roof': 'darkorange',
    'Floor': 'grey',
    'Wall': 'darkslategrey'
}

resources_data['colors'] = colors
