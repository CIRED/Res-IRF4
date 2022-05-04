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
import os

from building import AgentBuildings
from input.param import generic_input
from read_input import read_stock, read_policies, read_exogenous, read_revealed, parse_parameters
from write_output import parse_output, grouped_output

# TODO: zero-interest loan, policy analysis


def res_irf(config, path):
    """Res-IRF model.

    Parameters
    ----------
    config: dict
        Scenario-specific input
    path: str
        Scenario-specific output path

    Returns
    -------
    str
        Scenario name
    pd.DataFrame
        Detailed results
    """
    os.mkdir(path)

    stock, year = read_stock(config)
    policies_heater, policies_insulation, taxes = read_policies(config)
    param, summary_param = parse_parameters(config, generic_input, stock)
    energy_prices, cost_heater, cost_insulation = read_exogenous(config)
    efficiency, choice_insulation, ms_heater, restrict_heater, choice_heater, ms_extensive, ms_intensive = read_revealed(config)

    total_taxes = pd.DataFrame(0, index=energy_prices.index, columns=energy_prices.columns)
    for t in taxes:
        total_taxes = total_taxes.add(t.value, fill_value=0)
    energy_prices = energy_prices.add(total_taxes, fill_value=0)

    if config['prices_constant']:
        energy_prices = pd.concat([energy_prices.loc[year, :]] * energy_prices.shape[0], keys=energy_prices.index,
                                  axis=1).T

    print('Calibration {}'.format(year))
    buildings = AgentBuildings(stock, param['surface'], param['thermal_parameters'], efficiency, param['income'],
                               param['consumption_ini'], path, param['preferences'],
                               restrict_heater, ms_heater, choice_insulation, param['performance_insulation'],
                               year=year, demolition_rate=param['demolition_rate'],
                               data_calibration=param['data_ceren'])

    for year in range(config['start'], config['end']):
        print('Run {}'.format(year))
        buildings.calculate(energy_prices.loc[year, :], taxes)

        if False:
            flow_retrofit = buildings.flow_retrofit(energy_prices.loc[year, :], cost_heater, ms_heater, cost_insulation,
                                                    ms_intensive, ms_extensive,
                                                    [p for p in policies_heater if (year >= p.start) and (year < p.end)],
                                                    [p for p in policies_insulation if (year >= p.start) and (year < p.end)])
        else:
            flow_retrofit = buildings.flow_retrofit_simplified(cost_heater)

        buildings.year = year
        buildings.add_flows([flow_retrofit, - buildings.flow_demolition(), param['flow_built'].loc[:, year]])
    buildings.calculate(energy_prices.loc[year, :], taxes)

    stock, output = parse_output(buildings, param)
    output.round(2).to_csv(os.path.join(path, 'output.csv'))
    stock.round(2).to_csv(os.path.join(path, 'stock.csv'))

    return os.path.basename(os.path.normpath(path)), output


if __name__ == '__main__':

    import logging
    import os
    import json
    from time import time
    from multiprocessing import Pool
    from datetime import datetime

    start = time()

    if not os.path.isdir('project/output'):
        os.mkdir('project/output')

    with open('project/input/config.json') as file:
        configuration = json.load(file)

    folder = os.path.join('project/output', datetime.today().strftime('%Y%m%d_%H%M%S'))
    os.mkdir(folder)

    logging.debug('Launching processes')
    processes = list()
    with Pool() as pool:
        result = pool.starmap(res_irf,
                              zip(configuration.values(), [os.path.join(folder, n) for n in configuration.keys()]))

    result = {i[0]: i[1] for i in result}
    stocks = {i[0]: i[2] for i in result}

    logging.debug('Parsing results')
    grouped_output(result, stocks, folder)

    logging.debug('Run time: {:,.0f} minutes.'.format((time() - start) / 60))

