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
import copy
import logging
import json
from time import time
from multiprocessing import Pool
from datetime import datetime
import re
import argparse

from building import AgentBuildings
from input.param import generic_input
from read_input import read_stock, read_policies, read_exogenous, read_revealed, parse_parameters
from write_output import parse_output, grouped_output


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
    log_formatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename=os.path.join(path, 'log.log'),
                        level=logging.DEBUG,
                        format=log_formatter,
                        filemode='a')
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('matplotlib.axes').disabled = True
    root_logger = logging.getLogger("")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_formatter))
    root_logger.addHandler(console_handler)

    try:
        stock, year = read_stock(config)
        policies_heater, policies_insulation, taxes = read_policies(config)
        param, summary_param = parse_parameters(config, generic_input, stock)
        energy_prices, energy_taxes, cost_heater, cost_insulation = read_exogenous(config)
        efficiency, choice_insulation, ms_heater, restrict_heater, choice_heater, renovation_rate_ini, ms_intensive = read_revealed(config)

        if config['prices_constant']:
            energy_prices = pd.concat([energy_prices.loc[year, :]] * energy_prices.shape[0], keys=energy_prices.index,
                                      axis=1).T

        total_taxes = pd.DataFrame(0, index=energy_prices.index, columns=energy_prices.columns)
        for t in taxes:
            total_taxes = total_taxes.add(t.value, fill_value=0)

        if energy_taxes is not None:
            total_taxes = total_taxes.add(energy_taxes, fill_value=0)

        if config['taxes_constant']:
            total_taxes = pd.concat([total_taxes.loc[year, :]] * total_taxes.shape[0], keys=total_taxes.index,
                                    axis=1).T

        energy_vta = energy_prices * generic_input['vta_energy_prices']
        total_taxes += energy_vta

        energy_prices = energy_prices.add(total_taxes, fill_value=0)
        param['energy_prices'] = energy_prices

        t = total_taxes.copy()
        t.columns = t.columns.map(lambda x:  'Taxes {} (euro/kWh)'.format(x))
        temp = energy_prices.copy()
        temp.columns = temp.columns.map(lambda x:  'Prices {} (euro/kWh)'.format(x))
        pd.concat((summary_param, t, temp), axis=1).to_csv(os.path.join(path, 'input.csv'))

        logging.debug('Calibration {}'.format(year))
        buildings = AgentBuildings(stock, param['surface'], generic_input['ratio_surface'], efficiency, param['income'],
                                   param['consumption_ini'], path, param['preferences'],
                                   restrict_heater, ms_heater, choice_insulation, param['performance_insulation'],
                                   year=year, demolition_rate=param['demolition_rate'],
                                   data_calibration=param['data_ceren'], endogenous=config['endogenous'],
                                   number_exogenous=config['exogenous_detailed']['number'])

        buildings.calculate(energy_prices.loc[year, :], taxes)
        for year in range(config['start'] + 1, config['end']):
            logging.debug('Run {}'.format(year))

            buildings.year = year
            buildings.add_flows([- buildings.flow_demolition()])
            flow_retrofit = buildings.flow_retrofit(energy_prices.loc[year, :], cost_heater, ms_heater, cost_insulation,
                                                    ms_intensive, renovation_rate_ini,
                                                    [p for p in policies_heater if (year >= p.start) and (year < p.end)],
                                                    [p for p in policies_insulation if (year >= p.start) and (year < p.end)],
                                                    config['target_freeriders'],
                                                    supply_constraint=config['supply_constraint'])
            buildings.add_flows([flow_retrofit, param['flow_built'].loc[:, year]])
            buildings.calculate(energy_prices.loc[year, :], taxes)

        logging.debug('Writing output')
        stock, output = parse_output(buildings, param)
        output.round(2).to_csv(os.path.join(path, 'output.csv'))
        stock.round(2).to_csv(os.path.join(path, 'stock.csv'))

        return os.path.basename(os.path.normpath(path)), output, stock

    except Exception as e:
        root_logger.exception(e)


def run(path=None):

    start = time()

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='project/input/config.json', help='path config file')
    parser.add_argument('-d', '--directory', default='project/input/config/policies', help='path config directory')
    parser.add_argument('-y', '--year', default=None, help='end year')

    args = parser.parse_args()

    if not os.path.isdir('project/output'):
        os.mkdir('project/output')

    if path is None:
        path = args.config

    with open(path) as file:
        configuration = json.load(file)

    config_policies = None
    name_policy = ''
    if 'assessment' in configuration.keys():
        if configuration['assessment']['activated']:
            config_policies = configuration['assessment']
            config_policies = {key: item for key, item in config_policies.items() if item is not None}
            name_policy = configuration['assessment']['Policy name'] + '_'

            if config_policies['AP-1']:
                configuration['AP-1'] = copy.deepcopy(configuration['Reference'])
                configuration['AP-1']['policies'][config_policies['Policy name']]['end'] = configuration['Reference']['start'] + 2

            if config_policies['ZP']:
                configuration['ZP'] = copy.deepcopy(configuration['Reference'])
                for name, policy in configuration['ZP']['policies'].items():
                    policy['end'] = configuration['Reference']['start'] + 2
                    configuration['ZP']['policies'][name] = policy

            if config_policies['ZP'] and config_policies['ZP+1']:
                configuration['ZP+1'] = copy.deepcopy(configuration['ZP'])
                configuration['ZP+1']['policies'][config_policies['Policy name']]['end'] = configuration['Reference']['end']

            list_years = [int(re.search('20[0-9][0-9]', key)[0]) for key in config_policies.keys() if
                          re.search('20[0-9][0-9]', key)]
            for year in list_years:
                if config_policies['AP-{}'.format(year)] and year < configuration['Reference']['end']:
                    configuration['AP-{}'.format(year)] = copy.deepcopy(configuration['Reference'])
                    configuration['AP-{}'.format(year)]['policies'][config_policies['Policy name']]['end'] = year
                    configuration['AP-{}'.format(year)]['end'] = year + 1

        del configuration['assessment']
    
    config_sensitivity = None
    if 'sensitivity' in configuration.keys():
        if configuration['sensitivity']['activated']:
            name_policy = 'sensitivity_'
            config_sensitivity = configuration['sensitivity']
            if 'ZP' in config_sensitivity.keys():
                if config_sensitivity['ZP']:
                    configuration['ZP'] = copy.deepcopy(configuration['Reference'])
                    for name, policy in configuration['ZP']['policies'].items():
                        policy['end'] = configuration['Reference']['start'] + 2
                        configuration['ZP']['policies'][name] = policy
            if 'prices_constant' in config_sensitivity.keys():
                if config_sensitivity['prices_constant']:
                    configuration['PriceConstant'] = copy.deepcopy(configuration['Reference'])
                    configuration['PriceConstant']['prices_constant'] = True
            if 'prices_factor' in config_sensitivity.keys():
                values = config_sensitivity['prices_factor']
                if values:
                    if isinstance(values, float):
                        values = [values]
                    for v in values:
                        configuration['PriceFactor{:.0f}'.format(v * 100)] = copy.deepcopy(configuration['Reference'])
                        configuration['PriceFactor{:.0f}'.format(v * 100)]['prices_factor'] = v
            if 'cost_factor' in config_sensitivity.keys():
                values = config_sensitivity['cost_factor']
                if values:
                    if isinstance(values, float):
                        values = [values]
                    for v in values:
                        configuration['CostFactor{:.0f}'.format(v * 100)] = copy.deepcopy(configuration['Reference'])
                        configuration['CostFactor{:.0f}'.format(v * 100)]['cost_factor'] = v
            if 'renovation_rate_ini' in config_sensitivity.keys():
                if config_sensitivity['renovation_rate_ini']:
                    configuration['RetrofitIni'] = copy.deepcopy(configuration['Reference'])
                    configuration['RetrofitIni']['renovation_rate_ini'] = config_sensitivity['renovation_rate_ini']
            if 'target_freeriders' in config_sensitivity.keys():
                values = config_sensitivity['target_freeriders']
                if values:
                    if isinstance(values, float):
                        values = [values]
                    for v in values:
                        configuration['FreeridersIni{:.0f}'.format(v * 100)] = copy.deepcopy(configuration['Reference'])
                        configuration['FreeridersIni{:.0f}'.format(v * 100)]['target_freeriders'] = v
            if 'mpr_global_retrofit' in config_sensitivity.keys():
                if config_sensitivity['mpr_global_retrofit']:
                    configuration['MprGlobalRetrofit'] = copy.deepcopy(configuration['Reference'])
                    configuration['MprGlobalRetrofit']['policies']['mpr']['global_retrofit'] = "project/input/policies/mpr_global_retrofit.csv"
            if 'mpr_no_bonus' in config_sensitivity.keys():
                if config_sensitivity['mpr_no_bonus']:
                    configuration['MprNoBonus'] = copy.deepcopy(configuration['Reference'])
                    configuration['MprNoBonus']['policies']['mpr']['bonus'] = None
        del configuration['sensitivity']

    folder = os.path.join('project/output', '{}{}'.format(name_policy, datetime.today().strftime('%Y%m%d_%H%M%S')))
    os.mkdir(folder)

    log_formatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename=os.path.join(folder, 'log_run.log'), level=logging.DEBUG,
                        format=log_formatter)
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('matplotlib.axes').disabled = True
    logger = logging.getLogger("")
    """root_logger = logging.getLogger("")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_formatter))
    root_logger.addHandler(console_handler)"""

    if args.year:
        for key in configuration.keys():
            configuration[key]['end'] = int(args.year)

    logging.debug('Scenarios: {}'.format(', '.join(configuration.keys())))
    try:
        logging.debug('Launching processes')
        with Pool() as pool:
            results = pool.starmap(res_irf,
                                   zip(configuration.values(), [os.path.join(folder, n) for n in configuration.keys()]))
        result = {i[0]: i[1] for i in results}
        stocks = {i[0]: i[2] for i in results}

        logging.debug('Parsing results')
        grouped_output(result, stocks, folder, config_policies, config_sensitivity)

        logging.debug('Run time: {:,.0f} minutes.'.format((time() - start) / 60))
    except Exception as e:
        logger.exception(e)


if __name__ == '__main__':
    run()
