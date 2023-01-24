import time

import pandas as pd
from pandas import read_csv, concat, Series, Index, DataFrame
# imports from ResIRF
from project.model import config2inputs, initialize, stock_turnover, calibration_res_irf
from project.read_input import PublicPolicy
from project.utils import make_plot
from multiprocessing import Pool
import os
from pickle import dump, load
import json

from pathlib import Path


def ini_res_irf(path=None, logger=None, config=None, export_calibration=None, import_calibration=None, cost_factor=1):
    """Initialize and calibrate Res-IRF.

    Parameters
    ----------
    path
    logger
    config
    export_calibration
    import_calibration

    Returns
    -------
    AgentBuildings
        Building stock object initialize.
    """
    # creating object to calibrate with calibration
    if config is not None:
        with open(config) as file:
            config = json.load(file).get('Reference')

    if path is None:
        path = os.path.join('output', 'ResIRF')
    if not os.path.isdir(path):
        os.mkdir(path)

    if import_calibration is not None and os.path.isfile(import_calibration):
        with open(import_calibration, "rb") as file:
            calibration = load(file)
    else:
        calibration = calibration_res_irf(os.path.join(path, 'calibration'), config=config, cost_factor=cost_factor)

    if export_calibration is not None:
        export_calibration = Path(export_calibration)
        parent = export_calibration.parent.absolute()
        if not os.path.isdir(parent):
            print('Creation of calibration folder here: {}'.format(parent))
            os.mkdir(parent)
        with open(export_calibration, "wb") as file:
            dump(calibration, file)

    inputs, stock, year, policies_heater, policies_insulation, taxes = config2inputs(config)
    buildings, energy_prices, taxes, post_inputs, cost_heater, ms_heater, cost_insulation, calibration_intensive, calibration_renovation, flow_built, financing_cost = initialize(
        inputs, stock, year, taxes, path=path, config=config, logger=logger)
    cost_insulation *= cost_factor

    buildings.calibration_exogenous(**calibration)

    return buildings, energy_prices, taxes, cost_heater, cost_insulation, flow_built, post_inputs, policies_heater, policies_insulation


def select_output(output):
    """Select output

    Parameters
    ----------
    output: DataFrame
        Res-IRF  output.

    Returns
    -------
    DataFrame
        Selected rows
    """

    energy = ['Electricity', 'Natural gas', 'Oil fuel', 'Wood fuel']
    heater_replacement = ['Electricity-Heat pump water',
                          'Electricity-Heat pump air',
                          'Electricity-Performance boiler',
                          'Natural gas-Performance boiler',
                          'Wood fuel-Performance boiler'
                          ]
    heater_stock = heater_replacement + ['Natural gas-Standard boiler',
                                         'Wood fuel-Standard boiler',
                                         'Oil fuel-Standard boiler',
                                         'Oil fuel-Performance boiler',
                                         ]

    variables = list()
    variables += ['Consumption {} (TWh)'.format(i) for i in energy]
    variables += [
        'Investment heater (Billion euro)',
        'Investment insulation (Billion euro)',
        'Investment total (Billion euro)',
        'Subsidies heater (Billion euro)',
        'Subsidies insulation (Billion euro)',
        'Subsidies total (Billion euro)',
        'Health cost (Billion euro)',
        'Energy poverty (Million)',
        'Heating intensity (%)',
        'Emission (MtCO2)',
        'Cost rebound (Billion euro)',
        'Consumption saving renovation (TWh)',
        'Consumption saving heater (TWh)',
        'Investment insulation / saving (euro / kWh.year)',
        'Investment heater / saving (euro / kWh.year)'
    ]
    variables += ['Replacement heater {} (Thousand households)'.format(i) for i in heater_replacement]
    variables += ['Stock {} (Thousand households)'.format(i) for i in heater_stock]

    variables = [v for v in variables if v in output.index]
    return output.loc[variables]


def simu_res_irf(buildings, sub_heater, sub_insulation, start, end, energy_prices, taxes, cost_heater, cost_insulation,
                 flow_built, post_inputs, policies_heater, policies_insulation, climate=2006, smooth=False, efficiency_hour=False,
                 output_consumption=False, full_output=True):

    # initialize policies
    if sub_heater is not None and sub_heater != 0:
        sub_heater = Series([sub_heater, sub_heater],
                            index=Index(['Electricity-Heat pump water', 'Electricity-Heat pump air'],
                                        name='Heating system final'))
        policies_heater.append(PublicPolicy('sub_heater_optim', start, end, sub_heater, 'subsidy_ad_volarem',
                                            gest='heater', by='columns'))  # heating policy during considered years

    if sub_insulation is not None and sub_insulation != 0:
        policies_insulation.append(
            PublicPolicy('sub_insulation_optim', start, end, sub_insulation, 'subsidy_ad_volarem',
                         gest='insulation'))  # insulation policy during considered years

    output, consumption, prices = dict(), None, None
    for year in range(start, end):
        prices = energy_prices.loc[year, :]
        f_built = flow_built.loc[:, year]
        p_heater = [p for p in policies_heater if (year >= p.start) and (year < p.end)]
        p_insulation = [p for p in policies_insulation if (year >= p.start) and (year < p.end)]

        buildings, _, o = stock_turnover(buildings, prices, taxes, cost_heater, cost_insulation, p_heater,
                                         p_insulation, f_built, year, post_inputs)

        # o.to_csv(os.path.join(buildings.path, 'output.csv'))
        if full_output is False:
            output.update({year: select_output(o)})
        else:
            output.update({year: o})

    if output_consumption is True:
        buildings.logger.info('Calculating hourly consumption')
        consumption = buildings.consumption_total(prices=prices, freq='hour', standard=False, climate=climate,
                                                  smooth=smooth, efficiency_hour=efficiency_hour)

    output = DataFrame(output)
    buildings.logger.info('End of Res-IRF simulation')
    return output, consumption


def run_multi_simu(buildings, sub_heater, start, end, energy_prices, taxes, cost_heater, cost_insulation,
                   flow_built, post_inputs, policies_heater, policies_insulation):

    sub_insulation = [i / 10 for i in range(0, 11)]
    _len = len(sub_insulation)
    sub_heater = [sub_heater] * _len
    start = [start] * _len
    end = [end] * _len
    energy_prices = [energy_prices] * _len
    taxes = [taxes] * _len
    cost_heater = [cost_heater] * _len
    cost_insulation = [cost_insulation] * _len
    flow_built = [flow_built] * _len
    post_inputs = [post_inputs] * _len
    policies_heater = [policies_heater] * _len
    policies_insulation = [policies_insulation] * _len
    buildings = [deepcopy(buildings)] * _len

    list_argument = list(zip(deepcopy(buildings), deepcopy(sub_heater), deepcopy(sub_insulation), start, end, energy_prices, taxes,
                             cost_heater, cost_insulation, flow_built, post_inputs, policies_heater,
                             policies_insulation))

    with Pool() as pool:
        results = pool.starmap(simu_res_irf, list_argument)

    result = {list_argument[i][2]: results[i][0].squeeze() for i in range(len(results))}
    result = DataFrame(result)
    return result


if __name__ == '__main__':
    from copy import deepcopy
    # first time
    name = 'calibration'
    calibration_threshold = True
    config = 'project/input/config/test/config_optim.json'
    if calibration_threshold is True:
        name = '{}_threshold'.format(name)
        config = 'project/input/config/test/config_optim_threshold.json'

    _export_calibration = os.path.join('project', 'output', 'calibration', '{}.pkl'.format(name))
    _import_calibration = os.path.join('project', 'output', 'calibration', '{}.pkl'.format(name))
    _path = os.path.join('project', 'output', 'ResIRF')
    _buildings, _energy_prices, _taxes, _cost_heater, _cost_insulation, _flow_built, _post_inputs, _p_heater, _p_insulation = ini_res_irf(
        path=_path,
        logger=None,
        config=config,
        import_calibration=None,
        export_calibration=_export_calibration)

    _sub_heater = 1
    _result = run_multi_simu(_buildings, _sub_heater, 2020, 2021, _energy_prices, _taxes, _cost_heater,
                             _cost_insulation, _flow_built, _post_inputs, _p_heater, _p_insulation)
    name = 'cost_efficiency_insulation.png'
    if calibration_threshold is True:
        name = 'cost_efficiency_insulation_threshold.png'
    make_plot(_result.loc['Investment insulation / saving (euro/kWh)', :], 'Investment insulation / saving (euro/kWh)',
              integer=False, save=os.path.join(_path, name))

    variables = ['Consumption saving renovation (TWh)',
                 'Investment insulation (euro/year)',
                 'Investment insulation / saving (euro/kWh)',
                 ]
    _result_diff = _result.loc[variables, :].diff(axis=1).dropna(axis=1, how='all')
    _result_diff.loc['Investment insulation / saving (euro/kWh)'] = _result_diff.loc['Investment insulation (euro/year)'] / _result_diff.loc['Consumption saving renovation (TWh)']
    name = 'marginal_cost_efficiency_insulation.png'
    if calibration_threshold is True:
        name = 'marginal_cost_efficiency_insulation_threshold.png'

    make_plot(_result_diff.loc['Investment insulation / saving (euro/kWh)', :],
              'Marginal investment insulation / saving (euro/kWh)',
              integer=False, save=os.path.join(_path, name))

    print('break')


    """
    timestep = 1
    _year = 2020

    _sub_heater = 1
    _sub_insulation = 0

    _concat_output = DataFrame()
    for _year in range(2020, 2021):
        _start = _year
        _end = _year + timestep

        _output, _consumption = simu_res_irf(_buildings, _sub_heater, _sub_insulation, _start, _end, _energy_prices, _taxes,
                                             _cost_heater, _cost_insulation, _flow_built, _post_inputs, _p_heater,
                                             _p_insulation, climate=2006, smooth=False, efficiency_hour=False,
                                             output_consumption=False)
        _concat_output = concat((_concat_output, _output), axis=1)

    _concat_output.to_csv(os.path.join(_buildings.path, 'output.csv'))

    """


