import time

import pandas as pd
from pandas import read_csv, concat, Series, Index, DataFrame
# imports from ResIRF
from project.model import config2inputs, initialize, stock_turnover, calibration_res_irf
from project.read_input import PublicPolicy
from multiprocessing import Pool
import os
from pickle import dump, load
import json


def ini_res_irf(path=None, logger=None, config=None, export_calibration=None, import_calibration=None):
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
        calibration = calibration_res_irf(os.path.join(path, 'calibration'), config=config)

    if export_calibration is not None:
        with open(export_calibration, "wb") as file:
            dump(calibration, file)

    inputs, stock, year, policies_heater, policies_insulation, taxes = config2inputs(config)
    buildings, energy_prices, taxes, post_inputs, cost_heater, ms_heater, cost_insulation, calibration_intensive, calibration_renovation, flow_built, financing_cost = initialize(
        inputs, stock, year, taxes, path=path, config=config, logger=logger)
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

    # setting output format
    buildings.full_output = full_output

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

        o.to_csv(os.path.join(buildings.path, 'output.csv'))
        output.update({year: select_output(o)})

    if output_consumption is True:
        buildings.logger.info('Calculating hourly consumption')
        consumption = buildings.consumption_total(prices=prices, freq='hour', standard=False, climate=climate,
                                                  smooth=smooth, efficiency_hour=efficiency_hour)

    output = DataFrame(output)
    buildings.logger.info('End of Res-IRF simulation')
    return output, consumption


if __name__ == '__main__':
    from copy import deepcopy
    # first time
    name = 'calibration'
    calibration_threshold = True
    if calibration_threshold is True:
        name = '{}_threshold'.format(name)

    _export_calibration = os.path.join('project', 'output', 'calibration', '{}.pkl'.format(name))
    _import_calibration = os.path.join('project', 'output', 'calibration', '{}.pkl'.format(name))

    _buildings, _energy_prices, _taxes, _cost_heater, _cost_insulation, _flow_built, _post_inputs, _p_heater, _p_insulation = ini_res_irf(
        path=os.path.join('project', 'output', 'ResIRF'),
        logger=None,
        config=os.path.join('project/input/config/test/config_optim_threshold.json'),
        import_calibration=None,
        export_calibration=_export_calibration)

    timestep = 1
    _year = 2020
    _start = _year
    _end = _year + timestep

    _sub_heater = 0.1
    _sub_insulation = 0.1

    _output, _consumption = simu_res_irf(_buildings, _sub_heater, _sub_insulation, _start, _end, _energy_prices, _taxes,
                                         _cost_heater, _cost_insulation, _flow_built, _post_inputs, _p_heater,
                                         _p_insulation, climate=2006, smooth=False, efficiency_hour=False,
                                         output_consumption=True)


    """_sub_insulation = [i / 10 for i in range(4, 8)]
    _len = len(_sub_insulation)
    _sub_heater = [0] * _len
    _start = [_start] * _len
    _end = [_end] * _len
    _energy_prices = [_energy_prices] * _len
    _taxes = [_taxes] * _len
    _cost_heater = [_cost_heater] * _len
    _cost_insulation = [_cost_insulation] * _len
    _flow_built = [_flow_built] * _len
    _post_inputs = [_post_inputs] * _len
    _p_heater = [_p_heater] * _len
    _p_insulation = [_p_insulation] * _len
    _buildings = [deepcopy(_buildings)] * _len

    list_argument = list(zip(_buildings, _sub_heater, _sub_insulation, _start, _end, _energy_prices, _taxes,
                             _cost_heater, _cost_insulation, _flow_built, _post_inputs, _p_heater,
                             _p_insulation))

    with Pool() as pool:

        results = pool.starmap(simu_res_irf, list_argument)

    test = {list_argument[i][2]: results[i][0].squeeze() for i in range(len(results))}
    test = DataFrame(test)"""
    print('break')



    """
        result = dict()
    for _sub_insulation in range(5, 8):
        
    list_argument = [(sub_heater, 0.5, 2020, 2021) for sub_heater in [0.1, 0.9]]

    with Pool(4) as pool:
        results = pool.starmap(run_resirf, list_argument)

    sub_heater = Series([i[0] for i in results], name='Sub heater')
    sub_insulation = Series([i[1] for i in results], name='Sub insulation')
    df = concat([Series(i[2]) for i in results], axis=1)
    df = concat((sub_heater, sub_insulation, df.T), axis=1).T
    df.to_csv('output/sensitivity.csv')
    
        if 'Investment insulation (Billion euro)' not in _output.index:
        _output['Investment insulation (Billion euro)'] = 0

    result.update({_sub_insulation: _output.loc['Investment insulation (Billion euro)']})
    print(Series(result))
    
    
    """