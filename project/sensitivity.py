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
        'Consumption saving insulation (TWh)',
        'Consumption saving heater (TWh)',
        'Investment insulation / saving (euro/kWh)',
        'Investment heater / saving (euro/kWh)'
    ]
    variables += ['Replacement heater {} (Thousand households)'.format(i) for i in heater_replacement]
    variables += ['Stock {} (Thousand households)'.format(i) for i in heater_stock]

    variables = [v for v in variables if v in output.index]
    return output.loc[variables]


def create_subsidies(sub_insulation, sub_design, start, end):
    """

    Parameters
    ----------
    sub_insulation
    sub_design: {'very_low_income', 'low_income', 'wall', 'natural_gas', 'fossil', 'global_renovation',
    'global_renovation_low_income', 'mpr_serenite', 'bonus_best', 'bonus_worst'}
    start
    end

    Returns
    -------

    """

    low_income_index = pd.Index(['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10'], name='Income owner')
    energy_index = pd.Index(['Electricity-Heat pump water', 'Electricity-Heat pump air',
                             'Electricity-Performance boiler',
                             'Natural gas-Performance boiler', 'Natural gas-Standard boiler',
                             'Oil fuel-Performance boiler', 'Oil fuel-Standard boiler',
                             'Wood fuel-Performance boiler', 'Wood fuel-Standard boiler'], name='Heating system')

    target = None
    if sub_design == 'very_low_income':
        sub_insulation = pd.Series([sub_insulation, sub_insulation,
                                    0, 0, 0, 0, 0, 0, 0, 0],
                                   index=low_income_index)

    if sub_design == 'low_income':
        sub_insulation = pd.Series([sub_insulation, sub_insulation, sub_insulation, sub_insulation,
                                    0, 0, 0, 0, 0, 0],
                                   index=low_income_index)
    if sub_design == 'wall':
        sub_insulation = pd.DataFrame([[sub_insulation, 0, 0, 0] for _ in range(len(low_income_index))],
                                      columns=['Wall', 'Floor', 'Roof', 'Windows'],
                                      index=low_income_index)
    if sub_design == 'natural_gas':
        sub_insulation = pd.Series([0, 0, 0, sub_insulation, sub_insulation, 0, 0, 0, 0],
                                   index=energy_index)

    if sub_design == 'fossil':
        sub_insulation = pd.Series([0, 0, 0, sub_insulation, sub_insulation, sub_insulation, sub_insulation, 0, 0],
                                   index=energy_index)

    if sub_design == 'global_renovation':
        target = 'global_renovation'

    if sub_design == 'global_renovation_low_income':
        target = 'global_renovation_low_income'

    if sub_design == 'mpr_serenite':
        target = 'mpr_serenite'

    if sub_design == 'bonus_best':
        target = 'bonus_best'

    if sub_design == 'bonus_worst':
        target = 'bonus_worst'

    policy = PublicPolicy('sub_insulation_optim', start, end, sub_insulation, 'subsidy_ad_volarem',
                          gest='insulation', target=target)

    return policy


def simu_res_irf(buildings, sub_heater, sub_insulation, start, end, energy_prices, taxes, cost_heater, cost_insulation,
                 flow_built, post_inputs, policies_heater, policies_insulation, climate=2006, smooth=False, efficiency_hour=False,
                 output_consumption=False, full_output=True, sub_design='global_renovation'):

    # initialize policies
    if sub_heater is not None:
        sub_heater = Series([sub_heater, sub_heater],
                            index=Index(['Electricity-Heat pump water', 'Electricity-Heat pump air'],
                                        name='Heating system final'))
        policies_heater.append(PublicPolicy('sub_heater_optim', start, end, sub_heater, 'subsidy_ad_volarem',
                                            gest='heater', by='columns'))  # heating policy during considered years

    # , 'target_global', 'target_low_efficient', 'target_wall'
    if sub_insulation is not None:
        policy = create_subsidies(sub_insulation, sub_design, start, end)
        policies_insulation.append(policy)  # insulation policy during considered years

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
    calibration_threshold = False
    config = 'project/input/config/test/config_celia.json'
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
        import_calibration=_import_calibration,
        export_calibration=_export_calibration)

    """
    _sub_heater = 0
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

    """
    timestep = 1
    _year = 2020

    _sub_heater = 0.1
    _sub_insulation = 0.5

    _concat_output = DataFrame()
    for _year in range(2025, 2027):
        _start = _year
        _end = _year + timestep

        _output, _consumption = simu_res_irf(_buildings, _sub_heater, _sub_insulation, _start, _end, _energy_prices, _taxes,
                                             _cost_heater, _cost_insulation, _flow_built, _post_inputs, _p_heater,
                                             _p_insulation, climate=2006, smooth=False, efficiency_hour=True,
                                             output_consumption=False, full_output=False)
        _concat_output = concat((_concat_output, _output), axis=1)

    _concat_output.to_csv(os.path.join(_buildings.path, 'output.csv'))


