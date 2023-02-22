import time

import pandas as pd
from pandas import read_csv, concat, Series, Index, DataFrame
# imports from ResIRF
from project.model import config2inputs, initialize, stock_turnover, calibration_res_irf
from project.read_input import PublicPolicy
from project.utils import make_plot, make_plots, get_size, size_locals
from multiprocessing import Pool
import os
from pickle import dump, load
import json
import psutil
from pathlib import Path

from copy import deepcopy

ENERGY = ['Electricity', 'Natural gas', 'Oil fuel', 'Wood fuel']

CONFIG_TEST = 'project/config/coupling/config_coupling_test.json'
CONFIG_THRESHOLD_TEST = 'project/config/coupling/config_coupling_simple_threshold.json'


def ini_res_irf(path=None, config=None, climate=2006):
    """Initialize and calibrate Res-IRF.

    Parameters
    ----------
    path
    config: str

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
        try:
            os.mkdir(path)
        except:
            pass

    inputs, stock, year, policies_heater, policies_insulation, taxes = config2inputs(config)
    buildings, energy_prices, taxes, post_inputs, cost_heater, lifetime_heater, ms_heater, cost_insulation, calibration_intensive, calibration_renovation, demolition_rate, flow_built, financing_cost, technical_progress, consumption_ini = initialize(
        inputs, stock, year, taxes, path=path, config=config)


    # calibration
    if config.get('calibration'):
        with open(config['calibration'], "rb") as file:
            calibration = load(file)
    else:
        calibration = calibration_res_irf(path, config=config)

    buildings.calibration_exogenous(**calibration)

    output = pd.DataFrame()
    _, o = buildings.parse_output_run(energy_prices.loc[buildings.first_year, :], post_inputs)
    output = pd.concat((output, o), axis=1)
    year = 2019
    prices = energy_prices.loc[year, :]
    p_heater = [p for p in policies_heater if (year >= p.start) and (year < p.end)]
    p_insulation = [p for p in policies_insulation if (year >= p.start) and (year < p.end)]
    f_built = flow_built.loc[:, year]

    buildings, _, o = stock_turnover(buildings, prices, taxes, cost_heater, lifetime_heater,
                                     cost_insulation, p_heater,
                                     p_insulation, f_built, year, post_inputs,
                                     ms_heater=ms_heater, financing_cost=financing_cost,
                                     climate=climate)

    output = pd.concat((output, o), axis=1)
    output.to_csv(os.path.join(buildings.path, 'output_ini.csv'))

    return buildings, energy_prices, taxes, cost_heater, cost_insulation, lifetime_heater, demolition_rate, flow_built, post_inputs, policies_heater, policies_insulation, technical_progress, financing_cost


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
    consumption_energy = ['Consumption {} climate (TWh)'.format(i) for i in energy]
    rebound_energy = ['Rebound {} (TWh)'.format(i) for i in energy]

    variables = list()
    variables += consumption_energy
    variables += rebound_energy

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

    if sub_design == 'natural_gas':
        sub_insulation = pd.Series([0, 0, 0, sub_insulation, sub_insulation, 0, 0, 0, 0],
                                   index=energy_index)

    if sub_design == 'fossil':
        sub_insulation = pd.Series([0, 0, 0, sub_insulation, sub_insulation, sub_insulation, sub_insulation, 0, 0],
                                   index=energy_index)

    if sub_design == 'electricity':
        sub_insulation = pd.Series([sub_insulation, sub_insulation, sub_insulation, 0, 0, 0, 0, 0, 0],
                                   index=energy_index)

    if sub_design == 'global_renovation':
        target = 'global_renovation'

    if sub_design == 'global_renovation_low_income':
        target = 'global_renovation_low_income'

    if sub_design == 'best_efficiency':
        target = 'best_efficiency'

    if sub_design == 'best_efficiency_fg':
        target = 'best_efficiency_fg'

    if sub_design == 'global_renovation_fg':
        target = 'global_renovation_fg'

    if sub_design == 'global_renovation_fge':
        target = 'global_renovation_fge'

    if sub_design == 'efficiency_100':
        target = 'efficiency_100'

    policy = PublicPolicy('sub_insulation_optim', start, end, sub_insulation, 'subsidy_ad_valorem',
                          gest='insulation', target=target)

    return policy


def simu_res_irf(buildings, sub_heater, sub_insulation, start, end, energy_prices, taxes, cost_heater, cost_insulation,
                 lifetime_heater, flow_built, post_inputs, policies_heater, policies_insulation, sub_design, financing_cost,
                 climate=2006, smooth=False, efficiency_hour=False, demolition_rate=None,
                 output_consumption=False, full_output=True, rebound=True, technical_progress=None,
                 ):
    # initialize policies
    if sub_heater is not None:
        sub_heater = Series([sub_heater, sub_heater],
                            index=Index(['Electricity-Heat pump water', 'Electricity-Heat pump air'],
                                        name='Heating system final'))
        policies_heater.append(PublicPolicy('sub_heater_optim', start, end, sub_heater, 'subsidy_ad_valorem',
                                            gest='heater', by='columns'))  # heating policy during considered years

    if sub_insulation is not None:
        policy = create_subsidies(sub_insulation, sub_design, start, end)
        policies_insulation.append(policy)  # insulation policy during considered years

    output, consumption, prices = dict(), None, None
    for year in range(start, end):
        prices = energy_prices.loc[year, :]
        f_built = flow_built.loc[:, year]
        p_heater = [p for p in policies_heater if (year >= p.start) and (year < p.end)]
        p_insulation = [p for p in policies_insulation if (year >= p.start) and (year < p.end)]

        if technical_progress is not None:
            if technical_progress.get('insulation') is not None:
                cost_insulation *= (1 + technical_progress['insulation'].loc[year])
            if technical_progress.get('heater') is not None:
                heat_pump = ['Electricity-Heat pump air', 'Electricity-Heat pump water']
                cost_heater.loc[heat_pump] *= (1 + technical_progress['heater'].loc[year])

        buildings, s, o = stock_turnover(buildings, prices, taxes, cost_heater, lifetime_heater,
                                         cost_insulation, p_heater,
                                         p_insulation, f_built, year, post_inputs,
                                         financing_cost=financing_cost,
                                         climate=climate, demolition_rate=demolition_rate)

        if full_output is False:
            output.update({year: select_output(o)})
        else:
            output.update({year: o})

    output = DataFrame(output)
    if output_consumption is True:
        buildings.logger.info('Calculating hourly consumption')

        consumption = buildings.consumption_total(prices=prices, freq='hour', standard=False, climate=climate,
                                                  smooth=smooth, efficiency_hour=efficiency_hour)
        if rebound is False:
            # TODO: only work if there at least two years
            consumption_energy = output.loc[['Consumption {} climate (TWh)'.format(i) for i in ENERGY], :].sum(axis=1).set_axis(ENERGY)
            rebound_energy = output.loc[['Rebound {} (TWh)'.format(i) for i in ENERGY], :].sum(axis=1).set_axis(ENERGY)
            rebound_energy.index = ENERGY
            rebound_factor = rebound_energy / consumption_energy
            consumption = (consumption.T * (1 - rebound_factor)).T

    buildings.logger.info('End of Res-IRF simulation')
    return output, consumption


def run_multi_simu(buildings, sub_heater, start, end, energy_prices, taxes, cost_heater, cost_insulation,
                   lifetime_heater, flow_built, post_inputs, policies_heater, policies_insulation, financing_cost, sub_design=None):

    sub_insulation = [i / 10 for i in range(0, 11, 5)]
    _len = len(sub_insulation)
    sub_heater = [sub_heater] * _len
    start = [start] * _len
    end = [end] * _len
    energy_prices = [energy_prices] * _len
    taxes = [taxes] * _len
    cost_heater = [cost_heater] * _len
    cost_insulation = [cost_insulation] * _len
    lifetime_heater = [lifetime_heater] * _len
    flow_built = [flow_built] * _len
    post_inputs = [post_inputs] * _len
    policies_heater = [policies_heater] * _len
    policies_insulation = [policies_insulation] * _len
    buildings = [deepcopy(buildings)] * _len
    sub_design = [sub_design] * _len
    financing_cost = [financing_cost] * _len

    list_argument = list(zip(deepcopy(buildings), deepcopy(sub_heater), deepcopy(sub_insulation), start, end, energy_prices, taxes,
                             cost_heater, cost_insulation, lifetime_heater, flow_built, post_inputs, policies_heater,
                             policies_insulation, sub_design, financing_cost))

    with Pool() as pool:
        results = pool.starmap(simu_res_irf, list_argument)

    result = {list_argument[i][2]: results[i][0].squeeze() for i in range(len(results))}
    result = DataFrame(result)
    return result


def test_design_subsidies():
    sub_design_list = {'Efficiency measure': 'best_efficiency',
                       'Efficiency measure FG': 'best_efficiency_fg',
                       'Global renovation': 'global_renovation',
                       'Global renovation FG': 'global_renovation_fg',
                       'Global renovation FGE': 'global_renovation_fge',
                       'Cost efficiency 100': 'efficiency_100',
                       'Uniform': None
                       }

    path = os.path.join('project', 'output', 'ResIRF')
    buildings, energy_prices, taxes, cost_heater, cost_insulation, lifetime_heater, demolition_rate, flow_built, post_inputs, p_heater, p_insulation, technical_progress, financing_cost = ini_res_irf(
        path=path,
        config=CONFIG_TEST)

    concat_result, concat_result_marginal = dict(), dict()
    for k, sub_design in sub_design_list.items():
        print(sub_design)

        sub_heater = 0
        result = run_multi_simu(buildings, sub_heater, 2020, 2021, energy_prices, taxes, cost_heater,
                                cost_insulation, lifetime_heater, flow_built, post_inputs, p_heater, p_insulation, financing_cost,
                                sub_design=sub_design)

        variables = ['Consumption saving insulation (TWh/year)',
                     'Annuities insulation (Billion euro/year)',
                     'Efficiency insulation (euro/kWh)'
                     ]
        result_diff = result.loc[variables, :].diff(axis=1).dropna(axis=1, how='all')
        result_diff.loc['Efficiency insulation (euro/kWh)'] = result_diff.loc['Annuities insulation (Billion euro/year)'] / \
                                                                        result_diff.loc['Consumption saving insulation (TWh/year)']

        concat_result.update({k: result.loc['Efficiency insulation (euro/kWh)', :]})
        concat_result_marginal.update({k: result_diff.loc['Efficiency insulation (euro/kWh)', :]})

    make_plots(concat_result, 'Efficiency insulation (euro/kWh)',
               save=os.path.join(path, 'cost_efficiency_insulation_comparison.png'),
               format_y=lambda y, _: '{:.2f}'.format(y)
               )
    make_plots(concat_result_marginal, 'Marginal efficiency insulation (euro/kWh)',
               save=os.path.join(path, 'marginal_cost_efficiency_insulation_comparison.png'),
               format_y=lambda y, _: '{:.2f}'.format(y)
               )


def run_simu(output_consumption=False, rebound=True, start=2020, end=2021,
             sub_design='global_renovation'):

    path = os.path.join('project', 'output', 'ResIRF')
    buildings, energy_prices, taxes, cost_heater, cost_insulation, lifetime_heater, demolition_rate, flow_built, post_inputs, p_heater, p_insulation, technical_progress, financing_cost = ini_res_irf(
        path=path,
        config=CONFIG_TEST)

    sub_heater = 0.5
    sub_insulation = 1

    concat_output = DataFrame()
    output, consumption = simu_res_irf(buildings, sub_heater, sub_insulation, start, end, energy_prices, taxes,
                                       cost_heater, cost_insulation, lifetime_heater, flow_built, post_inputs, p_heater,
                                       p_insulation, sub_design, financing_cost, climate=2006, smooth=False,
                                       efficiency_hour=True, demolition_rate=demolition_rate,
                                       output_consumption=output_consumption, full_output=True, rebound=rebound,
                                       technical_progress=technical_progress)

    concat_output = concat((concat_output, output), axis=1)
    concat_output.to_csv(os.path.join(buildings.path, 'output.csv'))


if __name__ == '__main__':
    # test_design_subsidies()
    run_simu(output_consumption=False, rebound=False, start=2020, end=2021,
             sub_design='efficiency_100')
