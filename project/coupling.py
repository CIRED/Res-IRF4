import pandas as pd
from pandas import read_csv, concat, Series, Index, DataFrame
from multiprocessing import Pool
import os
from pickle import dump, load
import json
from datetime import datetime
from copy import deepcopy

# imports from ResIRF
from project.model import config2inputs, initialize, stock_turnover, calibration_res_irf
from project.read_input import PublicPolicy
from project.utils import make_plot, make_plots
from project.input.resources import resources_data


ENERGY = ['Electricity', 'Natural gas', 'Oil fuel', 'Wood fuel']

CONFIG_TEST = 'project/config/coupling/config_coupling_simple.json'
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

    t = datetime.today().strftime('%Y%m%d')

    # export calibration
    with open(os.path.join(buildings.path_calibration, 'calibration_{}.pkl'.format(t)), 'wb') as file:
        dump({
            'coefficient_global': buildings.coefficient_global,
            'coefficient_heater': buildings.coefficient_heater,
            'constant_insulation_extensive': buildings.constant_insulation_extensive,
            'constant_insulation_intensive': buildings.constant_insulation_intensive,
            'constant_heater': buildings.constant_heater,
            'scale': buildings.scale
        }, file)

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
                                     demolition_rate=demolition_rate)

    output = pd.concat((output, o), axis=1)
    output.to_csv(os.path.join(buildings.path, 'output_ini.csv'))

    return buildings, energy_prices, taxes, cost_heater, cost_insulation, lifetime_heater, demolition_rate, flow_built, post_inputs, policies_heater, policies_insulation, technical_progress, financing_cost, config.get('premature_replacement')


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
                 premature_replacement=None
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

    output, stock, consumption, prices = dict(), dict(), None, None
    s = None
    for year in range(start, end):
        prices = energy_prices.loc[year, :]
        f_built = flow_built.loc[:, year]
        p_heater = [p for p in policies_heater if (year >= p.start) and (year < p.end)]
        p_insulation = [p for p in policies_insulation if (year >= p.start) and (year < p.end)]

        if technical_progress is not None:
            if technical_progress.get('insulation') is not None:
                cost_insulation *= (1 + technical_progress['insulation'].loc[year])
            if technical_progress.get('heater') is not None:
                heat_pump = [i for i in resources_data['index']['Heat pumps'] if i in cost_heater.index]
                cost_heater.loc[heat_pump] *= (1 + technical_progress['heater'].loc[year])

        buildings, s, o = stock_turnover(buildings, prices, taxes, cost_heater, lifetime_heater,
                                         cost_insulation, p_heater,
                                         p_insulation, f_built, year, post_inputs,
                                         financing_cost=financing_cost,
                                         demolition_rate=demolition_rate,
                                         full_output=full_output,
                                         premature_replacement=premature_replacement)

        output.update({year: o})
        if full_output:
            stock.update({year: s})

    output = DataFrame(output)
    if full_output:
        stock = DataFrame(stock)
        stock.index.names = s.index.names

    if output_consumption is True:
        buildings.logger.info('Calculating hourly consumption')

        consumption = buildings.consumption_agg(prices=prices, freq='hour', standard=False, climate=climate,
                                                smooth=smooth, efficiency_hour=efficiency_hour)
        if rebound is False:
            # TODO: only work if there at least two years
            consumption_energy = output.loc[['Consumption {} climate (TWh)'.format(i) for i in ENERGY], :].sum(axis=1).set_axis(ENERGY)
            rebound_energy = output.loc[['Rebound {} (TWh)'.format(i) for i in ENERGY], :].sum(axis=1).set_axis(ENERGY)
            rebound_energy.index = ENERGY
            rebound_factor = rebound_energy / consumption_energy
            consumption = (consumption.T * (1 - rebound_factor)).T

    buildings.logger.info('End of Res-IRF simulation')
    return output, stock, consumption


def run_multi_simu(buildings, sub_heater, start, end, energy_prices, taxes, cost_heater, cost_insulation,
                   lifetime_heater, flow_built, post_inputs, policies_heater, policies_insulation, financing_cost, sub_design=None):

    sub_insulation = [i / 10 for i in range(0, 11, 2)]
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
    sub_design_list = {'Global renovation FG': 'global_renovation_fg',
                       'Global renovation FGE': 'global_renovation_fge',
                       'Cost efficiency 100': 'efficiency_100',
                       'Uniform': None,

                       'Efficiency measure': 'best_efficiency',
                       'Efficiency measure FG': 'best_efficiency_fg',
                       'Global renovation': 'global_renovation'

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

        variables = ['Consumption standard saving insulation (TWh/year)',
                     'Annuities insulation (Billion euro/year)',
                     'Efficiency insulation (euro/kWh)'
                     ]
        result_diff = result.loc[variables, :].diff(axis=1).dropna(axis=1, how='all')
        result_diff.loc['Efficiency insulation (euro/kWh)'] = result_diff.loc['Annuities insulation (Billion euro/year)'] / \
                                                                        result_diff.loc['Consumption standard saving insulation (TWh/year)']

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
    config = CONFIG_TEST
    buildings, energy_prices, taxes, cost_heater, cost_insulation, lifetime_heater, demolition_rate, flow_built, post_inputs, p_heater, p_insulation, technical_progress, financing_cost, premature_replacement = ini_res_irf(
        path=path,
        config=config)

    sub_heater = 0
    sub_insulation = 0

    concat_output, concat_stock = DataFrame(), DataFrame()
    output, stock, consumption = simu_res_irf(buildings, sub_heater, sub_insulation, start, end, energy_prices, taxes,
                                              cost_heater, cost_insulation, lifetime_heater, flow_built, post_inputs,
                                              p_heater,
                                              p_insulation, sub_design, financing_cost, climate=2006, smooth=False,
                                              efficiency_hour=False, demolition_rate=demolition_rate,
                                              output_consumption=output_consumption, full_output=True, rebound=rebound,
                                              technical_progress=technical_progress,
                                              premature_replacement=premature_replacement)

    concat_output = concat((concat_output, output), axis=1)

    concat_output.to_csv(os.path.join(buildings.path, 'output.csv'))
    from project.write_output import plot_scenario
    concat_stock = concat((concat_stock, stock), axis=1)

    plot_scenario(concat_output, concat_stock, buildings)


if __name__ == '__main__':
    # test_design_subsidies()
    run_simu(output_consumption=True, rebound=True, start=2020, end=2021, sub_design='efficiency_100')
