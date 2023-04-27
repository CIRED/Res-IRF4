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
from project.write_output import plot_scenario

ENERGY = ['Electricity', 'Natural gas', 'Oil fuel', 'Wood fuel']


def ini_res_irf(config=None, path=None):
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
        if isinstance(config, str):
            if os.path.isfile(config):
                with open(config) as file:
                    config = json.load(file).get('Reference')
        elif isinstance(config, dict):
            pass
        else:
            raise NotImplemented('Config should be dict or str')

    if path is None:
        path = os.path.join('output', 'ResIRF')
    if not os.path.isdir(path):
        os.mkdir(path)

    """if not os.path.isdir(path):
        try:
            os.mkdir(path)
        except:
            pass"""

    config, inputs, stock, year, policies_heater, policies_insulation, taxes = config2inputs(config)
    inputs_dynamics = initialize(inputs, stock, year, taxes, path=path, config=config)
    buildings, energy_prices = inputs_dynamics['buildings'], inputs_dynamics['energy_prices']

    # calibration
    if config.get('calibration'):
        with open(config['calibration'], "rb") as file:
            calibration = load(file)
    else:
        calibration = calibration_res_irf(path, config=config)

    buildings.calibration_exogenous(**calibration)

    # export calibration
    t = datetime.today().strftime('%Y%m%d')
    with open(os.path.join(buildings.path_calibration, 'calibration_{}.pkl'.format(t)), 'wb') as file:
        dump({
            'coefficient_global': buildings.coefficient_global,
            'coefficient_heater': buildings.coefficient_heater,
            'constant_insulation_extensive': buildings.constant_insulation_extensive,
            'constant_insulation_intensive': buildings.constant_insulation_intensive,
            'constant_heater': buildings.constant_heater,
            'scale_insulation': buildings.scale_insulation,
            'scale_heater': buildings.scale_heater,
            'rational_hidden_cost': buildings.rational_hidden_cost
        }, file)

    output = pd.DataFrame()
    # run first year - consumption
    _, o = buildings.parse_output_run(energy_prices.loc[buildings.first_year, :], inputs_dynamics['post_inputs'])
    output = pd.concat((output, o), axis=1)

    # run second year - renovation
    year = buildings.first_year + 1
    prices = energy_prices.loc[year, :]
    p_heater = [p for p in policies_heater if (year >= p.start) and (year < p.end)]
    p_insulation = [p for p in policies_insulation if (year >= p.start) and (year < p.end)]
    f_built = inputs_dynamics['flow_built'].loc[:, year]

    flow_district_heating = None
    if inputs_dynamics['flow_district_heating'] is not None:
        flow_district_heating = inputs_dynamics['flow_district_heating'].loc[year]

    buildings, _, o = stock_turnover(buildings, prices, taxes,
                                     inputs_dynamics['cost_heater'], inputs_dynamics['lifetime_heater'],
                                     inputs_dynamics['cost_insulation'], inputs_dynamics['lifetime_insulation'],
                                     p_heater, p_insulation, f_built, year,
                                     inputs_dynamics['post_inputs'],
                                     calib_renovation=inputs_dynamics['calibration_renovation'],
                                     calib_heater=inputs_dynamics['calibration_heater'],
                                     premature_replacement=inputs_dynamics['premature_replacement'],
                                     financing_cost=inputs_dynamics['financing_cost'],
                                     district_heating=flow_district_heating,
                                     demolition_rate=inputs_dynamics['demolition_rate'],
                                     exogenous_social=inputs.get('exogenous_social'),
                                     output_details=config['output']
                                     )

    output = pd.concat((output, o), axis=1)
    output.to_csv(os.path.join(buildings.path, 'output_ini.csv'))

    return buildings, inputs_dynamics, policies_heater, policies_insulation


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
                 lifetime_heater, lifetime_insulation, flow_built, post_inputs, policies_heater, policies_insulation, sub_design, financing_cost,
                 climate=2006, smooth=False, efficiency_hour=False, demolition_rate=None,
                 output_consumption=False, output_details='full', technical_progress=None,
                 premature_replacement=None, flow_district_heating=None, exogenous_social=None
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

        buildings, s, o = stock_turnover(buildings, prices, taxes,
                                         cost_heater, lifetime_heater,
                                         cost_insulation, lifetime_insulation,
                                         p_heater, p_insulation, f_built, year,
                                         post_inputs,
                                         premature_replacement=premature_replacement,
                                         financing_cost=financing_cost,
                                         district_heating=flow_district_heating,
                                         demolition_rate=demolition_rate,
                                         exogenous_social=exogenous_social,
                                         output_details=output_details,
                                         )
        output.update({year: o})
        if output_details == 'full':
            stock.update({year: s})

    output = DataFrame(output)
    if output_details == 'full':
        stock = DataFrame(stock)
        stock.index.names = s.index.names

    if output_consumption is True:
        buildings.logger.info('Calculating hourly consumption')

        consumption = buildings.consumption_agg(prices=prices, freq='hour', standard=False, climate=climate,
                                                smooth=smooth, efficiency_hour=efficiency_hour)

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


def run_simu(config, output_consumption=False, start=2020, end=2021, sub_design=None):

    path = os.path.join('project', 'output', 'ResIRF')

    buildings, inputs_dynamics, policies_heater, policies_insulation = ini_res_irf(path=path, config=config)

    sub_heater, sub_insulation = 0, 0
    concat_output, concat_stock = DataFrame(), DataFrame()
    output, stock, consumption = simu_res_irf(buildings, sub_heater, sub_insulation, start, end,
                                              inputs_dynamics['energy_prices'], inputs_dynamics['taxes'],
                                              inputs_dynamics['cost_heater'], inputs_dynamics['cost_insulation'],
                                              inputs_dynamics['lifetime_heater'], inputs_dynamics['lifetime_insulation'],
                                              inputs_dynamics['flow_built'],
                                              inputs_dynamics['post_inputs'],
                                              policies_heater,
                                              policies_insulation, sub_design, inputs_dynamics['financing_cost'],
                                              climate=2006, smooth=False,
                                              efficiency_hour=False,
                                              demolition_rate=inputs_dynamics['demolition_rate'],
                                              output_consumption=output_consumption,
                                              technical_progress=inputs_dynamics['technical_progress'],
                                              premature_replacement=inputs_dynamics['premature_replacement'],
                                              output_details='full'
                                              )

    concat_output = concat((concat_output, output), axis=1)

    concat_output.to_csv(os.path.join(buildings.path, 'output.csv'))
    concat_stock = concat((concat_stock, stock), axis=1)

    plot_scenario(concat_output, concat_stock, buildings)


if __name__ == '__main__':
    # test_design_subsidies()
    _config = 'project/config/coupling/config.json'
    run_simu(output_consumption=False, start=2019, end=2022, sub_design='global_renovation', config=_config)
