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
from project.utils import get_series
from project.input.resources import resources_data
from project.write_output import plot_scenario

ENERGY = ['Electricity', 'Natural gas', 'Oil fuel', 'Wood fuel']


def ini_res_irf(config=None, path=None, level_logger='DEBUG'):
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

    config, inputs, stock, year, policies_heater, policies_insulation, taxes = config2inputs(config)
    inputs_dynamics = initialize(inputs, stock, year, taxes, path=path, config=config, level_logger=level_logger)
    buildings, energy_prices = inputs_dynamics['buildings'], inputs_dynamics['energy_prices']

    # calibration
    if config.get('calibration'):
        with open(config['calibration'], "rb") as file:
            calibration = load(file)
    else:
        calibration = calibration_res_irf(path, config=config, level_logger=level_logger)

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
    c_content = inputs_dynamics['post_inputs']['carbon_emission'].loc[year, :]

    flow_district_heating = None
    if inputs_dynamics['flow_district_heating'] is not None:
        flow_district_heating = inputs_dynamics['flow_district_heating'].loc[year]

    buildings, _, o = stock_turnover(buildings, prices, taxes,
                                     inputs_dynamics['cost_heater'],
                                     inputs_dynamics['lifetime_heater'],
                                     inputs_dynamics['cost_insulation'],
                                     inputs_dynamics['lifetime_insulation'],
                                     p_heater, p_insulation, f_built, year,
                                     inputs_dynamics['post_inputs'],
                                     calib_renovation=inputs_dynamics['calibration_renovation'],
                                     calib_heater=inputs_dynamics['calibration_heater'],
                                     premature_replacement=inputs_dynamics['premature_replacement'],
                                     financing_cost=inputs_dynamics['financing_cost'],
                                     district_heating=flow_district_heating,
                                     demolition_rate=inputs_dynamics['demolition_rate'],
                                     exogenous_social=inputs.get('exogenous_social'),
                                     output_details=config['output'],
                                     carbon_content=c_content
                                     )

    output = pd.concat((output, o), axis=1)
    output.round(3).to_csv(os.path.join(buildings.path, 'output_ini.csv'))

    return buildings, inputs_dynamics, policies_heater, policies_insulation


def read_ad_valorem(data):
    l = list()

    value = data['value']
    if isinstance(value, str):
        value = get_series(data['value'])

    by = 'index'
    if data.get('index') is not None:
        mask = get_series(data['index'], header=0)
        value *= mask

    if data.get('columns') is not None:
        mask = get_series(data['columns'], header=0)
        if isinstance(value, (float, int)):
            value *= mask
        else:
            value = value.to_frame().dot(mask.to_frame().T)
        by = 'columns'

    if data.get('growth'):
        growth = get_series(data['growth'], header=None)
        value = {k: i * value for k, i in growth.items()}

    name = 'sub_ad_valorem'
    if data.get('name') is not None:
        name = data['name']

    if isinstance(data['gest'], str):
        data['gest'] = [data['gest']]

    for gest in data['gest']:
        l.append(PublicPolicy(name, data['start'], data['end'], value, 'subsidy_ad_valorem',
                              gest=gest, by=by, target=data.get('target')))
    return l


def read_proportional(data):
    l = list()

    value = data['value']
    if isinstance(value, str):
        value = get_series(data['value'])

    by = 'index'
    if data.get('index') is not None:
        mask = get_series(data['index'], header=0)
        value *= mask

    if data.get('columns') is not None:
        mask = get_series(data['columns'],  header=0).rename(None)
        if isinstance(value, (float, int)):
            value *= mask
        else:
            value = value.to_frame().dot(mask.to_frame().T)
        by = 'columns'

    if data.get('growth'):
        growth = get_series(data['growth'], header=None)
        value = {k: i * value for k, i in growth.items()}

    name = 'sub_proportional'
    if data.get('name') is not None:
        name = data['name']

    proportional = 'MWh_cumac'  # tCO2_cumac
    if data.get('proportional') is not None:
        proportional = data['proportional']

    if isinstance(data['gest'], str):
        data['gest'] = [data['gest']]
    for gest in data['gest']:
        l.append(PublicPolicy(name, data['start'], data['end'], value, 'subsidy_proportional',
                              gest=gest, by=by, target=data.get('target'), proportional=proportional))

    return l


def simu_res_irf(buildings, start, end, energy_prices, taxes, cost_heater, cost_insulation,
                 lifetime_heater, lifetime_insulation, flow_built, post_inputs, policies_heater, policies_insulation,
                 financing_cost,
                 sub_heater=None, sub_insulation=None, climate=2006, smooth=False, efficiency_hour=False, demolition_rate=None,
                 output_consumption=False, output_details='full', technical_progress=None,
                 premature_replacement=None, flow_district_heating=None, exogenous_social=None, carbon_content=None
                 ):

    # initialize policies
    if sub_heater is not None:
        if sub_heater.get('policy') == 'subsidy_ad_valorem' and sub_heater.get('value') != 0:
            policies_heater += read_ad_valorem(sub_heater)
        elif sub_heater.get('policy') == 'subsidy_proportional' and sub_heater.get('value') != 0:
            policies_heater += read_proportional(sub_heater)

    if sub_insulation is not None:
        if sub_insulation.get('policy') == 'subsidy_ad_valorem' and sub_insulation.get('value') != 0:
            policies_insulation += read_ad_valorem(sub_insulation)
        elif sub_insulation.get('policy') == 'subsidy_proportional' and sub_insulation.get('value') != 0:
            policies_insulation += read_proportional(sub_insulation)

    output, stock, consumption, prices = dict(), dict(), None, None
    s = None
    for year in range(start, end):
        f_built = flow_built.loc[:, year]
        p_heater = [p for p in policies_heater if (year >= p.start) and (year < p.end)]
        p_insulation = [p for p in policies_insulation if (year >= p.start) and (year < p.end)]
        c_content = carbon_content.loc[year, :]

        prices = energy_prices.loc[year, :]
        prices_before = energy_prices.loc[year - 1, :]

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
                                         carbon_content=c_content,
                                         prices_before=prices_before
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
                   lifetime_heater, flow_built, post_inputs, policies_heater, policies_insulation, financing_cost,
                   sub_design=None):

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


def run_simu(config, output_consumption=False, start=2019, end=2021):

    path = os.path.join('project', 'output', 'ResIRF')

    buildings, inputs_dynamics, policies_heater, policies_insulation = ini_res_irf(path=path, config=config,
                                                                                   level_logger='DEBUG')

    sub_heater = {'name': 'sub_heater',
                  'start': start,
                  'end': end,
                  'value': 0,
                  'policy': 'subsidy_proportional',
                  'gest': 'heater',
                  'columns': 'project/input/policies/target/target_heat_pump.csv',
                  'proportional': 'tCO2_cumac'
                  }
    target = None
    sub_insulation = {'name': 'sub_insulation',
                      'start': start,
                      'end': end,
                      'value': 0,
                      'policy': 'subsidy_ad_valorem',
                      'gest': 'insulation',
                      'target': target,
                      'proportional': None
                      }

    # 'subsidy_proportional'
    # proportional = 'MWh_cumac'  # tCO2_cumac

    concat_output, concat_stock = DataFrame(), DataFrame()
    output, stock, consumption = simu_res_irf(buildings, start, end,
                                              inputs_dynamics['energy_prices'], inputs_dynamics['taxes'],
                                              inputs_dynamics['cost_heater'], inputs_dynamics['cost_insulation'],
                                              inputs_dynamics['lifetime_heater'], inputs_dynamics['lifetime_insulation'],
                                              inputs_dynamics['flow_built'],
                                              inputs_dynamics['post_inputs'],
                                              policies_heater,
                                              policies_insulation,
                                              inputs_dynamics['financing_cost'],
                                              sub_heater=sub_heater,
                                              sub_insulation=sub_insulation,
                                              climate=2006,
                                              smooth=False,
                                              efficiency_hour=False,
                                              demolition_rate=inputs_dynamics['demolition_rate'],
                                              output_consumption=output_consumption,
                                              technical_progress=inputs_dynamics['technical_progress'],
                                              premature_replacement=inputs_dynamics['premature_replacement'],
                                              output_details='cost_benefit',
                                              carbon_content=inputs_dynamics['post_inputs']['carbon_emission']
                                              )

    concat_output = concat((concat_output, output), axis=1)

    concat_output.round(3).to_csv(os.path.join(buildings.path, 'output.csv'))
    concat_stock = concat((concat_stock, stock), axis=1)

    plot_scenario(concat_output, concat_stock, buildings)


if __name__ == '__main__':
    # test_design_subsidies()
    _config = 'project/config/coupling/config.json'
    # _config = 'project/config/config.json'
    run_simu(output_consumption=False, start=2019, end=2025, config=_config)
