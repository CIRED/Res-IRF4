import time
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
        if not os.path.isdir(export_calibration):
            os.mkdir(export_calibration)

        with open(os.path.join(export_calibration, 'calibration.pkl'), "wb") as file:
            dump(calibration, file)

    inputs, stock, year, policies_heater, policies_insulation, taxes = config2inputs(config)
    buildings, energy_prices, taxes, post_inputs, cost_heater, ms_heater, cost_insulation, calibration_intensive, calibration_renovation, flow_built, financing_cost = initialize(
        inputs, stock, year, taxes, path=path, config=config, logger=logger)
    buildings.calibration_exogenous(**calibration)

    return buildings, energy_prices, taxes, cost_heater, cost_insulation, flow_built, post_inputs


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
    heater = ['Electricity-Heat pump water',
              'Electricity-Heat pump air',
              'Electricity-Performance boiler',
              'Natural gas-Performance boiler',
              'Wood fuel-Performance boiler'
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
        'Emission (MtCO2)'
    ]
    variables += ['Replacement heater {} (Thousand households)'.format(i) for i in heater]

    return output.loc[variables]


def simu_res_irf(buildings, sub_heater, sub_insulation, start, end, energy_prices, taxes, cost_heater, cost_insulation,
                 flow_built, post_inputs, climate=2006, smooth=False, efficiency_hour=False,
                 output_consumption=True, full_output=True):

    # setting output format
    buildings.full_output = full_output

    # initialize policies
    p_heater, p_insulation = [], []

    if sub_heater is not None:
        sub_heater = Series([sub_heater, sub_heater],
                            index=Index(['Electricity-Heat pump water', 'Electricity-Heat pump air'],
                                        name='Heating system final'))
        p_heater.append(PublicPolicy('sub_heater_optim', start, end, sub_heater, 'subsidy_ad_volarem',
                                     gest='heater', by='columns'))  # heating policy during considered years

    if sub_insulation is not None:
        p_insulation.append(
            PublicPolicy('sub_insulation_optim', start, end, sub_insulation, 'subsidy_ad_volarem',
                         gest='insulation'))  # insulation policy during considered years

    output, consumption, prices = dict(), None, None
    for y in range(start, end):
        prices = energy_prices.loc[y, :]
        f_built = flow_built.loc[:, y]

        buildings, _, o = stock_turnover(buildings, prices, taxes, cost_heater, cost_insulation, p_heater,
                                         p_insulation, f_built, y, post_inputs)

        output.update({y: select_output(o)})

    if output_consumption is True:
        buildings.logger.info('Calculating hourly consumption')
        consumption = buildings.consumption_total(prices=prices, freq='hour', standard=False, climate=climate,
                                                  smooth=smooth, efficiency_hour=efficiency_hour)

    output = DataFrame(output)
    buildings.logger.info('End of Res-IRF simulation')
    return output, consumption


if __name__ == '__main__':

    # first time
    _export_calibration = os.path.join('project', 'output', 'calibration')
    _import_calibration = os.path.join(_export_calibration, 'calibration.pkl')

    # then
    _buildings, _energy_prices, _taxes, _cost_heater, _cost_insulation, _flow_built, _post_inputs = ini_res_irf(
        path=os.path.join('project', 'output', 'ResIRF'),
        logger=None,
        config=os.path.join('project', 'input', 'config.json'),
        import_calibration=_import_calibration,
        export_calibration=_export_calibration)

    timestep = 1
    _year = 2020
    _start = _year
    _end = _year + timestep

    _sub_heater = 0.2
    _sub_insulation = 0.5

    _output, _consumption = simu_res_irf(_buildings, _sub_heater, _sub_insulation, _start, _end, _energy_prices, _taxes,
                                         _cost_heater, _cost_insulation, _flow_built, _post_inputs, climate=2006,
                                         smooth=False, efficiency_hour=False, output_consumption=True)

    print('break')
    print('break')

    """list_argument = [(sub_heater, 0.5, 2020, 2021) for sub_heater in [0.1, 0.9]]

    with Pool(4) as pool:
        results = pool.starmap(run_resirf, list_argument)

    sub_heater = Series([i[0] for i in results], name='Sub heater')
    sub_insulation = Series([i[1] for i in results], name='Sub insulation')
    df = concat([Series(i[2]) for i in results], axis=1)
    df = concat((sub_heater, sub_insulation, df.T), axis=1).T
    df.to_csv('output/sensitivity.csv')"""