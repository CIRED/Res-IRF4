import time
from pandas import read_csv, concat, Series, Index, DataFrame
# imports from ResIRF
from project.model import config2inputs, initialize, stock_turnover, calibration_res_irf
from project.read_input import PublicPolicy
from multiprocessing import Pool
import os
from pickle import dump, load


def ini_res_irf(path=None, logger=None, config=None, export_calibration=None, import_calibration=None):
    """Initialize and calibrate Res-IRF.

    Parameters
    ----------
    path
    logger
    config

    Returns
    -------
    AgentBuildings
        Building stock object initialize.
    """
    if path is None:
        path = os.path.join('output', 'ResIRF')
    if not os.path.isdir(path):
        os.mkdir(path)
    # TODO check if calibration
    if import_calibration is not None:
        with open(import_calibration, "rb") as file:
            calibration = load(file)
    else:
        calibration = calibration_res_irf(os.path.join(path, 'calibration'), config=config)

    if export_calibration is not None:
        if not os.path.isdir(export_calibration):
            os.mkdir(export_calibration)

        with open(os.path.join(export_calibration, 'calibration.pkl'), "wb") as file:
            dump(calibration, file)

    # creating object to calibrate with calibration
    inputs, stock, year, policies_heater, policies_insulation, taxes = config2inputs(config)
    buildings, energy_prices, taxes, post_inputs, cost_heater, ms_heater, cost_insulation, calibration_intensive, calibration_renovation, flow_built, financing_cost = initialize(
        inputs, stock, year, taxes, path=path, config=config, logger=logger)
    buildings.calibration_exogenous(**calibration)

    return buildings, energy_prices, taxes, cost_heater, cost_insulation, flow_built, post_inputs


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
        output.update({y: o})

    if output_consumption is True:
        buildings.logger.info('Calculating hourly consumption')
        consumption = buildings.consumption_total(prices=prices, freq='hour', standard=False, climate=climate,
                                                  smooth=smooth, efficiency_hour=efficiency_hour)

    output = DataFrame(output)
    buildings.logger.info('End of Res-IRF simulation')
    return output, consumption


if __name__ == '__main__':

    # first time
    export_calibration = os.path.join('project', 'output', 'calibration')
    import_calibration = os.path.join(export_calibration, 'calibration.pkl')

    # then
    buildings, energy_prices, taxes, cost_heater, cost_insulation, flow_built, post_inputs = ini_res_irf(path=os.path.join('project', 'output', 'ResIRF'),
                                                                                                         logger=None,
                                                                                                         config=None,
                                                                                                         import_calibration=import_calibration,
                                                                                                         export_calibration=None)

    timestep = 1
    year = 2020
    start = year
    end = year + timestep

    sub_heater = 0.2
    sub_insulation = 0.5

    simu_res_irf(buildings, sub_heater, sub_insulation, start, end, energy_prices, taxes,
                 cost_heater, cost_insulation, flow_built, post_inputs, climate=2006,
                 smooth=False, efficiency_hour=False, output_consumption=False,
                 full_output=True)

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