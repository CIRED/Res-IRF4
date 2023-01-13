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

    # initialization
    inputs, stock, year, policies_heater, policies_insulation, taxes = config2inputs(config)
    buildings, energy_prices, taxes, post_inputs, cost_heater, ms_heater, cost_insulation, calibration_intensive, calibration_renovation, flow_built, financing_cost = initialize(
        inputs, stock, year, taxes, path=path, config=config, logger=logger)

    calibration = calibration_res_irf(os.path.join(path, 'calibration'), config=config)
    if export_calibration is not None:
        if not os.path.isdir(export_calibration):
            os.mkdir(export_calibration)

        with open(os.path.join(export_calibration, 'calibration.pkl'), "wb") as file:
            dump(calibration, file)

    if import_calibration is not None:
        with open(import_calibration, "rb") as file:
            calibration = load(file)

    buildings.calibration_exogenous(**calibration)
    return buildings, energy_prices, taxes, cost_heater, cost_insulation, flow_built, post_inputs


def simu_res_irf(buildings, sub_heater, sub_insulation, start, end, energy_prices, taxes, cost_heater, cost_insulation, flow_built, post_inputs):
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

    output, o = dict(), None
    for y in range(start, end):
        s = time.time()
        prices = energy_prices.loc[y, :]
        f_built = flow_built.loc[:, y]

        buildings, _, _ = stock_turnover(buildings, prices, taxes, cost_heater, cost_insulation, p_heater,
                                         p_insulation, f_built, y, post_inputs)

        o = dict()

        o['Electricity (TWh)'] = buildings.heat_consumption_energy['Electricity'] / 10**9
        o['Natural gas (TWh)'] = buildings.heat_consumption_energy['Natural gas'] / 10**9
        o['Wood fuel (TWh)'] = buildings.heat_consumption_energy['Wood fuel'] / 10**9
        o['Oil fuel (TWh)'] = buildings.heat_consumption_energy['Oil fuel'] / 10**9

        o['Investment heater (Billion euro)'] = buildings.investment_heater.sum().sum() / 10**9
        o['Investment insulation (Billion euro)'] = buildings.investment_insulation.sum().sum() / 10**9
        o['Investment (Billion euro)'] = o['Investment heater (Billion euro)'] + o['Investment insulation (Billion euro)']
        o['Subsidies (Billion euro)'] = (buildings.subsidies_heater.sum().sum() + buildings.subsidies_insulation.sum().sum()) / 10**9
        o['Health cost (Billion euro)'], _ = buildings.health_cost(post_inputs)

        temp = buildings.heating_consumption(freq='hour', climate=2006, smooth=False)
        t = (temp.T * buildings.stock * buildings.surface).T
        # adding heating intensity
        t = (t.T * buildings.heating_intensity).T
        energy = temp.index.get_level_values('Heating system').str.split('-').str[0]
        t = t.groupby(energy).sum()
        t = (t.T * buildings.coefficient_consumption).T

        o['Hourly consumption (kWh)'] = t

        """o['Heat pump air'] = buildings.replacement_heater.sum().loc['Electricity-Heat pump air'] / 1e3
        o['Heat pump water'] = buildings.replacement_heater.sum().loc['Electricity-Heat pump water'] / 1e3"""

        temp = buildings.replacement_heater.sum() / 10**3
        temp.index = temp.index.map(lambda x: 'Replacement {} (Thousand)'.format(x))
        o.update(temp.to_dict())

        temp = buildings.stock.groupby('Heating system').sum() / 10**3
        temp.index = temp.index.map(lambda x: 'Stock {} (Thousand)'.format(x))
        o.update(temp.to_dict())

        output.update({y: o})

        # problem here with new code
        print(time.time() - s)
    return output


if __name__ == '__main__':

    # first time
    export_calibration = os.path.join('project', 'output', 'calibration')
    import_calibration = os.path.join(export_calibration, 'calibration.pkl')
    buildings, energy_prices, taxes, cost_heater, cost_insulation, flow_built, post_inputs = ini_res_irf(path=os.path.join('project', 'output', 'ResIRF'),
                                                                                                         logger=None,
                                                                                                         config=None,
                                                                                                         export_calibration=export_calibration)

    # then
    buildings, energy_prices, taxes, cost_heater, cost_insulation, flow_built, post_inputs = ini_res_irf(path=os.path.join('project', 'output', 'ResIRF'),
                                                                                                         logger=None,
                                                                                                         config=None,
                                                                                                         import_calibration=import_calibration)

    timestep = 1
    year = 2020
    start = year
    end = year + timestep

    sub_heater = 0.2
    sub_insulation = 0.5

    output = simu_res_irf(buildings, sub_heater, sub_insulation, start, end, energy_prices, taxes, cost_heater,
                          cost_insulation, flow_built, post_inputs)

    """list_argument = [(sub_heater, 0.5, 2020, 2021) for sub_heater in [0.1, 0.9]]

    with Pool(4) as pool:
        results = pool.starmap(run_resirf, list_argument)

    sub_heater = Series([i[0] for i in results], name='Sub heater')
    sub_insulation = Series([i[1] for i in results], name='Sub insulation')
    df = concat([Series(i[2]) for i in results], axis=1)
    df = concat((sub_heater, sub_insulation, df.T), axis=1).T
    df.to_csv('output/sensitivity.csv')"""

    print('ok')
    print('ok')
