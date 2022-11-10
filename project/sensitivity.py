import time
from pandas import read_csv, concat, Series, Index, DataFrame
# imports from ResIRF
from project.model import config2inputs, initialize, stock_turnover
from project.read_input import PublicPolicy
from multiprocessing import Pool


def run_resirf(sub_heater, sub_insulation, start, end, path='output', logger=None, config=None):
    """Simulating ResIRF, including initialization and calibration

    Parameters
    ----------
    sub_heater: float or None
    sub_insulation: float or None
    config: dict or None, default None
    start: int
    end: int
    path: str
    logger: default None

    Returns
    -------
    sub_heater : float
    sub_insulation: float
    o : dict
        All quantities of interest
    """
    # initialization
    inputs, stock, year, policies_heater, policies_insulation, taxes = config2inputs(config)
    buildings, energy_prices, taxes, post_inputs, cost_heater, ms_heater, cost_insulation, ms_intensive, renovation_rate_ini, flow_built = initialize(
        inputs, stock, year, taxes, path, config=config, logger=logger)

    # reduce the number of outputs
    buildings.detailed_mode = True

    # calibration
    buildings.calibration_exogenous(energy_prices, taxes)

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

    prices = energy_prices.loc[start, :]

    output, o = dict(), None
    for y in range(start, end):
        s = time.time()
        f_built = flow_built.loc[:, y]  # constructed fleet during year
        buildings, _, _ = stock_turnover(buildings, prices, taxes, cost_heater, cost_insulation, p_heater,
                                         p_insulation, f_built, y, post_inputs)

        o = dict()

        o['Electricity'] = buildings.heat_consumption_energy['Electricity'] / 1e9  # TWh
        o['Natural gas'] = buildings.heat_consumption_energy['Natural gas'] / 1e9  # TWh
        o['Wood fuel'] = buildings.heat_consumption_energy['Wood fuel'] / 1e9  # TWh
        o['Oil fuel'] = buildings.heat_consumption_energy['Oil fuel'] / 1e9  # TWh

        o['Investment'] = (buildings.investment_heater.sum().sum() + buildings.investment_insulation.sum().sum()) / 1e9  # Billion euro
        o['Subsidies'] = (buildings.subsidies_heater.sum().sum() + buildings.subsidies_insulation.sum().sum()) / 1e9  # Billion euro
        o['Health cost'], _ = buildings.health_cost(post_inputs)  # Thousand unit

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

    return sub_heater.values[0], sub_insulation, o


if __name__ == '__main__':

    list_argument = [(sub_heater, 0.5, 2020, 2021) for sub_heater in [0.1, 0.9]]

    with Pool(4) as pool:
        results = pool.starmap(run_resirf, list_argument)

    sub_heater = Series([i[0] for i in results], name='Sub heater')
    sub_insulation = Series([i[1] for i in results], name='Sub insulation')
    df = concat([Series(i[2]) for i in results], axis=1)
    df = concat((sub_heater, sub_insulation, df.T), axis=1).T
    df.to_csv('output/sensitivity.csv')

