import os
import pandas as pd
import logging
from time import time
import json
from importlib import resources

from project.building import AgentBuildings, ThermalBuildings
from project.read_input import read_stock, read_policies, read_inputs, parse_inputs, dump_inputs, PublicPolicy
from project.write_output import plot_scenario

LOG_FORMATTER = '%(asctime)s - %(process)s - %(name)s - %(levelname)s - %(message)s'


def get_config() -> dict:
    with resources.path('project.input', 'config.json') as f:
        with open(f) as file:
            return json.load(file)['Reference']


def config2inputs(config=None):
    """Create main Python object from configuration file.

    Parameters
    ----------
    config

    Returns
    -------

    """

    if config is None:
        config = get_config()

    stock, year = read_stock(config)
    policies_heater, policies_insulation, taxes = read_policies(config)
    inputs = read_inputs(config)
    if config['quintiles']:
        stock, policies_heater, policies_insulation, inputs = deciles2quintiles(stock, policies_heater,
                                                                                policies_insulation, inputs)

    return inputs, stock, year, policies_heater, policies_insulation, taxes


def deciles2quintiles(stock, policies_heater, policies_insulation, inputs):
    """Change all inputs from deciles to quintiles.

    Parameters
    ----------
    stock
    policies_heater
    policies_insulation
    inputs

    Returns
    -------

    """

    replace = {'D1': 'C1', 'D2': 'C1',
               'D3': 'C2', 'D4': 'C2',
               'D5': 'C3', 'D6': 'C3',
               'D7': 'C4', 'D8': 'C4',
               'D9': 'C5', 'D10': 'C5'}

    def apply_to_pandas(data, func='mean'):
        level_income = []
        for key in ['Income owner', 'Income tenant', 'Income']:
            if key in data.index.names:
                level_income += [key]

        for level in level_income:
            names = None
            if isinstance(data.index, pd.MultiIndex):
                names = data.index.names

            data = data.rename(index=replace, level=level)

            if func == 'mean':
                data = data.groupby(data.index).mean()
            elif func == 'sum':
                data = data.groupby(data.index).sum()

            if names:
                data.index = pd.MultiIndex.from_tuples(data.index)
                data.index.names = names

        return data

    for key, item in inputs.items():
        if isinstance(item, (pd.Series, pd.DataFrame)):
            inputs[key] = apply_to_pandas(item)
        elif isinstance(item, dict):
            for k, i in item.items():
                if isinstance(i, (pd.Series, pd.DataFrame)):
                    inputs[key][k] = apply_to_pandas(i)
                elif isinstance(i, dict):
                    for kk, ii in i.items():
                        if isinstance(ii, (pd.Series, pd.DataFrame)):
                            inputs[key][k][kk] = apply_to_pandas(ii)
                        elif isinstance(ii, dict):
                            for kkk, iii in ii.items():
                                if isinstance(iii, (pd.Series, pd.DataFrame)):
                                    inputs[key][k][kk][kkk] = apply_to_pandas(iii)

    stock = apply_to_pandas(stock, func='sum')

    for policy in policies_insulation + policies_heater:
        attributes = [a for a in dir(policy) if not a.startswith('__') and getattr(policy, a) is not None]
        for att in attributes:
            item = getattr(policy, att)
            if isinstance(item, (pd.Series, pd.DataFrame)):
                setattr(policy, att, apply_to_pandas(item))
            if isinstance(item, dict):
                print(item)

    return stock, policies_heater, policies_insulation, inputs


def select_post_inputs(parsed_inputs):
    """Inputs used during post-treatment but not used during the iteration.

    Parameters
    ----------
    parsed_inputs: dict

    Returns
    -------
    dict
    """

    vars = ['carbon_emission', 'population', 'surface', 'embodied_energy_renovation', 'carbon_footprint_renovation',
            'Carbon footprint construction (MtCO2)', 'Embodied energy construction (TWh PE)',
            'health_expenditure', 'mortality_cost', 'loss_well_being', 'carbon_value_kwh']

    return {key: item for key, item in parsed_inputs.items() if key in vars}


def get_inputs(path, config=None):
    """Initialize thermal buildings object based on inpuut dictionnary.

    Parameters
    ----------
    path: str

    Returns
    -------
    dict
    """

    inputs, stock, year, policies_heater, policies_insulation, taxes = config2inputs(config)
    buildings, energy_prices, taxes, post_inputs, cost_heater, ms_heater, cost_insulation, ms_intensive, renovation_rate_ini, flow_built = initialize(
        inputs, stock, year, taxes, path, config=config)
    output = {'buildings': buildings, 'energy_prices': energy_prices, 'cost_insulation': cost_insulation,
              'carbon_emission': post_inputs['carbon_emission'], 'carbon_value_kwh': post_inputs['carbon_value_kwh'],
              'health_cost': post_inputs['health_expenditure'] + post_inputs['mortality_cost'] + post_inputs['loss_well_being']
              }

    return output


def initialize(inputs, stock, year, taxes, path, config=None, logger=None):
    """Create main Python objects read by model.

    Parameters
    ----------
    inputs
    stock
    year
    taxes
    config
    path
    logger

    Returns
    -------

    """

    if config is None:
        config = get_config()

    parsed_inputs = parse_inputs(inputs, taxes, config, stock)
    dump_inputs(parsed_inputs, path)
    post_inputs = select_post_inputs(parsed_inputs)
    if logger is None:
        logger = create_logger(path)
    logger.info('Creating AgentBuildings object')

    with open(os.path.join(path, 'config.json'), 'w') as fp:
        json.dump(config, fp)

    buildings = AgentBuildings(stock, parsed_inputs['surface'], parsed_inputs['ratio_surface'], parsed_inputs['efficiency'],
                               parsed_inputs['income'], parsed_inputs['consumption_ini'], path, parsed_inputs['preferences'],
                               parsed_inputs['performance_insulation'],
                               year=year, demolition_rate=parsed_inputs['demolition_rate'],
                               endogenous=config['endogenous'], logger=logger,
                               remove_market_failures=config.get('remove_market_failures'),
                               quintiles=config.get('quintiles'))

    return buildings, parsed_inputs['energy_prices'], parsed_inputs['taxes'], post_inputs, parsed_inputs['cost_heater'], parsed_inputs['ms_heater'], \
           parsed_inputs['cost_insulation'], parsed_inputs['ms_intensive'], parsed_inputs[
               'renovation_rate_ini'], parsed_inputs['flow_built']


def stock_turnover(buildings, prices, taxes, cost_heater, cost_insulation, p_heater, p_insulation, flow_built, year,
                   post_inputs,  ms_heater=None,  ms_insulation=None, renovation_rate_ini=None,
                   target_freeriders=None):

    buildings.logger.info('Run {}'.format(year))
    buildings.year = year
    buildings.add_flows([- buildings.flow_demolition()])
    flow_retrofit = buildings.flow_retrofit(prices, cost_heater, cost_insulation,
                                            policies_heater=p_heater,
                                            policies_insulation=p_insulation,
                                            ms_insulation=ms_insulation,
                                            renovation_rate_ini=renovation_rate_ini,
                                            target_freeriders=target_freeriders,
                                            ms_heater=ms_heater)
    buildings.add_flows([flow_retrofit, flow_built])
    buildings.calculate_consumption(prices, taxes)
    buildings.logger.info('Writing output')
    if buildings.detailed_mode:
        s, o = buildings.parse_output_run(post_inputs)
    else:
        s = buildings.simplified_stock()
        o = dict()
        o['Electricity'] = buildings.heat_consumption_energy['Electricity'] / 10 ** 9
        o['Natural gas'] = buildings.heat_consumption_energy['Natural gas'] / 10 ** 9
        o['Wood fuel'] = buildings.heat_consumption_energy['Wood fuel'] / 10 ** 9
        o['Oil fuel'] = buildings.heat_consumption_energy['Oil fuel'] / 10 ** 9

    return buildings, s, o


def create_logger(path):
    """Create logger for one run.

    Parameters
    ----------
    path: str

    Returns
    -------
    Logger
    """
    logger = logging.getLogger('log_{}'.format(path.split('/')[-1].lower()))
    logger.setLevel('DEBUG')
    logger.propagate = False
    # consoler handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(LOG_FORMATTER))
    logger.addHandler(console_handler)
    # file handler
    file_handler = logging.FileHandler(os.path.join(path, 'log.log'))
    file_handler.setFormatter(logging.Formatter(LOG_FORMATTER))
    logger.addHandler(file_handler)
    return logger


def res_irf(config, path):
    """Res-IRF model.

    Parameters
    ----------
    config: dict
        Scenario-specific input
    path: str
        Scenario-specific output path

    Returns
    -------
    str
        Scenario name
    pd.DataFrame
        Detailed results
    """
    os.mkdir(path)
    logger = create_logger(path)

    try:
        logger.info('Reading input')

        inputs, stock, year, policies_heater, policies_insulation, taxes = config2inputs(config)
        buildings, energy_prices, taxes, post_inputs, cost_heater, ms_heater, cost_insulation, ms_intensive, renovation_rate_ini, flow_built = initialize(
            inputs, stock, year, taxes, path, config, logger)

        output, stock = pd.DataFrame(), pd.DataFrame()
        buildings.logger.info('Calibration energy consumption {}'.format(buildings.first_year))
        buildings.calculate_consumption(energy_prices.loc[buildings.first_year, :], taxes)
        s, o = buildings.parse_output_run(post_inputs)
        stock = pd.concat((stock, s), axis=1)
        output = pd.concat((output, o), axis=1)

        for year in range(config['start'] + 1, config['end']):
            start = time()

            prices = energy_prices.loc[year, :]
            p_heater = [p for p in policies_heater if (year >= p.start) and (year < p.end)]
            p_insulation = [p for p in policies_insulation if (year >= p.start) and (year < p.end)]
            f_built = flow_built.loc[:, year]
            target_freeriders = config['target_freeriders']

            buildings, s, o = stock_turnover(buildings, prices, taxes, cost_heater, cost_insulation, p_heater,
                                             p_insulation, f_built, year, post_inputs,
                                             ms_insulation=ms_intensive, renovation_rate_ini=renovation_rate_ini,
                                             target_freeriders=target_freeriders, ms_heater=ms_heater)

            stock = pd.concat((stock, s), axis=1)
            stock.index.names = s.index.names
            output = pd.concat((output, o), axis=1)
            buildings.logger.info('Run time {}: {:,.0f} seconds.'.format(year, round(time() - start, 2)))

        buildings.logger.info('Dumping output in {}'.format(path))
        output.round(3).to_csv(os.path.join(path, 'output.csv'))
        stock.round(2).to_csv(os.path.join(path, 'stock.csv'))
        if buildings.detailed_mode:
            plot_scenario(output, stock, buildings)

        return os.path.basename(os.path.normpath(path)), output, stock

    except Exception as e:
        buildings.logger.exception(e)
        raise e


if __name__ == '__main__':
    output = get_inputs('output')
    buildings = output['buildings']
    energy_prices = output['energy_prices']
    cost_insulation = output['cost_insulation']
    carbon_emission = output['carbon_emission']
    carbon_value_kwh = output['carbon_value_kwh']
    health_cost = output['health_cost']
    output = buildings.mitigation_potential(energy_prices, cost_insulation, carbon_emission, carbon_value_kwh,
                                            health_cost=health_cost)

