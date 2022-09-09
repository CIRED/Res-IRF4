import os
import pandas as pd
import logging
from time import time
import json

from project.building import AgentBuildings
from project.input.param import generic_input

from project.read_input import read_stock, read_policies, read_inputs, parse_inputs, dump_inputs, PublicPolicy
from project.write_output import plot_scenario

LOG_FORMATTER = '%(asctime)s - %(process)s - %(name)s - %(levelname)s - %(message)s'


def initialize(config, path, logger):
    stock, year = read_stock(config)
    policies_heater, policies_insulation, taxes = read_policies(config)
    inputs = read_inputs(config, generic_input)
    parsed_inputs = parse_inputs(inputs, config, stock)
    summary_inputs = dump_inputs(parsed_inputs)

    energy_prices = parsed_inputs['energy_prices'].copy()
    energy_taxes = parsed_inputs['energy_taxes'].copy()


    if config['prices_constant']:
        energy_prices = pd.concat([energy_prices.loc[year, :]] * energy_prices.shape[0], keys=energy_prices.index,
                                  axis=1).T

    total_taxes = pd.DataFrame(0, index=energy_prices.index, columns=energy_prices.columns)
    for t in taxes:
        total_taxes = total_taxes.add(t.value, fill_value=0)

    if energy_taxes is not None:
        total_taxes = total_taxes.add(energy_taxes, fill_value=0)
        taxes += [PublicPolicy('energy_taxes', energy_taxes.index[0], energy_taxes.index[-1], energy_taxes, 'tax')]

    if config['taxes_constant']:
        total_taxes = pd.concat([total_taxes.loc[year, :]] * total_taxes.shape[0], keys=total_taxes.index,
                                axis=1).T

    energy_vta = energy_prices * generic_input['vta_energy_prices']
    taxes += [PublicPolicy('energy_vta', energy_vta.index[0], energy_vta.index[-1], energy_vta, 'tax')]
    total_taxes += energy_vta

    energy_prices = energy_prices.add(total_taxes, fill_value=0)
    parsed_inputs['energy_prices'] = energy_prices

    t = total_taxes.copy()
    t.columns = t.columns.map(lambda x: 'Taxes {} (euro/kWh)'.format(x))
    temp = energy_prices.copy()
    temp.columns = temp.columns.map(lambda x: 'Prices {} (euro/kWh)'.format(x))
    pd.concat((summary_inputs, t, temp), axis=1).to_csv(os.path.join(path, 'input.csv'))

    logger.info('Creating AgentBuildings object')
    renovation_rate_max = 1.0
    if 'renovation_rate_max' in config.keys():
        renovation_rate_max = config['renovation_rate_max']

    calib_scale = True
    if 'calib_scale' in config.keys():
        calib_scale = config['calib_scale']

    preferences_zeros = False
    if 'preferences_zeros' in config.keys():
        preferences_zeros = config['preferences_zeros']
        if preferences_zeros:
            calib_scale = False

    debug_mode = False
    if 'debug_mode' in config.keys():
        debug_mode = config['debug_mode']

    with open(os.path.join(path, 'config.json'), 'w') as fp:
        json.dump(config, fp)

    buildings = AgentBuildings(stock, parsed_inputs['surface'], parsed_inputs['ratio_surface'], parsed_inputs['efficiency'],
                               parsed_inputs['income'], parsed_inputs['consumption_ini'], path, parsed_inputs['preferences'],
                               parsed_inputs['performance_insulation'],
                               year=year, demolition_rate=parsed_inputs['demolition_rate'],
                               endogenous=config['endogenous'], logger=logger)

    return buildings, energy_prices, taxes, parsed_inputs, parsed_inputs['cost_heater'], parsed_inputs['ms_heater'], \
           parsed_inputs['cost_insulation'], parsed_inputs['ms_intensive'], parsed_inputs[
               'renovation_rate_ini'], policies_heater, policies_insulation


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

    try:
        logger.info('Reading input')

        buildings, energy_prices, taxes, parsed_inputs, cost_heater, ms_heater, cost_insulation, ms_intensive, renovation_rate_ini, policies_heater, policies_insulation = initialize(
            config, path, logger)

        output, stock = pd.DataFrame(), pd.DataFrame()
        logger.info('Calibration energy consumption {}'.format(buildings.first_year))
        buildings.calculate(energy_prices.loc[buildings.first_year, :], taxes)
        s, o = buildings.parse_output_run(parsed_inputs)
        stock = pd.concat((stock, s), axis=1)
        output = pd.concat((output, o), axis=1)

        for year in range(config['start'] + 1, config['end']):
            start = time()
            logger.info('Run {}'.format(year))
            buildings.year = year
            buildings.add_flows([- buildings.flow_demolition()])
            flow_retrofit = buildings.flow_retrofit(energy_prices.loc[year, :], cost_heater, cost_insulation,
                                                    policies_heater=[p for p in policies_heater if (year >= p.start) and (year < p.end)],
                                                    policies_insulation=[p for p in policies_insulation if (year >= p.start) and (year < p.end)],
                                                    ms_insulation=ms_intensive,
                                                    renovation_rate_ini=renovation_rate_ini,
                                                    target_freeriders=config['target_freeriders'],
                                                    ms_heater=ms_heater)
            buildings.add_flows([flow_retrofit, parsed_inputs['flow_built'].loc[:, year]])
            buildings.calculate(energy_prices.loc[year, :], taxes)
            logger.info('Writing output')
            s, o = buildings.parse_output_run(parsed_inputs)
            stock = pd.concat((stock, s), axis=1)
            stock.index.names = s.index.names
            output = pd.concat((output, o), axis=1)
            logger.info('Run time {}: {:,.0f} seconds.'.format(year, round(time() - start, 2)))

        logger.info('Dumping output in {}'.format(path))
        output.round(3).to_csv(os.path.join(path, 'output.csv'))
        stock.round(2).to_csv(os.path.join(path, 'stock.csv'))
        plot_scenario(output, stock, buildings)

        return os.path.basename(os.path.normpath(path)), output, stock

    except Exception as e:
        logger.exception(e)
        # logging.error(traceback.format_exc())
        raise e
