import os
import pandas as pd
from building import AgentBuildings
from input.param import generic_input
from read_input import read_stock, read_policies, read_exogenous, read_revealed, parse_parameters
from write_output import plot_scenario
import logging
from time import time

LOG_FORMATTER = '%(asctime)s - %(process)s - %(name)s - %(levelname)s - %(message)s'


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

    # logging.getLogger('matplotlib.font_manager').disabled = True
    # logging.getLogger('matplotlib.axes').disabled = True

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
        stock, year = read_stock(config)
        policies_heater, policies_insulation, taxes = read_policies(config)
        param, summary_param = parse_parameters(config, generic_input, stock)
        energy_prices, energy_taxes, cost_heater, cost_insulation = read_exogenous(config)
        efficiency, choice_insulation, ms_heater, restrict_heater, choice_heater, renovation_rate_ini, ms_intensive = read_revealed(config)

        if config['prices_constant']:
            energy_prices = pd.concat([energy_prices.loc[year, :]] * energy_prices.shape[0], keys=energy_prices.index,
                                      axis=1).T

        total_taxes = pd.DataFrame(0, index=energy_prices.index, columns=energy_prices.columns)
        for t in taxes:
            total_taxes = total_taxes.add(t.value, fill_value=0)

        if energy_taxes is not None:
            total_taxes = total_taxes.add(energy_taxes, fill_value=0)

        if config['taxes_constant']:
            total_taxes = pd.concat([total_taxes.loc[year, :]] * total_taxes.shape[0], keys=total_taxes.index,
                                    axis=1).T

        energy_vta = energy_prices * generic_input['vta_energy_prices']
        total_taxes += energy_vta

        energy_prices = energy_prices.add(total_taxes, fill_value=0)
        param['energy_prices'] = energy_prices

        t = total_taxes.copy()
        t.columns = t.columns.map(lambda x:  'Taxes {} (euro/kWh)'.format(x))
        temp = energy_prices.copy()
        temp.columns = temp.columns.map(lambda x:  'Prices {} (euro/kWh)'.format(x))
        pd.concat((summary_param, t, temp), axis=1).to_csv(os.path.join(path, 'input.csv'))

        logger.info('Creating AgentBuildings object')
        buildings = AgentBuildings(stock, param['surface'], generic_input['ratio_surface'], efficiency, param['income'],
                                   param['consumption_ini'], path, param['preferences'],
                                   restrict_heater, ms_heater, choice_insulation, param['performance_insulation'],
                                   year=year, demolition_rate=param['demolition_rate'],
                                   data_calibration=param['data_ceren'], endogenous=config['endogenous'],
                                   number_exogenous=config['exogenous_detailed']['number'], logger=logger,
                                   debug_mode=config['debug_mode'])

        output, stock = pd.DataFrame(), pd.DataFrame()
        logger.info('Calibration energy consumption {}'.format(year))
        buildings.calculate(energy_prices.loc[year, :], taxes)
        s, o = buildings.parse_output_run(param)
        stock = pd.concat((stock, s), axis=1)
        output = pd.concat((output, o), axis=1)

        for year in range(config['start'] + 1, config['end']):
            start = time()
            logger.info('Run {}'.format(year))
            buildings.year = year
            buildings.add_flows([- buildings.flow_demolition()])
            flow_retrofit = buildings.flow_retrofit(energy_prices.loc[year, :], cost_heater, ms_heater, cost_insulation,
                                                    ms_intensive, renovation_rate_ini,
                                                    [p for p in policies_heater if (year >= p.start) and (year < p.end)],
                                                    [p for p in policies_insulation if (year >= p.start) and (year < p.end)],
                                                    config['target_freeriders'],
                                                    supply_constraint=config['supply_constraint'])
            buildings.add_flows([flow_retrofit, param['flow_built'].loc[:, year]])
            buildings.calculate(energy_prices.loc[year, :], taxes)
            logger.info('Writing output')
            s, o = buildings.parse_output_run(param)
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