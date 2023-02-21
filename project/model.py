import os
import pandas as pd
import logging
from time import time
import json
from importlib import resources
from pickle import load, dump

from project.building import AgentBuildings
from project.read_input import read_stock, read_policies, read_inputs, parse_inputs, dump_inputs, create_simple_policy
from project.write_output import plot_scenario, compare_results
from project.utils import reindex_mi, deciles2quintiles_pandas, deciles2quintiles_dict, get_json
from project.input.resources import resources_data

LOG_FORMATTER = '%(asctime)s - %(process)s - %(name)s - %(levelname)s - %(message)s'


def create_logger(path=None):
    """Create logger for one run.

    Parameters
    ----------
    path: str

    Returns
    -------
    Logger
    """
    if path is None:
        name = ''
    else:
        name = path.split('/')[-1].lower()

    logger = logging.getLogger('log_{}'.format(name))
    logger.setLevel('DEBUG')
    logger.propagate = False
    # remove existing handlers
    logger.handlers.clear()
    # consoler handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(LOG_FORMATTER))
    logger.addHandler(console_handler)
    # file handler
    if path is not None:
        file_handler = logging.FileHandler(os.path.join(path, 'log.log'))
        file_handler.setFormatter(logging.Formatter(LOG_FORMATTER))
        logger.addHandler(file_handler)
    return logger


def get_config() -> dict:
    with resources.path('project.config', 'config.json') as f:
        with open(f) as file:
            return json.load(file)['Reference']


def config2inputs(config=None, building_stock=None, end=None):
    """Create main Python object from configuration file.

    Parameters
    ----------
    config: dict
    building_stock: str, optional
        Path to other building stock than reference.
    end: int, optional

    Returns
    -------

    """

    if config is None:
        config = get_config()

    if building_stock is not None:
        config['building_stock'] = building_stock

    if end is not None:
        config['end'] = end

    year = config['start']
    stock = read_stock(config)
    inputs = read_inputs(config)

    if isinstance(config['policies'], str):
        config['policies'] = get_json(config['policies'])['policies']

    if config['simple'].get('heating_system'):
        replace = config['simple']['heating_system']
        shape_ini = stock.shape[0]
        stock = stock.reset_index('Heating system')
        stock['Heating system'] = stock['Heating system'].replace(replace)
        stock = stock.dropna()
        stock = stock.set_index('Heating system', append=True).squeeze()
        stock = stock.groupby(stock.index.names).sum()
        shape_out = stock.shape[0]
        print('From {} to {}'.format(shape_ini, shape_out))
        to_drop = []
        if 'Heating-District heating' in config['simple']['heating_system'].keys():
            if config['simple']['heating_system']['Heating-District heating'] is None:
                inputs['district_heating'] = None
                to_drop += ['Heating-District heating']
        to_replace = [k for k, i in replace.items() if i not in inputs['ms_heater'].columns and i is not None]
        if to_replace:
            raise NotImplemented
        inputs['ms_heater'].drop([i for i in replace.keys() if i in inputs['ms_heater'].columns], axis=1, inplace=True)

    if config['simple'].get('insulation'):
        pass

    if config['simple']['surface']:
        surface = (reindex_mi(inputs['surface'], stock.index) * stock).sum() / stock.sum()
        inputs['surface'] = pd.Series(round(surface, 0), index=inputs['surface'].index)
    if config['simple']['ratio_surface']:
        ratio_surface = (reindex_mi(inputs['ratio_surface'], stock.index).T * stock).T.sum() / stock.sum()
        for idx in inputs['ratio_surface'].index:
            inputs['ratio_surface'].loc[idx, :] = ratio_surface.round(1)

    if config['simple'].get('no_policies'):
        for name, policy in config['policies'].items():
            if policy['end'] > policy['start']:
                policy['end'] = config['start'] + 2
                config['policies'][name] = policy

    if config['simple'].get('policies'):
        for name, policy in config['policies'].items():
            if policy['start'] > config['start'] + 1:
                policy['end'] = policy['start']

            if policy['end'] > policy['start']:
                policy['end'] = config['end']
            config['policies'][name] = policy

    policies_heater, policies_insulation, taxes = read_policies(config)

    if config['simple']['quintiles']:
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

    inputs = deciles2quintiles_dict(inputs)

    stock = deciles2quintiles_pandas(stock, func='sum')

    for policy in policies_insulation + policies_heater:
        attributes = [a for a in dir(policy) if not a.startswith('__') and getattr(policy, a) is not None]
        for att in attributes:
            item = getattr(policy, att)
            if isinstance(item, (pd.Series, pd.DataFrame)):
                setattr(policy, att, deciles2quintiles_pandas(item, func='mean'))
            if isinstance(item, dict):
                new_item = {k: deciles2quintiles_pandas(i, func='mean') for k, i in item.items()}
                setattr(policy, att, new_item)

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
            'health_expenditure', 'mortality_cost', 'loss_well_being', 'carbon_value_kwh', 'carbon_value']

    return {key: item for key, item in parsed_inputs.items() if key in vars}


def get_inputs(path=None, config=None, variables=None, building_stock=None):
    """Initialize thermal buildings object based on input dictionnary.

    Parameters
    ----------
    path: str, optional
        If None do not write output.
    config: dict, optional
        If config is None use configuration file of Reference scenario
    variables: list, optional
        'buildings', 'energy_prices', 'cost_insulation', 'carbon_emission', 'carbon_value_kwh', 'health_cost'
    building_stock

    Returns
    -------
    dict
    """
    if variables is None:
        variables = ['buildings', 'energy_prices', 'cost_insulation', 'carbon_emission', 'carbon_value_kwh',
                     'health_cost', 'income']

    inputs, stock, year, policies_heater, policies_insulation, taxes = config2inputs(config, building_stock=building_stock)
    buildings, energy_prices, taxes, post_inputs, cost_heater, lifetime_heater, ms_heater, cost_insulation, calibration_intensive, calibration_renovation, demolition_rate, flow_built, financing_cost, technical_progress, consumption_ini = initialize(
        inputs, stock, year, taxes, path=path, config=config)
    output = {'buildings': buildings,
              'income': inputs['income'],
              'energy_prices': energy_prices,
              'cost_insulation': cost_insulation,
              'carbon_emission': post_inputs['carbon_emission'],
              'carbon_value_kwh': post_inputs['carbon_value_kwh'],
              'health_cost': post_inputs['health_expenditure'] + post_inputs['mortality_cost'] + post_inputs['loss_well_being'],
              'efficiency': inputs['efficiency'],
              'performance_insulation': inputs['performance_insulation']
              }
    output = {k: item for k, item in output.items() if k in variables}

    return output


def initialize(inputs, stock, year, taxes, path=None, config=None, logger=None):
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
    if path is not None and config.get('full_output') is True:
        dump_inputs(parsed_inputs, path)
    post_inputs = select_post_inputs(parsed_inputs)
    if logger is None and config.get('full_output') is True:
        logger = create_logger(path)
    logger.info('Creating AgentBuildings object')

    if path is not None:
        with open(os.path.join(path, 'config.json'), 'w') as fp:
            json.dump(config, fp)

    buildings = AgentBuildings(stock, parsed_inputs['surface'], parsed_inputs['ratio_surface'], parsed_inputs['efficiency'],
                               parsed_inputs['income'], parsed_inputs['preferences'],
                               parsed_inputs['performance_insulation'], path=path,
                               year=year,
                               endogenous=config['renovation']['endogenous'],
                               exogenous=config['renovation']['exogenous'],
                               logger=logger,
                               quintiles=config['simple']['quintiles'],
                               full_output=config.get('full_output'),
                               financing_cost=config.get('financing_cost'),
                               debug_mode=config.get('debug_mode'),
                               threshold=config['renovation'].get('threshold'),
                               resources_data=resources_data)

    technical_progress = None
    if 'technical_progress' in parsed_inputs.keys():
        technical_progress = parsed_inputs['technical_progress']

    return buildings, parsed_inputs['energy_prices'], parsed_inputs['taxes'], post_inputs, parsed_inputs['cost_heater'], parsed_inputs['lifetime_heater'], parsed_inputs['ms_heater'], \
           parsed_inputs['cost_insulation'], parsed_inputs['calibration_intensive'], parsed_inputs[
               'calibration_renovation'], parsed_inputs['demolition_rate'], parsed_inputs['flow_built'], parsed_inputs.get('input_financing'), technical_progress, parsed_inputs['consumption_ini']


def stock_turnover(buildings, prices, taxes, cost_heater, lifetime_heater, cost_insulation, p_heater, p_insulation, flow_built, year,
                   post_inputs,  ms_heater=None,  calib_intensive=None, calib_renovation=None, financing_cost=None,
                   prices_before=None, climate=None, district_heating=None, step=1, demolition_rate=None):
    """Update stock vintage due to renovation, demolition and construction.
    
    
    Parameters
    ----------
    buildings
    prices
    taxes
    cost_heater
    lifetime_heater: pd.Series
    cost_insulation
    p_heater
    p_insulation
    flow_built
    year
    post_inputs
    ms_heater
    calib_intensive
    calib_renovation
    financing_cost
    prices_before
    climate: int, optional

    Returns
    -------
    buildings : AgentBuildings
        Updated AgentBuildings object.
    stock: : Series
    output : Series
    """

    buildings.logger.info('Run {}'.format(year))
    if prices_before is None:
        prices_before = prices

    buildings.year = year
    if demolition_rate is not None:
        buildings.add_flows([- buildings.flow_demolition(demolition_rate, step=step)])
    buildings.logger.info('Calculation retrofit')
    buildings.consumption_before_retrofit = buildings.store_consumption(prices)
    flow_retrofit = buildings.flow_retrofit(prices, cost_heater, lifetime_heater, cost_insulation,
                                            policies_heater=p_heater,
                                            policies_insulation=p_insulation,
                                            calib_renovation=calib_renovation,
                                            calib_intensive=calib_intensive,
                                            ms_heater=ms_heater,
                                            district_heating=district_heating,
                                            financing_cost=financing_cost,
                                            climate=climate,
                                            step=step)

    buildings.add_flows([flow_retrofit])

    flows_obligation = buildings.flow_obligation(p_insulation, prices, cost_insulation,
                                                 financing_cost=financing_cost)
    if flows_obligation is not None:
        buildings.add_flows(flows_obligation)

    buildings.store_information_retrofit(prices)

    if flow_built is not None:
        buildings.add_flows([flow_built])

    buildings.logger.info('Writing output')
    if buildings.full_output:
        stock, output = buildings.parse_output_run(prices, post_inputs, climate=climate, step=step, taxes=taxes)
    else:
        stock = buildings.simplified_stock().rename(year)
        output = buildings.heat_consumption_energy.rename(year) / 10 ** 9
        output.index = output.index.map(lambda x: 'Consumption {} (TWh)'.format(x))

    return buildings, stock, output


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
        buildings, energy_prices, taxes, post_inputs, cost_heater, lifetime_heater, ms_heater, cost_insulation, calibration_intensive, calibration_renovation, demolition_rate, flow_built, financing_cost, technical_progress, consumption_ini = initialize(
            inputs, stock, year, taxes, path=path, config=config, logger=logger)

        output, stock = pd.DataFrame(), pd.DataFrame()
        buildings.logger.info('Calibration energy consumption {}'.format(buildings.first_year))
        buildings.calibration_consumption(energy_prices.loc[buildings.first_year, :], consumption_ini)
        s, o = buildings.parse_output_run(energy_prices.loc[buildings.first_year, :], post_inputs)
        stock = pd.concat((stock, s), axis=1)
        output = pd.concat((output, o), axis=1)

        if config.get('calibration'):
            with open(config['calibration'], "rb") as file:
                calibration = load(file)
                buildings.calibration_exogenous(**calibration)

        timestep = 1
        if config.get('step'):
            timestep = config['step']
        years = [config['start'] + 1]
        years += range(config['start'] + 2, config['end'], timestep)
        if config['end'] - 1 not in years:
            years.append(config['end'] - 1)

        for k, year in enumerate(years):
            start = time()

            if year == config['start'] + 1:
                yrs = [year]
            else:
                yrs = range(years[k - 1] + 1, year + 1)
            step = len(yrs)

            prices = energy_prices.loc[year, :]
            p_heater = [p for p in policies_heater if (year >= p.start) and (year < p.end)]
            p_insulation = [p for p in policies_insulation if (year >= p.start) and (year < p.end)]
            f_built = flow_built.loc[:, yrs]
            if isinstance(f_built, pd.DataFrame):
                f_built = f_built.sum(axis=1).rename(year)

            if technical_progress is not None:
                if technical_progress.get('insulation') is not None:
                    cost_insulation *= (1 + technical_progress['insulation'].loc[year])**step
                if technical_progress.get('heater') is not None:
                    heat_pump = ['Electricity-Heat pump air', 'Electricity-Heat pump water']
                    cost_heater.loc[heat_pump] *= (1 + technical_progress['heater'].loc[year])**step

            buildings, s, o = stock_turnover(buildings, prices, taxes, cost_heater, lifetime_heater,
                                             cost_insulation, p_heater, p_insulation, f_built, year, post_inputs,
                                             calib_intensive=inputs['calibration_intensive'],
                                             calib_renovation=inputs['calibration_renovation'],
                                             ms_heater=ms_heater, financing_cost=financing_cost,
                                             climate=config.get('climate'),
                                             district_heating=inputs.get('district_heating'),
                                             step=step, demolition_rate=demolition_rate)

            stock = pd.concat((stock, s), axis=1)
            stock.index.names = s.index.names
            output = pd.concat((output, o), axis=1)
            buildings.logger.info('Run time {}: {:,.0f} seconds.'.format(year, round(time() - start, 2)))
            if year == 2019 and buildings.full_output:
                compare_results(o, buildings.path)
                with open(os.path.join(buildings.path_calibration, 'calibration.pkl'), 'wb') as file:
                    dump({
                        'coefficient_global': buildings.coefficient_global,
                        'coefficient_heater': buildings.coefficient_heater,
                        'constant_insulation_extensive': buildings.constant_insulation_extensive,
                        'constant_insulation_intensive': buildings.constant_insulation_intensive,
                        'constant_heater': buildings.constant_heater,
                        'scale': buildings.scale
                    }, file)

        if path is not None:
            buildings.logger.info('Dumping output in {}'.format(path))
            if not buildings.full_output:
                # for price elasticity calculation
                temp = energy_prices.T
                temp.index = temp.index.map(lambda x: 'Prices {} (euro/kWh)'.format(x))
                output = pd.concat((output, temp), axis=0)
                output = output.sort_index(axis=1)

            output.round(3).to_csv(os.path.join(path, 'output.csv'))
            if buildings.full_output:
                stock.round(2).to_csv(os.path.join(path, 'stock.csv'))
                plot_scenario(output, stock, buildings)

        return os.path.basename(os.path.normpath(path)), output, stock

    except Exception as e:
        logger.exception(e)
        raise e


def cost_curve(consumption_saved, cost_insulation, percent=True, marginal=False, consumption_before=None):
    """Create cost curve.

    Parameters
    ----------
    consumption_before
    consumption_saved
    cost_insulation
    percent: bool, default True
    marginal: bool, default False

    Returns
    -------

    """

    insulation = {'Wall': (True, False, False, False), 'Floor': (False, True, False, False),
                  'Roof': (False, False, True, False), 'Windows': (False, False, False, True)}
    insulation = pd.MultiIndex.from_frame(pd.DataFrame(insulation))

    consumption_saved = consumption_saved.loc[:, insulation]
    cost_insulation = cost_insulation.loc[:, insulation]

    cost_efficiency = cost_insulation / consumption_saved

    x = consumption_saved.stack(consumption_saved.columns.names).squeeze().rename('Consumption saved')
    y = cost_efficiency.stack(cost_efficiency.columns.names).squeeze().rename('Cost efficiency (euro/kWh/year)')
    c = (x * y).rename('Cost (Billion euro)') / 10**9
    df = pd.concat((x, y, c), axis=1)

    # sort by marginal cost
    df.sort_values(y.name, inplace=True)

    if percent is True:
        if consumption_before is None:
            raise AttributeError
        df[x.name] = x / consumption_before.sum()
        # x.name = '{} (%/initial)'.format(x.name)
    else:
        df[x.name] /= 10**9
        # x.name = '{} (TWh/an)'.format(x.name)

    if marginal is False:
        df['{} cumulated'.format(x.name)] = df[x.name].cumsum()
        df['{} cumulated'.format(c.name)] = df[c.name].cumsum()
        df.dropna(inplace=True)
        df = df.set_index('{} cumulated'.format(x.name))['{} cumulated'.format(c.name)]
    else:
        df.dropna(inplace=True)
        df[y.name] = df[y.name].round(1)
        df = df.groupby([y.name]).agg({x.name: 'sum', y.name: 'first'})
        df = df.set_index(x.name)[y.name]
    return df


def calibration_res_irf(path, config=None):
    """Calibrate Res-IRF and returns calibrated parameters.
    Function is useful for running multiple scenarios with the same calibration.
    Typical example is for sensitivity analysis or elasticity calculation.
    Parameters
    ----------
    config
    path
    Returns
    -------
    """
    if not os.path.isdir(path):
        os.mkdir(path)
    logger = create_logger(path)
    try:
        logger.info('Reading input')
        inputs, stock, year, policies_heater, policies_insulation, taxes = config2inputs(config)
        buildings, energy_prices, taxes, post_inputs, cost_heater, lifetime_heater, ms_heater, cost_insulation, calibration_intensive, calibration_renovation, demolition_rate, flow_built, financing_cost, technical_progress, consumption_ini = initialize(
            inputs, stock, year, taxes, path=path, config=config, logger=logger)

        buildings.logger.info('Calibration energy consumption {}'.format(buildings.first_year))
        buildings.calibration_consumption(energy_prices.loc[buildings.first_year, :], consumption_ini)

        output = pd.DataFrame()
        _, o = buildings.parse_output_run(energy_prices.loc[buildings.first_year, :], post_inputs)
        output = pd.concat((output, o), axis=1)

        year = buildings.first_year + 1
        prices = energy_prices.loc[year, :]
        p_heater = [p for p in policies_heater if (year >= p.start) and (year < p.end)]
        p_insulation = [p for p in policies_insulation if (year >= p.start) and (year < p.end)]
        f_built = flow_built.loc[:, year]

        buildings, s, o = stock_turnover(buildings, prices, taxes, cost_heater, lifetime_heater,
                                         cost_insulation, p_heater,
                                         p_insulation, f_built, year, post_inputs,
                                         calib_intensive=inputs['calibration_intensive'],
                                         calib_renovation=inputs['calibration_renovation'],
                                         ms_heater=ms_heater, financing_cost=financing_cost,
                                         demolition_rate=demolition_rate)

        output = pd.concat((output, o), axis=1)
        output.to_csv(os.path.join(buildings.path, 'output_calibration.csv'))

        if year == 2019 and buildings.full_output:
            compare_results(o, buildings.path)

        calibration = {
            'coefficient_global': buildings.coefficient_global,
            'coefficient_heater': buildings.coefficient_heater,
            'constant_heater': buildings.constant_heater,
            'constant_insulation_intensive': buildings.constant_insulation_intensive,
            'constant_insulation_extensive': buildings.constant_insulation_extensive,
            'scale': buildings.scale,
            'threshold_indicator': buildings.threshold_indicator
        }

        return calibration

    except Exception as e:
        logger.exception(e)
        raise e


def social_planner(aggregation_archetype=None, climate=2006, smooth=False, building_stock='medium_3', freq='hour',
                   percent=True, marginal=False, hourly_profile=None):
    """Function used when coupling with power system model.


    Parameters
    ----------
    aggregation_archetype
    climate
    smooth
    building_stock: optional, {'medium_1', 'medium_3', 'medium_5', 'simple_1', 'simple_3', 'simple_5'}
        Numbers of clusters + heterogeneity of u values.
    freq: optional, {'hour', 'day', 'month', 'year'}
    percent: bool, default True
    marginal: bool, default
    hourly_profile

    Returns
    -------

    """
    resirf_inputs = get_inputs(variables=['buildings', 'energy_prices', 'cost_insulation'],
                               building_stock=os.path.join('project', 'input', 'stock', 'buildingstock_sdes2018_{}.csv'.format(building_stock)))
    buildings = resirf_inputs['buildings']
    energy_prices = resirf_inputs['energy_prices']
    cost_insulation = resirf_inputs['cost_insulation']

    heating_need = buildings.need_heating(freq=freq, climate=climate, smooth=smooth, hourly_profile=hourly_profile)
    heating_need_class = heating_need.sum(axis=1) / (buildings.stock * reindex_mi(buildings._surface, buildings.stock.index))

    insulation_class = heating_need_class.copy()
    insulation_class[insulation_class <= 100] = 1
    insulation_class[(insulation_class > 100) & (insulation_class <= 200)] = 2
    insulation_class[(insulation_class > 200) & (insulation_class <= 300)] = 3
    insulation_class[insulation_class > 300] = 4
    insulation_class = insulation_class.astype(str).rename('Insulation')

    wall_class = heating_need_class.copy()
    wall_class[wall_class.index.get_level_values('Wall') < 1] = 1
    wall_class[(wall_class.index.get_level_values('Wall') >= 1) & (wall_class.index.get_level_values('Wall') < 2)] = 2
    wall_class[wall_class.index.get_level_values('Wall') >= 2] = 3
    wall_class = wall_class.astype(str).rename('Wall class')

    heating_intensity = buildings.to_heating_intensity(heating_need.index, energy_prices.loc[buildings.first_year, :])

    heating_need = (heating_intensity * heating_need.T).T

    output = buildings.mitigation_potential(energy_prices, cost_insulation)

    consumption_saved = output['Need saved (kWh/segment)']

    cost_insulation = output['Cost insulation (euro/segment)']
    cost_insulation[consumption_saved == 0] = 0

    consumption_before = output['Need before (kWh/segment)']

    if aggregation_archetype is not None:
        if 'Performance' in aggregation_archetype:
            heating_need = buildings.add_certificate(heating_need)

            consumption_saved = buildings.add_certificate(consumption_saved)
            consumption_saved.columns = output['Need saved (kWh/segment)'].columns

            cost_insulation = buildings.add_certificate(cost_insulation)
            cost_insulation.columns = output['Cost insulation (euro/segment)'].columns

            consumption_before = buildings.add_certificate(consumption_before)

        if 'Energy' in aggregation_archetype:
            heating_need = buildings.add_energy(heating_need)

            consumption_saved = buildings.add_energy(consumption_saved)
            consumption_saved.columns = output['Need saved (kWh/segment)'].columns

            cost_insulation = buildings.add_energy(cost_insulation)
            cost_insulation.columns = output['Cost insulation (euro/segment)'].columns

            consumption_before = buildings.add_energy(consumption_before)

        if 'Insulation' in aggregation_archetype:
            heating_need = pd.concat((heating_need, insulation_class), axis=1).set_index('Insulation', append=True)

            consumption_saved = pd.concat((consumption_saved, insulation_class), axis=1).set_index('Insulation', append=True)
            consumption_saved.columns = output['Need saved (kWh/segment)'].columns

            cost_insulation = pd.concat((cost_insulation, insulation_class), axis=1).set_index('Insulation', append=True)
            cost_insulation.columns = output['Cost insulation (euro/segment)'].columns

            consumption_before = pd.concat((consumption_before, insulation_class), axis=1).set_index('Insulation', append=True).squeeze()

        if 'Wall class' in aggregation_archetype:
            heating_need = pd.concat((heating_need, wall_class), axis=1).set_index('Wall class', append=True)

            consumption_saved = pd.concat((consumption_saved, wall_class), axis=1).set_index('Wall class',
                                                                                                   append=True)
            consumption_saved.columns = output['Need saved (kWh/segment)'].columns

            cost_insulation = pd.concat((cost_insulation, wall_class), axis=1).set_index('Wall class',
                                                                                               append=True)
            cost_insulation.columns = output['Cost insulation (euro/segment)'].columns

            consumption_before = pd.concat((consumption_before, wall_class), axis=1).set_index('Wall class',
                                                                                                     append=True).squeeze()

    dict_cost, dict_heat = dict(), dict()
    if aggregation_archetype is not None:
        dict_cost = {n: cost_curve(g, cost_insulation.loc[g.index, :], percent=percent,
                                   marginal=marginal, consumption_before=consumption_before.loc[g.index]) for n, g in consumption_saved.groupby(aggregation_archetype)}
        heating_need_grouped = heating_need.groupby(aggregation_archetype).sum()
        dict_heat = {i: heating_need_grouped.loc[i, :] for i in heating_need_grouped.index}
    else:
        dict_cost['global'] = cost_curve(consumption_saved, cost_insulation, marginal=marginal,
                                         percent=percent, consumption_before=consumption_before)
        dict_heat['global'] = heating_need.sum()

    return dict_cost, dict_heat


if __name__ == '__main__':
    _dict_cost, _dict_heat = social_planner(aggregation_archetype=None, building_stock='medium_5', freq='hour',
                                            percent=False)


    """
    resirf_inputs = get_inputs(variables=['buildings', 'energy_prices', 'income'],
                               building_stock=os.path.join('project', 'input', 'stock', 'buildingstock_example.csv'))
    _buildings = resirf_inputs['buildings']
    _prices = resirf_inputs['energy_prices']
    _income = resirf_inputs['income']
    """



    """from utils import make_plots
    hourly_profile = [0.035, 0.039, 0.041, 0.042, 0.046, 0.05, 0.055, 0.058, 0.053, 0.049, 0.045, 0.041, 0.037, 0.034,
     0.03, 0.033, 0.037, 0.042, 0.046, 0.041, 0.037, 0.034, 0.033, 0.042]
    hourly_profile = pd.Series(hourly_profile, index=pd.TimedeltaIndex(range(0, 24), unit='h'))

    dict_cost, dict_heat = social_planner(aggregation_archetype=None, building_stock='medium_5', freq='hour',
                                          percent=False, marginal=True, hourly_profile=hourly_profile)
    make_plots(dict_cost, 'Cost (Billion euro)')"""

    """buildings = get_inputs(variables=['buildings'])['buildings']

    h_month = buildings.heating_need(climate=2006, smooth=False, freq='month')
    h_year = buildings.heating_need(climate=2006, smooth=False, freq='year')

    h_month = buildings.heating_need(climate=2006, smooth=False, freq='month')
    h_day = buildings.heating_need(climate=2006, smooth=False, freq='day')
    h_hour = buildings.heating_need(climate=2006, smooth=False, freq='hour')"""





