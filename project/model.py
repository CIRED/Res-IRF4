import os
import pandas as pd
import logging
from time import time
import json
from importlib import resources
from pickle import load, dump
import psutil

from project.building import AgentBuildings
from project.read_input import read_stock, read_policies, read_inputs, parse_inputs, dump_inputs, create_simple_policy
from project.write_output import plot_scenario, compare_results
from project.utils import reindex_mi, deciles2quintiles_pandas, deciles2quintiles_dict, get_json, get_size, size_dict, make_policies_tables, subplots_attributes, plot_thermal_insulation, parse_policies
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

    if config.get('policies') is not None:
        parse_policies(config)

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

        list_heater = list(stock.index.get_level_values('Heating system').unique())
        list_heater.extend([i if i not in replace.keys() else replace[i] for i in inputs['ms_heater'].columns])
        list_heater = list(set(list_heater))
        list_heater = [i for i in list_heater if i is not None]

        if isinstance(inputs['ms_heater'], pd.DataFrame):
            inputs['ms_heater'].drop([i for i in inputs['ms_heater'].columns if i not in list_heater], axis=1, inplace=True)
            inputs['ms_heater'].drop([i for i in inputs['ms_heater'].index.get_level_values('Heating system') if i not in list_heater], axis=0, inplace=True, level='Heating system')
            inputs['ms_heater'] = (inputs['ms_heater'].T / inputs['ms_heater'].sum(axis=1)).T

        inputs['cost_heater'].drop([i for i in inputs['cost_heater'].index if i not in list_heater], inplace=True)

    if config['simple'].get('insulation'):
        pass

    if config['simple'].get('no_heating_switch'):
        inputs['lifetime_heater'].loc[:] = float('inf')

    if config['simple']['surface']:
        surface = (reindex_mi(inputs['surface'], stock.index) * stock).sum() / stock.sum()
        inputs['surface'] = pd.Series(round(surface, 0), index=inputs['surface'].index)

    if config['simple']['ratio_surface']:
        ratio_surface = (reindex_mi(inputs['ratio_surface'], stock.index).T * stock).T.sum() / stock.sum()
        for idx in inputs['ratio_surface'].index:
            inputs['ratio_surface'].loc[idx, :] = ratio_surface.round(1)

    if config['simple'].get('emission_constant'):
        inputs['carbon_emission'] = pd.concat([inputs['carbon_emission'].loc[config['start'], :]] * inputs['carbon_emission'].shape[0], axis=1,
                                              keys=inputs['carbon_emission'].index).T

    if config['simple'].get('no_natural_replacement'):
        inputs['demolition_rate'] = None
        config['macro']['construction'] = False

    if config['simple'].get('no_policy'):
        for name, policy in config['policies'].items():
            if policy['end'] > policy['start']:
                policy['end'] = config['start'] + 2
                config['policies'][name] = policy

    if config['simple'].get('current_policies'):
        for name, policy in config['policies'].items():
            # remove policies that start after calibration
            if policy['start'] > config['start'] + 1:
                policy['end'] = policy['start']

            if policy['end'] > policy['start']:
                policy['end'] = config['end']

            if policy.get('growth_insulation'):
                policy['growth_insulation'] = None

            for k in policy.keys():
                if isinstance(policy[k], dict):
                    if policy[k].get('start') is not None:
                        if policy[k]['start'] > config['start'] + 1:
                            policy[k]['end'] = policy[k]['start']
                        if policy[k]['end'] > policy[k]['start']:
                            policy[k]['end'] = config['end']

            config['policies'][name] = policy

    if config.get('policies') is not None:
        policies_heater, policies_insulation, taxes = read_policies(config)
    else:
        policies_insulation, policies_heater, taxes = [], [], []

    policies_insulation = [p for p in policies_insulation if p.end > p.start]
    policies_heater = [p for p in policies_heater if p.end > p.start]

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


def memory_object(buildings):
    temp = {}
    for k, item in buildings.__dict__.items():
        if isinstance(item, dict):
            temp.update(item)
        else:
            temp.update({k: item})

    return temp


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
            'health_expenditure', 'mortality_cost', 'loss_well_being', 'carbon_value_kwh', 'carbon_value',
            'use_subsidies', 'health_cost', 'implicit_discount_rate']

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
    inputs_dynamics = initialize(inputs, stock, year, taxes, path=path, config=config)
    output = {'buildings': inputs_dynamics['buildings'],
              'income': inputs['income'],
              'energy_prices': inputs_dynamics['energy_prices'],
              'cost_insulation': inputs_dynamics['cost_insulation'],
              'carbon_emission': inputs_dynamics['post_inputs']['carbon_emission'],
              'carbon_value_kwh': inputs_dynamics['post_inputs']['carbon_value_kwh'],
              'health_cost': inputs_dynamics['post_inputs']['health_expenditure'] + inputs_dynamics['post_inputs']['mortality_cost'] + inputs_dynamics['post_inputs']['loss_well_being'],
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
    if path is not None:
        dump_inputs(parsed_inputs, path)
    post_inputs = select_post_inputs(parsed_inputs)
    if logger is None:
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
                               financing_cost=config.get('financing_cost'),
                               rational_behavior=parsed_inputs['rational_behavior'],
                               resources_data=resources_data,
                               expected_utility=config['renovation'].get('expected_utility'))

    technical_progress = None
    if 'technical_progress' in parsed_inputs.keys():
        technical_progress = parsed_inputs['technical_progress']

    inputs_dynamic = {
        'buildings': buildings,
        'energy_prices': parsed_inputs['energy_prices'],
        'taxes': parsed_inputs['taxes'],
        'post_inputs': post_inputs,
        'cost_heater': parsed_inputs['cost_heater'],
        'lifetime_heater': parsed_inputs['lifetime_heater'],
        'ms_heater': parsed_inputs['ms_heater'],
        'cost_insulation': parsed_inputs['cost_insulation'],
        'calibration_intensive': parsed_inputs['calibration_intensive'],
        'calibration_renovation': parsed_inputs['calibration_renovation'],
        'demolition_rate': parsed_inputs['demolition_rate'],
        'flow_built': parsed_inputs['flow_built'],
        'financing_cost': parsed_inputs.get('input_financing'),
        'technical_progress': technical_progress,
        'consumption_ini': parsed_inputs['consumption_ini'],
        'supply': parsed_inputs['supply'],
        'premature_replacement': parsed_inputs['premature_replacement'],
        'output': config['output']
    }
    return inputs_dynamic


def stock_turnover(buildings, prices, taxes, cost_heater, lifetime_heater, cost_insulation, p_heater, p_insulation, flow_built, year,
                   post_inputs,  ms_heater=None,  calib_intensive=None, calib_renovation=None, financing_cost=None,
                   prices_before=None, climate=None, district_heating=None, step=1, demolition_rate=None, memory=False,
                   exogenous_social=None, output_details='full', premature_replacement=None, supply=None):
    """Update stock vintage due to renovation, demolition and construction.


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

    # bill rebate - taxes revenues recycling
    bill_rebate, bill_rebate_before = 0, 0
    for t in [t for t in taxes if t.recycling is not None]:
        if year - 1 in buildings.taxes_revenues.keys():
            target = (buildings.stock * reindex_mi(t.recycling, buildings.stock.index)).sum()
            bill_rebate = t.recycling * buildings.taxes_revenues[year - 1][t.name] * 10**9 / target
            buildings.bill_rebate.update({year: bill_rebate})

        if year - 1 in buildings.bill_rebate.keys():
            bill_rebate_before = buildings.bill_rebate[year - 1]

    if demolition_rate is not None:
        buildings.add_flows([- buildings.flow_demolition(demolition_rate, step=step)])
    buildings.logger.info('Calculation retrofit')
    if output_details == 'full':
        buildings.consumption_before_retrofit = buildings.store_consumption(prices_before, bill_rebate=bill_rebate_before)
    flow_retrofit = buildings.flow_retrofit(prices, cost_heater, lifetime_heater, cost_insulation,
                                            financing_cost=financing_cost,
                                            policies_heater=p_heater,
                                            policies_insulation=p_insulation,
                                            calib_renovation=calib_renovation,
                                            calib_intensive=calib_intensive,
                                            ms_heater=ms_heater,
                                            premature_replacement=premature_replacement,
                                            district_heating=district_heating,
                                            step=step,
                                            exogenous_social=exogenous_social,
                                            supply=supply,
                                            carbon_value=post_inputs['carbon_value_kwh'].loc[year, :],
                                            carbon_content=post_inputs['carbon_emission'].loc[year, :],
                                            bill_rebate=bill_rebate)

    if memory:
        memory_dict = {'Memory': '{:.1f} MiB'.format(psutil.Process().memory_info().rss / (1024 * 1024)),
                       'AgentBuildings': '{:.1f} MiB'.format(get_size(buildings) / 10 ** 6)}
        memory_dict.update(size_dict(memory_object(buildings), n=50, display=False))
        buildings.memory.update({year: memory_dict})

    buildings.add_flows([flow_retrofit])

    flows_obligation = buildings.flow_obligation(p_insulation, prices, cost_insulation,
                                                 financing_cost=financing_cost)
    if flows_obligation is not None:
        buildings.add_flows(flows_obligation)

    if flow_built is not None:
        buildings.add_flows([flow_built])

    buildings.logger.info('Writing output')
    if output_details == 'full':
        buildings.logger.debug('Full output')
        stock, output = buildings.parse_output_run(prices, post_inputs, climate=climate, step=step, taxes=taxes,
                                                   bill_rebate=bill_rebate)
    elif output_details == 'cost_benefit':
        buildings.logger.debug('Cost-benefit output')
        stock = buildings.simplified_stock().rename(year)
        output = buildings.parse_output_run_cba(prices, post_inputs, step=step, taxes=taxes, bill_rebate=bill_rebate)
    elif output_details == 'consumption':
        buildings.logger.debug('Consumption output')
        stock = buildings.simplified_stock().rename(year)
        output = buildings.parse_output_consumption(prices, bill_rebate=bill_rebate)

    else:
        raise NotImplemented('output_details should be full, cost_benefit or consumption')

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

        if False:
            policies_calibration = [p for p in policies_insulation + policies_heater if p.start < config['start'] + 2]
            if policies_calibration:
                make_policies_tables(policies_calibration, os.path.join(path, 'policies_calibration.csv'), plot=True)
        if policies_heater + policies_insulation:
            make_policies_tables(policies_heater + policies_insulation, os.path.join(path, 'policy_scenario.csv'), plot=True)

        inputs_dynamics = initialize(inputs, stock, year, taxes, path=path, config=config, logger=logger)
        buildings, energy_prices = inputs_dynamics['buildings'], inputs_dynamics['energy_prices']
        technical_progress = inputs_dynamics['technical_progress']

        output, stock = pd.DataFrame(), pd.DataFrame()
        buildings.logger.info('Calibration energy consumption {}'.format(buildings.first_year))

        if config['output'] == 'full':
            plot_thermal_insulation(buildings.stock, save=os.path.join(buildings.path_ini, 'thermal_insulation.png'))
            _stock = buildings.simplified_stock(energy_level=True)
            _stock = _stock.groupby(
                ['Occupancy status', 'Income owner', 'Income tenant', 'Housing type', 'Energy', 'Performance']).sum()
            subplots_attributes(_stock, dict_order=resources_data['index'],
                                dict_color=resources_data['colors'],
                                sharey=True, save=os.path.join(buildings.path_ini, 'stock.png'))

        if config.get('calibration'):
            with open(config['calibration'], "rb") as file:
                calibration = load(file)
                buildings.calibration_exogenous(**calibration)
        else:
            buildings.calibration_consumption(energy_prices.loc[buildings.first_year, :], inputs_dynamics['consumption_ini'])

        s, o = buildings.parse_output_run(energy_prices.loc[buildings.first_year, :], inputs_dynamics['post_inputs'])
        stock = pd.concat((stock, s), axis=1)
        output = pd.concat((output, o), axis=1)

        timestep = 1
        if config.get('step'):
            timestep = config['step']
        years = [config['start'] + 1]
        years += range(config['start'] + 2, config['end'], timestep)
        if config['end'] - 1 not in years:
            years.append(config['end'] - 1)

        if inputs_dynamics['supply']['insulation'] is not None:
            inputs_dynamics['cost_insulation'] /= inputs_dynamics['supply']['insulation']['markup_insulation']

        for k, year in enumerate(years):
            start = time()

            if year == config['end'] - 1:
                yrs = [year]
            else:
                yrs = range(year, years[k + 1])
            step = len(yrs)

            prices = energy_prices.loc[year, :]
            if year > config['start']:
                prices_before = energy_prices.loc[year - 1, :]
            else:
                prices_before = prices

            p_heater = [p for p in policies_heater if (year >= p.start) and (year < p.end)]
            p_insulation = [p for p in policies_insulation if (year >= p.start) and (year < p.end)]
            f_built = inputs_dynamics['flow_built'].loc[:, yrs]
            if isinstance(f_built, pd.DataFrame):
                f_built = f_built.sum(axis=1).rename(year)

            if technical_progress is not None:
                if technical_progress.get('insulation') is not None:
                    inputs_dynamics['cost_insulation'] *= (1 + technical_progress['insulation'].loc[year])**step
                if technical_progress.get('heater') is not None:
                    heat_pump = [i for i in resources_data['index']['Heat pumps'] if i in inputs_dynamics['cost_heater'].index]
                    inputs_dynamics['cost_heater'].loc[heat_pump] *= (1 + technical_progress['heater'].loc[year])**step

            buildings, s, o = stock_turnover(buildings, prices, taxes, inputs_dynamics['cost_heater'],
                                             inputs_dynamics['lifetime_heater'],
                                             inputs_dynamics['cost_insulation'], p_heater, p_insulation, f_built, year,
                                             inputs_dynamics['post_inputs'],
                                             calib_intensive=inputs_dynamics['calibration_intensive'],
                                             calib_renovation=inputs_dynamics['calibration_renovation'],
                                             ms_heater=inputs_dynamics['ms_heater'],
                                             premature_replacement=inputs_dynamics['premature_replacement'],
                                             financing_cost=inputs_dynamics['financing_cost'],
                                             supply=inputs_dynamics['supply'],
                                             district_heating=inputs.get('district_heating'),
                                             demolition_rate=inputs_dynamics['demolition_rate'],
                                             exogenous_social=inputs.get('exogenous_social'),
                                             output_details=config['output'],
                                             climate=config.get('climate'),
                                             prices_before=prices_before,
                                             step=step,
                                             )

            stock = pd.concat((stock, s), axis=1)
            stock.index.names = s.index.names
            output = pd.concat((output, o), axis=1)
            buildings.logger.info('Run time {}: {:,.0f} seconds.'.format(year, round(time() - start, 2)))
            if year == buildings.first_year + 1 and config['output'] == 'full':
                compare_results(o, buildings.path)

                buildings.make_static_analysis(inputs_dynamics['cost_insulation'], inputs_dynamics['cost_heater'],
                                               prices, 0.05,
                                               inputs_dynamics['post_inputs']['implicit_discount_rate'],
                                               inputs_dynamics['post_inputs']['health_cost'].sum(axis=1),
                                               inputs_dynamics['post_inputs']['carbon_value_kwh'].loc[year, :],
                                               inputs_dynamics['post_inputs']['carbon_emission'].loc[year, :])

                with open(os.path.join(buildings.path_calibration, 'calibration.pkl'), 'wb') as file:
                    dump({
                        'coefficient_global': buildings.coefficient_global,
                        'coefficient_heater': buildings.coefficient_heater,
                        'constant_insulation_extensive': buildings.constant_insulation_extensive,
                        'constant_insulation_intensive': buildings.constant_insulation_intensive,
                        'constant_heater': buildings.constant_heater,
                        'scale_insulation': buildings.scale_insulation,
                        'scale_heater': buildings.scale_heater
                    }, file)

        if path is not None:
            buildings.logger.info('Writing output in {}'.format(path))

            if buildings.memory:
                pd.DataFrame(buildings.memory).to_csv(os.path.join(path, 'memory.csv'))

            output.round(3).to_csv(os.path.join(path, 'output.csv'))
            buildings.logger.info('Dumping output in {}'.format(os.path.join(path, 'output.csv')))

            if config['output'] == 'full':
                stock.round(2).to_csv(os.path.join(path, 'stock.csv'))
                buildings.logger.info('Creating standard figures')
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
        inputs_dynamics = initialize(inputs, stock, year, taxes, path=path, config=config, logger=logger)
        buildings, energy_prices = inputs_dynamics['buildings'], inputs_dynamics['energy_prices']

        buildings.logger.info('Calibration energy consumption {}'.format(buildings.first_year))
        buildings.calibration_consumption(energy_prices.loc[buildings.first_year, :], inputs_dynamics['consumption_ini'])

        output = pd.DataFrame()
        _, o = buildings.parse_output_run(energy_prices.loc[buildings.first_year, :], inputs_dynamics['post_inputs'])
        output = pd.concat((output, o), axis=1)

        year = buildings.first_year + 1
        prices = energy_prices.loc[year, :]
        p_heater = [p for p in policies_heater if (year >= p.start) and (year < p.end)]
        p_insulation = [p for p in policies_insulation if (year >= p.start) and (year < p.end)]
        f_built = inputs_dynamics['flow_built'].loc[:, year]

        buildings, s, o = stock_turnover(buildings, prices, taxes, inputs_dynamics['cost_heater'],
                                         inputs_dynamics['lifetime_heater'],
                                         inputs_dynamics['cost_insulation'], p_heater,
                                         p_insulation, f_built, year, inputs_dynamics['post_inputs'],
                                         calib_intensive=inputs_dynamics['calibration_intensive'],
                                         calib_renovation=inputs_dynamics['calibration_renovation'],
                                         ms_heater=inputs_dynamics['ms_heater'],
                                         financing_cost=inputs_dynamics['financing_cost'],
                                         demolition_rate=inputs_dynamics['demolition_rate'],
                                         supply=inputs_dynamics['supply'],
                                         premature_replacement=inputs_dynamics['premature_replacement'],
                                         )

        output = pd.concat((output, o), axis=1)
        output.to_csv(os.path.join(buildings.path, 'output_calibration.csv'))

        if year == 2019:
            compare_results(o, buildings.path)

        calibration = {
            'coefficient_global': buildings.coefficient_global,
            'coefficient_heater': buildings.coefficient_heater,
            'constant_heater': buildings.constant_heater,
            'constant_insulation_intensive': buildings.constant_insulation_intensive,
            'constant_insulation_extensive': buildings.constant_insulation_extensive,
            'scale': buildings.scale,
            'number_firms_insulation': buildings.number_firms_insulation,
            'number_firms_heater': buildings.number_firms_heater,
            'rational_hidden_cost': buildings.rational_hidden_cost
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
    building_stock: optional
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





