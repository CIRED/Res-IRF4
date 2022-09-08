import os
import json
import logging
from pandas import Series, Index
from model import initialize
from read_input import PublicPolicy

LOG_FORMATTER = '%(asctime)s - %(process)s - %(name)s - %(levelname)s - %(message)s'

config = os.path.join('project/input/config/optim/config.json')

with open(config) as file:
    config = json.load(file)
    config = config['Reference']

path = 'output_residential'
if not os.path.isdir(path):
    os.mkdir(path)

# logger
logger = logging.getLogger('log')
logger.setLevel('DEBUG')
logger.propagate = False
# file handler
file_handler = logging.FileHandler(os.path.join(path, 'log.log'))
file_handler.setFormatter(logging.Formatter(LOG_FORMATTER))
logger.addHandler(file_handler)
# consoler handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(LOG_FORMATTER))
logger.addHandler(console_handler)

# initializing
buildings, energy_prices, taxes, param, cost_heater, ms_heater, cost_insulation, ms_intensive, renovation_rate_ini, policies_heater, policies_insulation = initialize(
    config, path, logger)

# calibration energy consumption first year
buildings.calculate(energy_prices.loc[buildings.first_year, :], taxes)

# calibration renovation flow
year = buildings.first_year + 1
buildings.year = year
buildings.add_flows([- buildings.flow_demolition()])
flow_retrofit = buildings.flow_retrofit(energy_prices.loc[year, :], cost_heater, cost_insulation,
                                        policies_heater=[p for p in policies_heater if
                                                         (year >= p.start) and (year < p.end)],
                                        policies_insulation=[p for p in policies_insulation if
                                                             (year >= p.start) and (year < p.end)],
                                        ms_insulation=ms_intensive,
                                        renovation_rate_ini=renovation_rate_ini,
                                        target_freeriders=config['target_freeriders'],
                                        ms_heater=ms_heater)
buildings.add_flows([flow_retrofit, param['flow_built'].loc[:, year]])
buildings.calculate(energy_prices.loc[year, :], taxes)


# run
def simple_resirf(sub_heater, sub_insulation, buildings, energy_price, taxes, policies_heater,
                  policies_insulation, year, param):
    """Calculate energy consumption and investment cost.

    Parameters
    ----------
    buildings: AgentBuildings
    energy_price: pd.Series
    taxes: list
    year: int
    param: dict
    policies_heater: list
    policies_insulation: list

    Returns
    -------
    float
    Annual electricity consumption [TWh]
    gas
    Annual natural gas consumption [TWh]
    investment
    Annual retrofitting investment (heating system + insulation) [Billion euro]
    health_cost
    Annual health cost. Sum of health expenditure, social cost of mortality and loss of well-being [Billion euro]

    """

    sub_heater = Series(sub_heater, index=Index(['Electricity-Heat pump'], name='Heating system final'))
    policies_heater = [p for p in policies_heater if (year >= p.start) and (year < p.end)]
    policies_heater.append(PublicPolicy('sub_heater_optim', year, year + 1, sub_heater, 'subsidy_ad_volarem',
                                        gest='heater', by='columns'))

    policies_insulation = [p for p in policies_insulation if (year >= p.start) and (year < p.end)]
    policies_insulation.append(PublicPolicy('sub_insulation_optim', year, year + 1, sub_insulation, 'subsidy_ad_volarem',
                               gest='insulation'))

    buildings.year = year
    buildings.add_flows([- buildings.flow_demolition()])
    flow_retrofit = buildings.flow_retrofit(energy_price, cost_heater, cost_insulation,
                                            policies_heater=policies_heater,
                                            policies_insulation=policies_insulation)
    buildings.add_flows([flow_retrofit, param['flow_built'].loc[:, year]])
    buildings.calculate(energy_price, taxes)

    # output
    electricity = buildings.heat_consumption_energy['Electricity'] / 10**9
    gas = buildings.heat_consumption_energy['Natural gas'] / 10**9
    wood = buildings.heat_consumption_energy['Wood fuel'] / 10**9
    oil = buildings.heat_consumption_energy['Oil fuel'] / 10**9

    investment = (buildings.investment_heater.sum().sum() + buildings.investment_insulation.sum().sum()) / 10**9
    subsidies = (buildings.subsidies_heater.sum().sum() + buildings.subsidies_insulation.sum().sum()) / 10**9
    health_cost, _ = buildings.health_cost(param)

    return electricity, gas, wood, oil, investment, subsidies, health_cost


sub_insulation, sub_heater = 0.1, 0.1
year += 1
y = simple_resirf(sub_heater, sub_insulation, buildings, energy_prices.loc[year, :], taxes, policies_heater,
                  policies_insulation, year, param)
print(y)
