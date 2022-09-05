import os
import json
import logging
from model import initialize

LOG_FORMATTER = '%(asctime)s - %(process)s - %(name)s - %(levelname)s - %(message)s'

config = os.path.join('project/input/config/optim/config.json')

with open(config) as file:
    config = json.load(file)
    config = config['Reference']

path = 'output_residential'
if not os.path.isdir(path):
    os.mkdir(path)

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
def simple_resirf(sub_heater, sub_insulation, buildings, energy_price, taxes, year, param):
    """Calculate energy consumption and investment cost.

    Parameters
    ----------
    buildings: AgentBuildings
    energy_price: pd.Series
    taxes: list
    year: int
    param: dict

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



    buildings.year = year
    buildings.add_flows([- buildings.flow_demolition()])
    flow_retrofit = buildings.flow_retrofit(energy_prices.loc[year, :], cost_heater, cost_insulation,
                                            policies_heater=[p for p in policies_heater if
                                                             (year >= p.start) and (year < p.end)],
                                            policies_insulation=[p for p in policies_insulation if
                                                                 (year >= p.start) and (year < p.end)])
    buildings.add_flows([flow_retrofit, param['flow_built'].loc[:, year]])
    buildings.calculate(energy_price, taxes)

    # output
    electricity = buildings.heat_consumption_energy['Electricity'] / 10**9
    gas = buildings.heat_consumption_energy['Natural gas'] / 10**9
    investment = (buildings.investment_heater.sum().sum() + buildings.investment_insulation.sum().sum()) / 10**9
    health_cost, _ = buildings.health_cost(param)

    return electricity, gas, investment, health_cost


y = simple_resirf(buildings, energy_prices.loc[year, :], taxes, year + 1, param)
print(y)
