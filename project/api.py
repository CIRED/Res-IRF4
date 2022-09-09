import os
import json
from importlib import resources

from project.read_input import read_stock, read_policies, read_exogenous, read_revealed, parse_parameters, PublicPolicy
from project.input.param import generic_input


def get_config() -> dict:
    with resources.path('project.input', 'config.json') as f:
        with open(f) as file:
            return json.load(file)['Reference']


config = get_config()

stock, year = read_stock(config)
policies_heater, policies_insulation, taxes = read_policies(config)
param, summary_param = parse_parameters(config, generic_input, stock)
energy_prices, energy_taxes, cost_heater, cost_insulation = read_exogenous(config)
efficiency, choice_insulation, ms_heater, choice_heater, renovation_rate_ini, ms_intensive = read_revealed(
    config)

print('break')
print('break')
