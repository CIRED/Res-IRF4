import os
import json
from importlib import resources

from pandas import Series, DataFrame, concat, MultiIndex, Index

from project.read_input import read_stock, read_policies, read_technical, read_revealed, parse_parameters, PublicPolicy
from project.input.param import generic_input


def get_config() -> dict:
    with resources.path('project.input', 'config.json') as f:
        with open(f) as file:
            return json.load(file)['Reference']


config = get_config()

inputs = dict()

stock, year = read_stock(config)
# inputs.update({'stock': stock})
policies_heater, policies_insulation, taxes = read_policies(config)
param, summary_param = parse_parameters(config, generic_input, stock, inputs)
energy_prices, energy_taxes, cost_heater, cost_insulation, efficiency = read_technical(config, inputs)
ms_heater, renovation_rate_ini, ms_intensive = read_revealed(config, inputs)



self.subsidies_details_insulation
self.certificate_jump_heater

2022-09-12 11:31:17,888 - 83610 - log_reference - INFO - Run time 2020: 43 seconds.

2022-09-12 11:42:08,908 - 83716 - log_reference - INFO - Run time 2020: 38 seconds.

2022-09-12 11:50:43,303 - 83920 - log_reference - INFO - Run time 2020: 32 seconds.

2022-09-12 12:51:38,277 - 87797 - log_reference - INFO - Run time 2020: 32 seconds.

2022-09-14 09:30:13,140 - 96799 - log_reference - INFO - Run time 2020: 30 seconds.

print('break')
print('break')

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