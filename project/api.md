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

2022-09-15 11:07:38,397 - 3638 - log_reference - INFO - Run time 2020: 23 seconds.

2022-09-15 12:07:01,038 - 4154 - log_reference - INFO - Run time 2020: 20 seconds.


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


    def calculation_indicators(surface, subsidies_total, stock):
        """NOT IMPLEMENTED YET. Mac curve and payback period investment for the calibration year

        Returns
        -------

        """
        if self._debug_mode and self.year == self.first_year + 1:
            consumption_before = None
            # consumption_before = (agent.heating_consumption_sd().T * surface).T
            consumption = (consumption_sd.T * surface).T
            consumption_saving = - consumption.T.subtract(consumption_before).T
            discount_factor = reindex_mi(self.discount_factor, bill_saved.index)
            npv = - cost_insulation + subsidies_total + (discount_factor * bill_saved.T).T

            def calculate_financial_indicator(columns, consumption, consumption_before,
                                              bill_saved, cost_insulation, subsidies_total, npv,
                                              certificate, stock, name='', discount_factor=discount_factor):

                # find result for each technology selected
                bill_saved_select, cost_insulation_select, subsidies_total_select, consumption_select, npv_select = Series(
                    dtype=float), Series(dtype=float), Series(dtype=float), Series(dtype=float), Series(dtype=float)
                for c in columns.unique():
                    idx = columns.index[columns == c]
                    consumption_select = concat((consumption_select, consumption.loc[idx, c]), axis=0)
                    bill_saved_select = concat((bill_saved_select, bill_saved.loc[idx, c]), axis=0)
                    cost_insulation_select = concat((cost_insulation_select, cost_insulation.loc[idx, c]), axis=0)
                    subsidies_total_select = concat((subsidies_total_select, subsidies_total.loc[idx, c]), axis=0)
                    npv_select = concat((npv_select, npv.loc[idx, c]), axis=0)

                # reindex
                consumption_select.index = MultiIndex.from_tuples(consumption_select.index).set_names(
                    consumption.index.names)
                bill_saved_select.index = MultiIndex.from_tuples(bill_saved_select.index).set_names(
                    bill_saved.index.names)
                subsidies_total_select.index = MultiIndex.from_tuples(subsidies_total_select.index).set_names(
                    subsidies_total.index.names)
                cost_insulation_select.index = MultiIndex.from_tuples(cost_insulation_select.index).set_names(
                    cost_insulation.index.names)
                npv_select.index = MultiIndex.from_tuples(npv_select.index).set_names(
                    npv.index.names)

                # calculate payback
                def calculate_payback(cost, subsidies, revenue, certificate, stock, name=name):
                    cash_flow = concat([revenue] * 80, keys=range(0, 80), axis=1)
                    capex = (cost.rename(0) - subsidies.rename(0)).to_frame().reindex(
                        cash_flow.columns, axis=1).fillna(0)
                    cash_flow_cumsum = (- capex + cash_flow).cumsum(axis=1)
                    mask = (cash_flow_cumsum > 0).cumsum(axis=1).eq(1)
                    payback = mask.idxmax(axis=1)
                    payback[~(mask == True).any(axis=1)] = float('nan')
                    payback = concat((certificate.rename('Peformance'), revenue.rename('Bill saved'),
                                      stock.rename('Stock'), payback.rename('Payback')),
                                     axis=1).sort_values('Payback')
                    payback['Stock cumulated'] = payback['Stock'].cumsum() / 10 ** 6

                    fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))
                    payback.plot(x='Stock cumulated', y='Payback', ax=ax, fontsize=12, figsize=(8, 10))
                    format_ax(ax)
                    ax.get_legend().remove()
                    ax.axvline(x=payback['Stock cumulated'].iloc[-1], c='red')
                    ax.set_ylabel('Payback')
                    fig.savefig(os.path.join(self.path_static, 'payback_{}.png'.format(name)))
                    plt.close(fig)

                    return payback

                payback = calculate_payback(cost_insulation_select, subsidies_total_select, bill_saved_select,
                                            certificate, stock, name='npv_{}'.format(name))

                # MAC curve
                def mac_curve(npv, consumption_before, consumption, revenue, cost, subsidies, columns, stock,
                              discount_factor=discount_factor):
                    # details
                    consumption_saving = (consumption_before - consumption)
                    details = concat((npv, consumption_saving, consumption_before, consumption,
                                      revenue, revenue * discount_factor,
                                      cost, subsidies, cost - subsidies,
                                      columns),
                                     axis=1,
                                     keys=['NPV', 'Consumption saving', 'Consumption before', 'Consumption after',
                                           'Bill saving', 'Bill saving cumac', 'Cost insulation',
                                           'Subsidies', 'Cost net', 'Insulation'])
                    details.sort_values('NPV', inplace=True, ascending=False)

                    # mac curve
                    consumption_saving = consumption_saving * stock / 10 ** 9
                    temp = concat((consumption_saving.rename('Consumption'), npv.rename('NPV')),
                                  axis=1).sort_values('NPV')
                    temp['Consumption saving cumulated'] = (temp['Consumption'].cumsum())
                    temp['NPV'] = temp['NPV'] / 10 ** 3

                    fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))
                    temp.plot(x='Consumption saving cumulated', y='NPV', ax=ax, fontsize=12, figsize=(8, 10))
                    format_ax(ax, ymin=None)
                    ax.get_legend().remove()
                    ax.axvline(x=(consumption_before * stock).sum() / 10 ** 9, c='red')
                    ax.set_ylabel('NPV (k€)')
                    ax.set_xlabel('Consumption saving cumulated (TWh)')
                    fig.savefig(os.path.join(self.path_static, 'npv_{}.png'.format(name)))
                    plt.close(fig)

                    return details

                abatement = mac_curve(npv_select, consumption_before, consumption_select, bill_saved_select,
                                      cost_insulation_select, subsidies_total_select, columns, stock,
                                      discount_factor=discount_factor)

                return payback, abatement

            insulation_private_optim = npv.idxmax(axis=1)
            _, _ = calculate_financial_indicator(insulation_private_optim, consumption, consumption_before, bill_saved,
                                                 cost_insulation, subsidies_total, npv, agent.certificate, stock,
                                                 name='private_optim')
            insulation_max_saving = consumption_saving.idxmax(axis=1)
            _, _ = calculate_financial_indicator(insulation_max_saving, consumption, consumption_before,
                                                 bill_saved, cost_insulation, subsidies_total, npv,
                                                 agent.certificate, stock, name='max_saving')


    if index is None:
        index = self.stock.index
