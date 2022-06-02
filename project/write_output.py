# Copyright 2020-2021 Ecole Nationale des Ponts et Chaussées
#
# This file is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Original author Lucas Vivier <vivier@centre-cired.fr>

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import pickle

from input.param import generic_input
from utils import reverse_dict, make_plot, reindex_mi, make_grouped_subplots, make_area_plot, waterfall_chart, \
    assessment_scenarios, make_stackedbar_plot

SMALL_SIZE = 10
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def parse_output(buildings, param):
    """Parse output.

    Model output
    1. Energy consumption & emission
    2. Stock performance evolution
    3. Numbers retrofit (heater & envelope component retrofit)
    3.1 By technology
    3.2 By agent : income owner, and occupation status, housing type

    4. Policies

    Parameters
    ----------
    buildings
    param

    Returns
    -------
    """
    stock = pd.DataFrame(buildings.stock_yrs).fillna(0)
    certificate = buildings.certificate.rename('Performance')
    energy = buildings.energy.rename('Energy')
    stock = pd.concat((stock, certificate, energy), axis=1).set_index(['Performance', 'Energy'], append=True)
    stock = stock.groupby(
        ['Occupancy status', 'Income owner', 'Income tenant', 'Housing type', 'Heating system', 'Performance']).sum()

    detailed = dict()
    detailed['Consumption standard (TWh)'] = pd.Series({year: buildings.heat_consumption_sd_yrs[year].sum() for year in
                                                        buildings.heat_consumption_sd_yrs.keys()}) / 10 ** 9

    consumption = pd.DataFrame(buildings.heat_consumption_calib_yrs)
    detailed['Consumption (TWh)'] = consumption.sum() / 10 ** 9

    temp = consumption.groupby(buildings.energy).sum()
    temp.index = temp.index.map(lambda x: 'Consumption {} (TWh)'.format(x))
    detailed.update(temp.T / 10**9)

    temp = consumption.groupby(buildings.certificate).sum()
    temp.index = temp.index.map(lambda x: 'Consumption {} (TWh)'.format(x))
    detailed.update(temp.T / 10 ** 9)

    emission = (consumption.groupby(buildings.energy).sum() * param['carbon_emission'].T).dropna(axis=1, how='all')
    detailed['Emission (MtCO2)'] = emission.sum() / 10 ** 12
    detailed['Cumulated emission (MtCO2)'] = detailed['Emission (MtCO2)'].cumsum()
    detailed['Stock (Million)'] = stock.sum() / 10 ** 6
    detailed['Surface (Million m2)'] = pd.DataFrame(buildings.surface_yrs).sum() / 10**6

    temp = pd.DataFrame(buildings.certificate_nb)
    temp.index = temp.index.map(lambda x: 'Stock {} (Million)'.format(x))
    detailed.update(temp.T / 10 ** 6)
    try:
        detailed['Stock efficient (Million)'] = detailed['Stock A (Million)'] + detailed['Stock B (Million)']
    except KeyError:
        detailed['Stock efficient (Million)'] = detailed['Stock B (Million)']

    detailed['Stock low-efficient (Million)'] = detailed['Stock F (Million)'] + detailed['Stock G (Million)']
    detailed['Energy poverty (Million)'] = pd.Series(buildings.energy_poverty) / 10 ** 6
    detailed['Heating intensity (%)'] = pd.Series(buildings.heating_intensity_avg)
    temp = pd.DataFrame(buildings.heating_intensity_tenant)
    temp.index = temp.index.map(lambda x: 'Heating intensity {} (%)'.format(x))
    detailed.update(temp.T)

    detailed['New efficient (Thousand)'] = pd.Series(buildings.efficient_renovation_yrs) / 10**3

    detailed['Retrofit (Thousand)'] = pd.Series(
        {year: item.sum().sum() for year, item in buildings.certificate_jump_yrs.items()}) / 10**3
    detailed['Retrofit >= 1 EPC (Thousand)'] = pd.Series(
        {year: item.loc[:, [i for i in item.columns if i > 0]].sum().sum() for year, item in
         buildings.certificate_jump_yrs.items()}) / 10 ** 3

    for i in range(6):
        temp = pd.DataFrame({year: item.loc[:, i] for year, item in buildings.certificate_jump_yrs.items()})
        detailed['Retrofit {} EPC (Thousand)'.format(i)] = temp.sum() / 10 ** 3
        detailed['Retrofit rate {} EPC (%)'.format(i)] = temp.sum() / stock.sum()

        """t = temp.groupby('Income owner').sum() / stock.groupby('Income owner').sum()
        t.index = t.index.map(lambda x: 'Retrofit rate {} EPC {} (%)'.format(i, x))
        detailed.update(t.T)

        t = temp.groupby(['Occupancy status', 'Housing type']).sum() / stock.groupby(
            ['Occupancy status', 'Housing type']).sum()
        t.index = t.index.map(lambda x: 'Retrofit rate {} EPC {} - {} (%)'.format(i, x[0], x[1]))
        detailed.update(t.T)"""

    # for replacement output need to be presented by technologies (what is used) and by agent (who change)
    replacement_heater = buildings.replacement_heater
    temp = pd.DataFrame({year: item.sum() for year, item in replacement_heater.items()})
    t = temp.copy()
    t.index = t.index.map(lambda x: 'Replacement heater {} (Thousand)'.format(x))
    detailed.update((t / 10 ** 3).T)
    detailed['Replacement heater (Thousand)'] = temp.sum() / 10 ** 3

    """t = temp * pd.DataFrame(buildings.cost_heater)
    t.index = t.index.map(lambda x: 'Investment heater {} (Million)'.format(x))
    detailed.update((t / 10**6).T)"""

    temp = pd.DataFrame({year: item.sum(axis=1) for year, item in replacement_heater.items()})
    t = temp.groupby('Income owner').sum()
    t.index = t.index.map(lambda x: 'Replacement heater {} (Thousand)'.format(x))
    detailed.update((t / 10 ** 3).T)
    t = temp.groupby(['Housing type', 'Occupancy status']).sum()
    t.index = t.index.map(lambda x: 'Replacement heater {} - {} (Thousand)'.format(x[0], x[1]))
    detailed.update((t / 10 ** 3).T)

    replacement_insulation = buildings.replacement_insulation
    temp = pd.DataFrame({year: item.sum(axis=1) for year, item in replacement_insulation.items()})
    detailed['Insulation actions (Thousand)'] = temp.sum() / 10 ** 3
    t = temp.groupby('Income owner').sum()
    t.index = t.index.map(lambda x: 'Insulation actions {} (Thousand)'.format(x))
    detailed.update(t.T / 10**3)
    t = temp.groupby(['Housing type', 'Occupancy status']).sum()
    t.index = t.index.map(lambda x: 'Insulation actions {} - {} (Thousand)'.format(x[0], x[1]))
    detailed.update((t / 10**3).T)
    t.index = t.index.str.replace('Thousand', '%')
    s = stock.groupby(['Housing type', 'Occupancy status']).sum()
    s.index = s.index.map(lambda x: 'Insulation actions {} - {} (%)'.format(x[0], x[1]))
    t = t / s
    detailed.update(t.T)

    for i in ['Wall', 'Floor', 'Roof', 'Windows']:
        temp = pd.DataFrame(
            {year: item.xs(True, level=i, axis=1).sum(axis=1) for year, item in replacement_insulation.items()})
        t = (pd.DataFrame(buildings.cost_component).loc[i, :] * temp)
        detailed['Insulation {} (Thousand)'.format(i)] = temp.sum() / 10**3

        # only work because existing surface does not change over time
        detailed['Investment {} (Billion euro)'.format(i)] = (t * reindex_mi(param['surface'], t.index)).sum() / 10 ** 9

        detailed['Embodied energy {} (TWh PE)'.format(i)] = (temp * reindex_mi(param['surface'], temp.index) *
                                                             param['embodied_energy_renovation'][i]).sum() / 10 ** 9
        detailed['Carbon footprint {} (MtCO2)'.format(i)] = (temp * reindex_mi(param['surface'], temp.index) *
                                                             param['carbon_footprint_renovation'][i]).sum() / 10 ** 9

    detailed['Embodied energy renovation (TWh PE)'] = detailed['Embodied energy Wall (TWh PE)'] + detailed[
        'Embodied energy Floor (TWh PE)'] + detailed['Embodied energy Roof (TWh PE)'] + detailed[
                                                          'Embodied energy Windows (TWh PE)']

    detailed['Embodied energy construction (TWh PE)'] = param['Embodied energy construction (TWh PE)']
    detailed['Embodied energy (TWh PE)'] = detailed['Embodied energy renovation (TWh PE)'] + detailed[
        'Embodied energy construction (TWh PE)']

    detailed['Carbon footprint renovation (MtCO2)'] = detailed['Carbon footprint Wall (MtCO2)'] + detailed[
        'Carbon footprint Floor (MtCO2)'] + detailed['Carbon footprint Roof (MtCO2)'] + detailed[
                                                          'Carbon footprint Windows (MtCO2)']

    detailed['Carbon footprint construction (MtCO2)'] = param['Carbon footprint construction (MtCO2)']
    detailed['Carbon footprint (MtCO2)'] = detailed['Carbon footprint renovation (MtCO2)'] + detailed[
        'Carbon footprint construction (MtCO2)']

    temp = pd.DataFrame({year: item.sum() for year, item in buildings.investment_heater.items()})
    detailed['Investment heater (Billion euro)'] = temp.sum() / 10**9
    temp.index = temp.index.map(lambda x: 'Investment {} (Billion euro)'.format(x))
    detailed.update(temp.T / 10 ** 9)
    investment_heater = pd.DataFrame({year: item.sum(axis=1) for year, item in buildings.investment_heater.items()})

    investment_insulation = pd.DataFrame(
        {year: item.sum(axis=1) for year, item in buildings.investment_insulation.items()})
    detailed['Investment insulation (Billion euro)'] = investment_insulation.sum() / 10**9

    index = investment_heater.index.union(investment_insulation.index)
    investment_total = investment_heater.reindex(index, fill_value=0) + investment_insulation.reindex(index,
                                                                                                      fill_value=0)
    detailed['Investment total (Billion euro)'] = investment_total.sum() / 10**9
    temp = investment_total.groupby('Income owner').sum()
    temp.index = temp.index.map(lambda x: 'Investment total {} (Billion euro)'.format(x))
    detailed.update(temp.T / 10**9)
    temp = investment_total.groupby(['Housing type', 'Occupancy status']).sum()
    temp.index = temp.index.map(lambda x: 'Investment total {} - {} (Billion euro)'.format(x[0], x[1]))
    detailed.update(temp.T / 10**9)

    subsidies_heater = pd.DataFrame({year: item.sum(axis=1) for year, item in buildings.subsidies_heater.items()})
    detailed['Subsidies heater (Billion euro)'] = subsidies_heater.sum() / 10**9

    subsidies_insulation = pd.DataFrame(
        {year: item.sum(axis=1) for year, item in buildings.subsidies_insulation.items()})
    detailed['Subsidies insulation (Billion euro)'] = subsidies_insulation.sum() / 10**9

    index = subsidies_heater.index.union(subsidies_insulation.index)
    subsidies_total = subsidies_heater.reindex(index, fill_value=0) + subsidies_insulation.reindex(index, fill_value=0)
    detailed['Subsidies total (Billion euro)'] = subsidies_total.sum() / 10**9
    temp = subsidies_total.groupby('Income owner').sum()
    temp.index = temp.index.map(lambda x: 'Subsidies total {} (Billion euro)'.format(x))
    detailed.update(temp.T / 10**9)
    temp = subsidies_total.groupby(['Housing type', 'Occupancy status']).sum()
    temp.index = temp.index.map(lambda x: 'Subsidies total {} - {} (Billion euro)'.format(x[0], x[1]))
    detailed.update(temp.T / 10**9)

    subsidies = None
    for gest, subsidies_details in {'heater': buildings.subsidies_details_heater,
                                    'insulation': buildings.subsidies_details_insulation}.items():

        subsidies_details = reverse_dict(subsidies_details)
        subsidies_details = pd.DataFrame(
            {key: pd.Series({year: data.sum().sum() for year, data in item.items()}) for key, item in
             subsidies_details.items()}).T
        for i in subsidies_details.index:
            detailed['{} {} (Billion euro)'.format(i.capitalize().replace('_', ' '), gest)] = subsidies_details.loc[i,
                                                                                              :] / 10 ** 9
        if subsidies is None:
            subsidies = subsidies_details.copy()
        else:
            subsidies = pd.concat((subsidies, subsidies_details), axis=0)
    subsidies = subsidies.groupby(subsidies.index).sum()

    taxes_expenditures = buildings.taxes_expenditure_details
    taxes_expenditures = pd.DataFrame(
        {key: pd.Series({year: data.sum() for year, data in item.items()}, dtype=float) for key, item in
         taxes_expenditures.items()}).T
    taxes_expenditures.index = taxes_expenditures.index.map(
        lambda x: '{} (Billion euro)'.format(x.capitalize().replace('_', ' ')))
    detailed.update((taxes_expenditures / 10 ** 9).T)
    detailed['Taxes expenditure (Billion euro)'] = taxes_expenditures.sum() / 10 ** 9

    energy_expenditures = pd.DataFrame(buildings.energy_expenditure_yrs)
    detailed['Energy expenditures (Billion euro)'] = energy_expenditures.sum() / 10 ** 9
    temp = energy_expenditures.groupby('Income tenant').sum()
    temp.index = temp.index.map(lambda x: 'Energy expenditures {} (Billion euro)'.format(x))
    detailed.update(temp.T / 10 ** 9)

    detailed['VTA heater (Billion euro)'] = pd.DataFrame(
        {year: item.sum() for year, item in buildings.tax_heater.items()}).sum() / 10 ** 9

    detailed['VTA insulation (Billion euro)'] = pd.Series(
        {year: item.sum().sum() for year, item in buildings.taxed_insulation.items()}) / 10 ** 9
    detailed['VTA (Billion euro)'] = detailed['VTA heater (Billion euro)'] + detailed['VTA insulation (Billion euro)']

    detailed['Investment cost (Billion euro)'] = detailed['Investment insulation (Billion euro)'] + detailed[
        'Investment heater (Billion euro)']

    detailed['Carbon value (Billion euro)'] = (pd.DataFrame(buildings.heat_consumption_energy_yrs).T * param[
        'carbon_value_kwh']).sum(axis=1) / 10 ** 9

    health_cost = {'health_expenditure': 'Health expenditure (Billion euro)',
                   'mortality_cost': 'Social cost of mortality (Billion euro)',
                   'loss_well_being': 'Loss of well-being (Billion euro)'}
    for key, item in health_cost.items():
        detailed[item] = (stock.T * reindex_mi(param[key], stock.index)).T.sum() / 10 ** 9
    detailed['Health cost (Billion euro)'] = detailed['Health expenditure (Billion euro)'] + detailed[
        'Social cost of mortality (Billion euro)'] + detailed['Loss of well-being (Billion euro)']

    detailed['Income state (Billion euro)'] = detailed['VTA (Billion euro)'] + detailed[
        'Taxes expenditure (Billion euro)']
    detailed['Expenditure state (Billion euro)'] = detailed['Subsidies heater (Billion euro)'] + detailed[
        'Subsidies insulation (Billion euro)']
    detailed['Balance state (Billion euro)'] = detailed['Income state (Billion euro)'] - detailed[
        'Expenditure state (Billion euro)']

    levels = ['Occupancy status', 'Income owner', 'Housing type']
    for level in levels:
        temp = subsidies_total.groupby(level).sum() / investment_total.groupby(level).sum()
        temp.index = temp.index.map(lambda x: 'Share subsidies {} (%)'.format(x))
        detailed.update(temp.T)

    detailed = pd.DataFrame(detailed).loc[buildings.stock_yrs.keys(), :].T

    # graph subsidies
    subset = pd.concat((subsidies, -taxes_expenditures), axis=0).T
    subset = subset.loc[:, (subset != 0).any(axis=0)]

    if 'over_cap' in subset.columns:
        subset['over_cap'] = -subset['over_cap']

    subset.columns = [c.split(' (Billion euro)')[0].capitalize().replace('_', ' ') for c in subset.columns]
    subset.dropna(inplace=True, how='all')
    if not subset.empty:
        make_area_plot(subset / 10**9, 'Billion euro', save=os.path.join(buildings.path, 'policies.png'),
                       colors=generic_input['colors'])

    # graph public finance
    subset = detailed.loc[['VTA (Billion euro)', 'Taxes expenditure (Billion euro)', 'Subsidies heater (Billion euro)',
                           'Subsidies insulation (Billion euro)'], :].T
    subset['Subsidies heater (Billion euro)'] = -subset['Subsidies heater (Billion euro)']
    subset['Subsidies insulation (Billion euro)'] = -subset['Subsidies insulation (Billion euro)']
    subset.dropna(how='any', inplace=True)
    subset.columns = [c.split(' (Billion euro)')[0] for c in subset.columns]
    if not subset.empty:
        make_area_plot(subset, 'Billion euro', save=os.path.join(buildings.path, 'public_finance.png'),
                       colors=generic_input['colors'])

    # graph consumption
    temp = consumption.groupby('Existing').sum().rename(index={True: 'Existing', False: 'Construction'}).T
    temp = temp.loc[:, ['Existing', 'Construction']]
    make_area_plot(temp / 10**9, 'Consumption (TWh)', colors=generic_input['colors'],
                   save=os.path.join(buildings.path, 'consumption.png'))

    df = stock.groupby('Performance').sum().T.sort_index(axis=1, ascending=False)
    make_stackedbar_plot(df, 'Dwelling stock (Millions)', colors=generic_input['colors'],
                         format_y=lambda y, _: y / 10 ** 6,
                         save=os.path.join(buildings.path, 'stock_performance.png'))

    consumption = pd.concat((consumption, certificate, energy), axis=1).set_index(['Performance', 'Energy'],
                                                                                  append=True)

    df = consumption.groupby('Performance').sum().T.sort_index(axis=1, ascending=False)
    make_stackedbar_plot(df, 'Energy consumption (TWh)', colors=generic_input['colors'],
                         format_y=lambda y, _: y / 10 ** 9,
                         save=os.path.join(buildings.path, 'consumption_performance.png'))

    df = consumption.groupby('Energy').sum().T.loc[:, generic_input['index']['Heating energy']]
    make_stackedbar_plot(df, 'Energy consumption (TWh)', colors=generic_input['colors'],
                         format_y=lambda y, _: y / 10 ** 9,
                         save=os.path.join(buildings.path, 'consumption_energy.png'))

    df = consumption.groupby('Income tenant').sum().T.loc[:, generic_input['index']['Income tenant']]
    make_stackedbar_plot(df, 'Energy consumption (TWh)', colors=generic_input['colors'],
                         format_y=lambda y, _: y / 10 ** 9,
                         save=os.path.join(buildings.path, 'consumption_income.png'))

    return stock, detailed


def indicator_policies(result, folder, config, discount_rate=0.045, years=30):
    # TODO: energy taxes
    # TODO: vérifier le calcul sur un exemple simple (spreadsheet)
    folder_policies = os.path.join(folder, 'policies')
    os.mkdir(folder_policies)

    # list_years = [int(re.search('20[0-9][0-9]', key)[0]) for key in config.keys() if re.search('20[0-9][0-9]', key)]
    # temp = ['AP-{}'.format(year) for year in list_years]
    # for key, item in config.items():
    #    if key in ['All policies', 'All policies - 1', 'Zero policies', 'Zero policies + 1'] or key in temp:
    #        config[key] = item.replace(' ', '_')

    if 'Discount rate' in config.keys():
        discount_rate = float(config['Discount rate'])

    if 'Lifetime' in config.keys():
        lifetime = int(config['Lifetime'])

    discount_factor = (1 - (1 + discount_rate) ** -years) / discount_rate

    def double_difference(ref, scenario, values=None, discount_rate=0.045, years=30):
        """Calculate double difference.

        Double difference is a proxy of marginal flow produced in year.
        Flow have effect during a long period of time and need to be extended during the all period.
        Flow are discounted to year.

        Parameters
        ----------
        ref: pd.Series
        scenario: pd.Series or int
        values: pd.Series or None, default None
        discount_rate: float, default 0.045
            Discount rate.
        years: int
            Number of years to extend variables.

        Returns
        -------
        pd.Series
        """
        simple_diff = scenario - ref
        double_diff = simple_diff.diff()

        double_diff.iloc[0] = simple_diff.iloc[0]
        double_diff.rename(None, inplace=True)

        extend = max(double_diff.index) + years - 1

        discount = pd.Series([1 / (1 + discount_rate) ** i for i in range(extend + 1 - double_diff.index[0])],
                             index=range(double_diff.index[0], extend + 1))

        matrix_double_diff = double_diff.to_frame().dot(discount.to_frame().T)
        if values is not None:
            values = values.reindex(matrix_double_diff.columns, method='pad')
            matrix_double_diff = matrix_double_diff * values

        matrix_bool = pd.DataFrame(1, index=matrix_double_diff.index, columns=matrix_double_diff.columns)
        matrix_bool = pd.DataFrame(np.triu(matrix_bool, k=0)) * pd.DataFrame(np.tril(matrix_bool, k=years - 1))
        matrix_bool = matrix_bool.set_axis(matrix_double_diff.index).set_axis(matrix_double_diff.columns, axis=1)

        matrix_double_diff = (matrix_double_diff * matrix_bool).sum()

        discount = pd.Series([1 / (1 + discount_rate) ** i for i in range(matrix_double_diff.shape[0])],
                             index=matrix_double_diff.index)
        return (matrix_double_diff * discount).sum()

    energy_prices = pd.read_csv('project/input/energy_prices.csv', index_col=[0]) * 10 ** 9  # €/kWh to €/TWh
    carbon_value = pd.read_csv('project/input/policies/carbon_value.csv', index_col=[0]).squeeze()  # €/tCO2
    carbon_emission = pd.read_csv('project/input/policies/carbon_emission.csv',
                                  index_col=[0]) * 10 ** 3  # unit: gCO2/ kWh to tCO2/ TWh
    # €/tCO2 * tCO2/TWh  = €/TWh
    carbon_value = (carbon_value * carbon_emission.T).T  # €/TWh
    carbon_value.dropna(how='all', inplace=True)

    scenarios = [s for s in result.keys() if s != 'Reference']
    variables = ['Consumption (TWh)', 'Emission (MtCO2)', 'Health cost (Billion euro)',
                 'Energy expenditures (Billion euro)', 'Carbon value (Billion euro)', 'Balance state (Billion euro)']

    # Double difference = Scenario - Reference
    agg = {}
    for s in scenarios:
        rslt = {}
        for var in ['Consumption standard (TWh)', 'Consumption (TWh)', 'Health cost (Billion euro)',
                    'Health expenditure (Billion euro)', 'Emission (MtCO2)']:
            rslt[var] = double_difference(result['Reference'].loc[var, :], result[s].loc[var, :],
                                          values=None)

        for energy in generic_input['index']['Heating energy']:
            var = 'Consumption {} (TWh)'.format(energy)
            rslt[var] = double_difference(result['Reference'].loc[var, :], result[s].loc[var, :],
                                          values=None)
            rslt['Energy expenditures {} (Billion euro)'.format(energy)] = double_difference(
                result['Reference'].loc[var, :],
                result[s].loc[var, :],
                values=energy_prices[energy]) / (10 ** 9)
            # On a des euros

            rslt['Emission {} (tCO2)'.format(energy)] = double_difference(result['Reference'].loc[var, :],
                                                                          result[s].loc[var, :],
                                                                          values=carbon_emission[energy])

            rslt['Carbon value {} (Billion euro)'.format(energy)] = double_difference(result['Reference'].loc[var, :],
                                                                                      result[s].loc[var, :],
                                                                                      values=carbon_value[energy]) / (
                                                                            10 ** 9)

        # Simple Diff for subsidies, TVA, Investment, then discounted once
        # Simple diff scenario - ref
        for var in ['Subsidies total (Billion euro)', 'VTA (Billion euro)', 'Investment cost (Billion euro)']:
            discount = pd.Series(
                [1 / (1 + 0.045) ** i for i in range(result['Reference'].loc[var, :].shape[0])],
                index=result['Reference'].loc[var, :].index)
            rslt[var] = ((result[s].loc[var, :] - result['Reference'].loc[var, :]) * discount.T).sum()

        rslt['Retrofit ratio'] = (result[s].loc['Retrofit (Thousand)', :]).sum() / (result['Reference'].loc['Retrofit '
                                                                                                            '(Thousand)', :]).sum()
        agg[s] = rslt

    agg = pd.DataFrame(agg)

    def socioeconomic_npv(data, save=None):
        """Calculate socioeconomic NPV.
        Double difference is calculated with : scenario - reference
        If the scenario requires more investment than the reference, then the difference of investments is
        positive, and it is taken into account in the NPV as a negative impact: - Investment cost
        If the scenario results in less energy consumed then the reference, then energy savings is positive,
        and taken into account in the NPV as a positive account.

        Parameters
        ----------
        data: pd.DataFrame
        save: str, default None

        Returns
        -------
        pd.DataFrame
        """
        npv = {}
        for s in data.columns:
            df = data.loc[:, s]

            cofp = (df['Subsidies total (Billion euro)'] - df['VTA (Billion euro)']) * 0.2
            # df['Health Expenditures']

            energy_saved = - sum(df['Energy expenditures {} (Billion euro)'.format(i)]
                                 for i in generic_input['index']['Heating energy'])
            carbon_avoided = - sum(df['Carbon value {} (Billion euro)'.format(i)]
                                   for i in generic_input['index']['Heating energy'])

            temp = pd.Series({'Investment': - df['Investment cost (Billion euro)'],
                              'Energy saving': energy_saved,
                              'Emission saving': carbon_avoided,
                              'Health benefit': - df['Health cost (Billion euro)']
                              })
            if save:
                waterfall_chart(temp, title=s,
                                save=os.path.join(save, 'npv_{}.png'.format(s.lower().replace(' ', '_'))))

            npv[s] = temp

        npv = pd.DataFrame(npv)
        if save:
            assessment_scenarios(npv.T, save=os.path.join(save, 'npv.png'.lower().replace(' ', '_')))

        npv.loc['NPV', :] = npv.sum()
        return npv

    se_npv = socioeconomic_npv(agg, save=folder_policies)
    agg = pd.concat((agg, se_npv), axis=0)

    # Cost-effectiveness in real consumption
    analysis_scenarios = list(set(agg.columns).intersection(['AP-2021', 'AP-2025', 'AP-2030', 'AP-2035', 'AP-2040']))
    reduced_agg = agg.loc[:, analysis_scenarios]
    if 'tax' in config['Policy name']:  # pas sur de la resilience de cette ligne de code
        cost_eff_carbon = pd.DataFrame(reduced_agg.loc['VTA (Billion euro)'] / (reduced_agg.loc['Emission (MtCO2)'])
                                       * 10**3)
        cost_eff_std = pd.DataFrame(reduced_agg.loc['VTA (Billion euro)'] / reduced_agg.loc['Consumption '
                                                                                            'standard (TWh)'])
        cost_eff_real = pd.DataFrame(reduced_agg.loc['VTA (Billion euro)'] / reduced_agg.loc['Consumption (TWh)'])
        leverage_eff = pd.DataFrame(reduced_agg.loc['Investment cost (Billion euro)'] / reduced_agg.loc['VTA (Billion '
                                                                                                        'euro)'])
    else:
        cost_eff_std = pd.DataFrame(
            reduced_agg.loc['Subsidies total (Billion euro)'] / reduced_agg.loc['Consumption standard (TWh)'])
        cost_eff_real = pd.DataFrame(
            reduced_agg.loc['Subsidies total (Billion euro)'] / reduced_agg.loc['Consumption (TWh)'])
        leverage_eff = pd.DataFrame(reduced_agg.loc['Investment cost (Billion euro)'] / reduced_agg.loc['Subsidies '
                                                                                                        'total ('
                                                                                                        'Billion '
                                                                                                        'euro)'])
    policy_name = config['Policy name'].replace('_', ' ')
    cost_eff_carbon.rename(columns={0: "Cost effectiveness carbon {} (€/tCO2)".format(policy_name)}, inplace=True)
    cost_eff_std.rename(columns={0: "Cost effectiveness standard {} (€/kWh)".format(policy_name)}, inplace=True)
    cost_eff_real.rename(columns={0: "Cost effectiveness real {} (€/kWh)".format(policy_name)}, inplace=True)
    leverage_eff.rename(columns={0: "Leverage {} (€/kWh)".format(policy_name)}, inplace=True)

    agg = pd.concat((agg, cost_eff_real.T, cost_eff_std.T, cost_eff_carbon.T, leverage_eff.T), axis=0)

    return agg

    # variables = ['Consumption (TWh)', 'Emission (MtCO2)', 'Energy poverty (Million)', 'Stock low-efficient (Million)',
    #              'Stock efficient (Million)', 'Stock (Million)', 'New efficient (Thousand)',
    #              'Health cost (Billion euro)', 'Retrofit rate 1 EPC (%)']
    # years = [2020]
    # for year in years:
    #     temp = pd.DataFrame(
    #         {'{} - {}'.format(var, year): pd.Series({s: result[s].loc[var, year] for s in result.keys()}) for var in
    #          variables}).T
    #     agg = pd.concat((agg, temp), axis=0)
    # agg.round(2).to_csv(os.path.join(folder, 'comparison.csv'))


def grouped_output(result, stocks, folder, config_runs):
    """Grouped scenarios output.

    Renovation expenditure discounted (Billion euro)
    Subsidies expenditure discounted (Billion euro)
    Cumulated consumption discounted (TWh)
    Cumulated emission discounted (MtCO2)
    Health expenditure discounted (Billion euro)
    Carbon social expenditure discounted (Billion euro)
    Energy expenditure discounted (Billion euro)
    Cumulated emission (MtCO2) - 2030
    Energy poverty (Thousand) - 2030
    Stock low-efficient (Thousand) - 2030
    Consumption conventional (TWh) - 2030
    Consumption actual (TWh) - 2030
    Emission (MtCO2) - 2030
    Cumulated emission (MtCO2) - 2050
    Energy poverty (Thousand) - 2050
    Stock low-efficient (Thousand) - 2050
    Stock efficient (Thousand) - 2050
    Consumption conventional (TWh) - 2050
    Consumption actual (TWh) - 2050
    Emission (MtCO2) - 2050

    Parameters
    ----------
    result
    stocks
    folder
    config_runs: dict

    Returns
    -------

    """

    variables = {'Consumption (TWh)': ('consumption_hist.png', lambda y, _: '{:,.0f}'.format(y), generic_input['consumption_total_hist'], generic_input['consumption_total_objectives']),
                 'Heating intensity (%)': ('heating_intensity.png', lambda y, _: '{:,.1%}'.format(y)),
                 'Emission (MtCO2)': ('emission.png', lambda y, _: '{:,.0f}'.format(y)),
                 'Energy poverty (Million)': ('energy_poverty.png', lambda y, _: '{:,.1f}'.format(y)),
                 'Stock low-efficient (Million)': ('stock_low_efficient.png', lambda y, _: '{:,.1f}'.format(y)),
                 'Stock efficient (Million)': ('stock_efficient.png', lambda y, _: '{:,.1f}'.format(y)),
                 'New efficient (Thousand)': ('flow_efficient.png', lambda y, _: '{:,.0f}'.format(y)),
                 'Retrofit >= 1 EPC (Thousand)': ('flow_retrofit.png', lambda y, _: '{:,.0f}'.format(y)),
                 'Investment total (Billion euro)': ('flow_investment.png', lambda y, _: '{:,.0f}'.format(y)),
                 'Subsidies total (Billion euro)': ('flow_subsidies.png', lambda y, _: '{:,.0f}'.format(y)),
                 'Energy expenditures (Billion euro)': (
                 'flow_energy_expenditures.png', lambda y, _: '{:,.0f}'.format(y)),
                 'Health cost (Billion euro)': ('flow_health_cost.png', lambda y, _: '{:,.0f}'.format(y)),
                 'Insulation actions (Thousand)': ('flow_insulation.png', lambda y, _: '{:,.0f}'.format(y)),
                 'Replacement heater (Thousand)': ('flow_heater.png', lambda y, _: '{:,.0f}'.format(y))
                 }

    for variable, infos in variables.items():
        temp = pd.DataFrame({scenario: output.loc[variable, :] for scenario, output in result.items()})
        try:
            temp = pd.concat((temp, infos[2]), axis=1)
            temp.sort_index(inplace=True)
        except IndexError:
            pass

        try:
            scatter = infos[3]
        except IndexError:
            scatter = None

        make_plot(temp, variable, save=os.path.join(folder, '{}'.format(infos[0])), format_y=infos[1], scatter=scatter)

    def grouped(result, variables):
        """Group result.

        Parameters
        ----------
        result: dict
        variables: list
        """
        temp = {var: pd.DataFrame(
            {scenario: output.loc[var, :] for scenario, output in result.items() if var in output.index}) for var in
            variables}
        return {k: i for k, i in temp.items() if not i.empty}

    variables_detailed = {
        'Consumption {} (TWh)': [
            ('Heating energy', lambda y, _: '{:,.0f}'.format(y), 2, generic_input['consumption_hist'])],
        'Stock {} (Million)': [('Performance', lambda y, _: '{:,.0f}'.format(y))],
        'Heating intensity {} (%)': [('Income tenant', lambda y, _: '{:,.0%}'.format(y))],
        'Subsidies total {} (Billion euro)': [('Income owner', lambda y, _: '{:,.0f}'.format(y)),
                                              ('Decision maker', lambda y, _: '{:,.0f}'.format(y), 2)
                                              ],
        'Investment {} (Billion euro)': [('Insulation', lambda y, _: '{:,.0f}'.format(y), 2)],
        'Investment total {} (Billion euro)': [
            ('Decision maker', lambda y, _: '{:,.0f}'.format(y), 2)],
        'Insulation {} (Thousand)': [
            ('Insulation', lambda y, _: '{:,.0f}'.format(y), 2, None, generic_input['retrofit_hist'])],
        'Insulation actions {} (Thousand)': [
            ('Decision maker', lambda y, _: '{:,.0f}'.format(y), 2)],
        'Insulation actions {} (%)': [
            ('Decision maker', lambda y, _: '{:,.0%}'.format(y), 2)]
    }

    def details_graphs(data, v, inf):
        n = (v.split(' {}')[0] + '_' + inf[0] + '.png').replace(' ', '_').lower()
        temp = grouped(data, [v.format(i) for i in generic_input['index'][inf[0]]])
        replace = {v.format(i): i for i in generic_input['index'][inf[0]]}
        temp = {replace[key]: item for key, item in temp.items()}

        try:
            n_columns = inf[2]
        except IndexError:
            n_columns = len(temp.keys())

        try:
            if inf[3] is not None:
                for key in temp.keys():
                    temp[key] = pd.concat((temp[key], inf[3][key]), axis=1)
                    temp[key].sort_index(inplace=True)
        except IndexError:
            pass

        try:
            scatter = inf[4]
        except IndexError:
            scatter = None

        make_grouped_subplots(temp, format_y=inf[1], n_columns=n_columns,
                              save=os.path.join(folder, n), scatter=scatter)

    for var, infos in variables_detailed.items():
        for info in infos:
            details_graphs(result, var, info)

    if 'Reference' in result.keys() and len(result.keys()) > 1 and config_runs is not None:
        indicator_policies(result, folder, config_runs)
