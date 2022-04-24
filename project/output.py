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
import matplotlib.pyplot as plt
import os

from param import generic_input
from utils import reverse_dict, make_plot, reindex_mi, make_grouped_subplots

SMALL_SIZE = 10
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def write_output(buildings, param):
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
                                                        buildings.heat_consumption_sd_yrs.keys()}) / 10**9

    consumption = pd.DataFrame(buildings.heat_consumption_energy_yrs)
    detailed['Consumption (TWh)'] = consumption.sum() / 10**9
    for i in consumption.index:
        detailed['Consumption - {} (TWh)'.format(i)] = consumption.loc[i, :] / 10**9

    emission = (consumption * param['carbon_emission'].T).dropna(axis=1, how='all')
    detailed['Emission (MtCO2)'] = emission.sum() / 10**12
    detailed['Cumulated emission (MtCO2)'] = detailed['Emission (MtCO2)'].cumsum()
    detailed['Stock (Million)'] = stock.sum() / 10**6

    temp = pd.DataFrame(buildings.certificate_nb)
    temp.index = temp.index.map(lambda x: 'Stock {} (Million)'.format(x))
    detailed.update(temp.T / 10**6)
    try:
        detailed['Stock efficient (Million)'] = detailed['Stock A (Million)'] + detailed['Stock B (Million)']
    except KeyError:
        detailed['Stock efficient (Million)'] = detailed['Stock B (Million)']

    detailed['Stock low-efficient (Million)'] = detailed['Stock F (Million)'] + detailed['Stock G (Million)']
    detailed['Energy poverty (Million)'] = pd.Series(buildings.energy_poverty) / 10**6
    detailed['Heating intensity (%)'] = pd.Series(buildings.heating_intensity_avg)
    temp = pd.DataFrame(buildings.heating_intensity_tenant)
    temp.index = temp.index.map(lambda x: 'Heating intensity {} (%)'.format(x))
    detailed.update(temp.T)

    detailed['New efficient (Thousand)'] = pd.Series(buildings.efficient_renovation_yrs) / 10**3
    for i in range(6):
        temp = pd.DataFrame({year: item.loc[:, i] for year, item in buildings.certificate_jump_yrs.items()})
        detailed['Retrofit {} EPC (Thousand)'.format(i)] = temp.sum() / 10**3
        detailed['Retrofit rate {} EPC (%)'.format(i)] = temp.sum() / stock.sum()

        t = temp.groupby('Income owner').sum() / stock.groupby('Income owner').sum()
        t.index = t.index.map(lambda x: 'Retrofit rate {} EPC {} (%)'.format(i, x))
        detailed.update(t.T)

        t = temp.groupby(['Occupancy status', 'Housing type']).sum() / stock.groupby(
            ['Occupancy status', 'Housing type']).sum()
        t.index = t.index.map(lambda x: 'Retrofit rate {} EPC {} - {} (%)'.format(i, x[0], x[1]))
        detailed.update(t.T)

    # for replacement output need to be presented by technologies (what is used) and by agent (who change)
    replacement_heater = buildings.replacement_heater
    temp = pd.DataFrame({year: item.sum() for year, item in replacement_heater.items()})
    t = temp.copy()
    t.index = t.index.map(lambda x: 'Replacement heater {} (Thousand)'.format(x))
    detailed.update((t / 10**3).T)
    detailed['Replacement heater (Thousands)'] = temp.sum() / 10**3

    t = temp * pd.DataFrame(buildings.cost_heater)
    t.index = t.index.map(lambda x: 'Investment heater {} (Million)'.format(x))
    detailed.update((t / 10**6).T)

    temp = pd.DataFrame({year: item.sum(axis=1) for year, item in replacement_heater.items()})
    t = temp.groupby('Income owner').sum()
    t.index = t.index.map(lambda x: 'Replacement heater {} (Thousand)'.format(x))
    detailed.update((t / 10**3).T)
    t = temp.groupby(['Housing type', 'Occupancy status']).sum()
    t.index = t.index.map(lambda x: 'Replacement heater {} - {} (Thousand)'.format(x[0], x[1]))
    detailed.update((t / 10**3).T)

    replacement_insulation = buildings.replacement_insulation
    temp = pd.DataFrame({year: item.sum(axis=1) for year, item in replacement_insulation.items()})
    detailed['Replacement component (Thousands)'] = temp.sum() / 10**3
    t = temp.groupby('Income owner').sum()
    t.index = t.index.map(lambda x: 'Replacement component {} (Thousand)'.format(x))
    detailed.update(t.T / 10**3)
    t = temp.groupby(['Housing type', 'Occupancy status']).sum()
    t.index = t.index.map(lambda x: 'Replacement component {} - {} (Thousand)'.format(x[0], x[1]))
    detailed.update((t / 10**3).T)

    for i in ['Wall', 'Floor', 'Roof', 'Windows']:
        temp = pd.DataFrame(
            {year: item.xs(True, level=i, axis=1).sum(axis=1) for year, item in replacement_insulation.items()})
        t = (pd.DataFrame(buildings.cost_component).loc[i, :] * temp)
        detailed['Investment {} (Million euro)'.format(i)] = (t * reindex_mi(param['surface'], t.index)).sum() / 10**6
        # only work because existing surface does not change over time
        detailed['Replacement component {} (Thousand)'.format(i)] = temp.sum() / 10**3

    detailed['Energy expenditures (Billion euro)'] = pd.Series(
        {year: item.sum() for year, item in buildings.energy_expenditure.items()}) / 10**9

    temp = pd.DataFrame({year: item.sum() for year, item in buildings.investment_heater.items()})
    detailed['Investment heater (Billion euro)'] = temp.sum() / 10**9
    temp.index = temp.index.map(lambda x: 'Investment {} (Billion euro)'.format(x))
    detailed.update(temp.T / 10**9)

    investment_heater = pd.DataFrame({year: item.sum(axis=1) for year, item in buildings.investment_heater.items()})
    temp = investment_heater.groupby('Income owner').sum()
    temp.index = temp.index.map(lambda x: 'Investment heater {} (Billion euro)'.format(x))
    detailed.update(temp.T / 10**9)

    subsidies_heater = pd.DataFrame({year: item.sum(axis=1) for year, item in buildings.subsidies_heater.items()})
    detailed['Subsidies heater (Billion euro)'] = subsidies_heater.sum() / 10**9
    temp = subsidies_heater.groupby('Income owner').sum()
    temp.index = temp.index.map(lambda x: 'Subsidies heater {} (Billion euro)'.format(x))
    detailed.update(temp.T / 10**9)

    investment_insulation = pd.DataFrame({year: item.sum(axis=1) for year, item in buildings.investment_insulation.items()})
    detailed['Investment insulation (Billion euro)'] = investment_insulation.sum() / 10**9
    temp = investment_insulation.groupby('Income owner').sum()
    temp.index = temp.index.map(lambda x: 'Investment insulation {} (Billion euro)'.format(x))
    detailed.update(temp.T / 10**9)
    temp = investment_insulation.groupby(['Housing type', 'Occupancy status']).sum()
    temp.index = temp.index.map(lambda x: 'Investment insulation {} - {} (Billion euro)'.format(x[0], x[1]))
    detailed.update(temp.T / 10**9)

    subsidies_insulation = pd.DataFrame({year: item.sum(axis=1)for year, item in buildings.subsidies_insulation.items()})
    detailed['Subsidies insulation (Billion euro)'] = subsidies_insulation.sum() / 10**9
    temp = subsidies_insulation.groupby('Income owner').sum()
    temp.index = temp.index.map(lambda x: 'Subsidies insulation {} (Billion euro)'.format(x))
    detailed.update(temp.T / 10**9)
    temp = subsidies_insulation.groupby(['Housing type', 'Occupancy status']).sum()
    temp.index = temp.index.map(lambda x: 'Subsidies insulation {} - {} (Billion euro)'.format(x[0], x[1]))
    detailed.update(temp.T / 10**9)

    for gest, subsidies_details in {'heater': buildings.subsidies_details_heater,
                                    'insulation': buildings.subsidies_details_insulation}.items():

        subsidies_details = reverse_dict(subsidies_details)
        subsidies_details = pd.DataFrame(
            {key: pd.Series({year: data.sum().sum() for year, data in item.items()}) for key, item in
             subsidies_details.items()}).T
        for i in subsidies_details.index:
            detailed['{} {} (Billion euro)'.format(i.capitalize().replace('_', ' '), gest)] = subsidies_details.loc[i, :] / 10**9

    taxes_expenditures = buildings.taxes_expenditure_details
    taxes_expenditures = pd.DataFrame(
        {key: pd.Series({year: data.sum() for year, data in item.items()}) for key, item in
         taxes_expenditures.items()}).T
    taxes_expenditures.index = taxes_expenditures.index.map(
        lambda x: '{} (Billion euro)'.format(x.capitalize().replace('_', ' ')))
    detailed.update((taxes_expenditures / 10**9).T)

    detailed['VTA heater (Billion euro)'] = pd.DataFrame(
        {year: item.sum() for year, item in buildings.tax_heater.items()}).sum() / 10**9

    detailed['VTA insulation (Billion euro)'] = pd.Series(
        {year: item.sum().sum() for year, item in buildings.tax_insulation.items()}) / 10**9

    detailed['Income state (Billion euro)'] = detailed['VTA heater (Billion euro)'] + detailed[
        'VTA insulation (Billion euro)'] + taxes_expenditures.sum() / 10**9
    detailed['Expenditure state (Billion euro)'] = detailed['Subsidies heater (Billion euro)'] + detailed[
                                                    'Subsidies insulation (Billion euro)']
    detailed['Balance state (Billion euro)'] = detailed['Income state (Billion euro)'] - detailed[
        'Expenditure state (Billion euro)']

    detailed['Investment cost (Billion euro)'] = detailed['Investment insulation (Billion euro)'] + detailed[
        'Investment heater (Billion euro)']

    detailed['Energy expenditure (Billion euro)'] = pd.Series({year: item.sum() for year, item in
                                                               buildings.energy_expenditure.items()}) / 10 ** 9
    detailed['Carbon value (Billion euro)'] = (pd.DataFrame(buildings.heat_consumption_energy_yrs).T * param[
        'carbon_value_kwh']).sum(axis=1) / 10 ** 9

    detailed['Health cost (Billion euro)'] = (stock.T * reindex_mi(param['health_cost'], stock.index)).T.sum() / 10**9
    detailed = pd.DataFrame(detailed).loc[buildings.stock_yrs.keys(), :].T
    return stock, detailed


def grouped_output(result, stocks, folder):
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

    Returns
    -------

    """

    def double_difference(ref, scenario, discount=0.045, years=30):
        """Calculate double difference.

        Double difference is a proxy of marginal flow produced in year.
        Flow have effect during a long period of time and need to be extended during the all period.
        Flow are discounted to year.

        Parameters
        ----------
        ref: pd.Series
        scenario: pd.Series
        discount: float
            Discount rate.
        years: int
            Number of years to extend variables.

        Returns
        -------
        pd.Series
        """
        factor = (1 - (1 + discount) ** -years) / discount
        simple_diff = ref - scenario
        double_diff = simple_diff.diff()
        double_diff.iloc[0] = simple_diff.iloc[0]
        double_diff = double_diff * factor
        discount = pd.Series([1 / (1 + discount)**i for i in range(double_diff.shape[0])], index=double_diff.index)
        return (double_diff * discount).sum()

    variables = {'consumption': 'Consumption (TWh)',
                 'emission': 'Emission (MtCO2)',
                 'energy_poverty': 'Energy poverty (Million)',
                 'low_efficient': 'Stock low-efficient (Million)',
                 'efficient': 'Stock efficient (Million)',
                 'heating_intensity': 'Heating intensity (%)'}

    for name, variable in variables.items():
        temp = pd.DataFrame({scenario: output.loc[variable, :] for scenario, output in result.items()})
        make_plot(temp, variable, save=os.path.join(folder, '{}.png'.format(name)))

    scenarios = [s for s in result.keys() if s != 'Reference']
    variables = ['Consumption (TWh)', 'Emission (MtCO2)', 'Health cost (Billion euro)',
                 'Energy expenditure (Billion euro)', 'Carbon value (Billion euro)', 'Balance state (Billion euro)']

    agg = pd.DataFrame({var: pd.Series(
        [double_difference(result['Reference'].loc[var, :], result[s].loc[var, :]) for s in scenarios], index=scenarios)
                        for var in variables})
    temp = pd.Series([double_difference(result['Reference'].loc[var, :], 0) for var in variables], index=variables,
                     name='Reference')
    agg = pd.concat((temp, agg.T), axis=1)

    variables = ['Consumption (TWh)', 'Emission (MtCO2)', 'Energy poverty (Million)', 'Stock low-efficient (Million)',
                 'Stock efficient (Million)', 'Stock (Million)', 'New efficient (Thousand)',
                 'Health cost (Billion euro)', 'Renovation rate 1 (%)']
    years = [2020]
    for year in years:
        temp = pd.DataFrame(
            {'{} - {}'.format(var, year): pd.Series({s: result[s].loc[var, year] for s in result.keys()}) for var in
             variables}).T
        agg = pd.concat((agg, temp), axis=0)
    agg.round(2).to_csv(os.path.join(folder, 'comparison.csv'))

    variables_detailed = {'stock': ['Stock {} (Million)'.format(i) for i in generic_input['index']['Performance']],
                          'heating_intensity': ['Heating intensity {} (%)'.format(i) for i in
                                                generic_input['index']['Income tenant']]}

    def grouped(result, variables):
        """Group result.

        Parameters
        ----------
        result: dict
        variables: list

        Returns
        -------

        """
        temp = {var: pd.DataFrame(
            {scenario: output.loc[var, :] for scenario, output in result.items() if var in output.index}) for var in
                variables}
        return {k: i for k, i in temp.items() if not i.empty}

    temp = grouped(result, ['Heating intensity {} (%)'.format(i) for i in generic_input['index']['Income tenant']])
    make_grouped_subplots(temp, format_y=lambda y, _: '{:,.0f}%'.format(y * 100),
                          save=os.path.join(folder, 'heating_intensity.png'))

    temp = grouped(result, ['Stock {} (Million)'.format(i) for i in generic_input['index']['Performance']])
    make_grouped_subplots(temp, format_y=lambda y, _: '{:,.0f}%'.format(y * 100),
                          save=os.path.join(folder, 'stock.png'))

