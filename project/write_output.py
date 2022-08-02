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
import logging

from input.param import generic_input
from utils import reverse_dict, make_plot, reindex_mi, make_grouped_subplots, make_area_plot, waterfall_chart, \
    assessment_scenarios, format_ax, format_legend, save_fig

logger = logging.getLogger(__name__)


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
                                                        buildings.heat_consumption_sd_yrs.keys()}) / 10**9

    consumption = pd.DataFrame(buildings.heat_consumption_calib_yrs)
    detailed['Consumption (TWh)'] = consumption.sum() / 10**9

    temp = consumption.groupby(buildings.energy).sum()
    temp.index = temp.index.map(lambda x: 'Consumption {} (TWh)'.format(x))
    detailed.update(temp.T / 10**9)

    temp = consumption.groupby(buildings.certificate).sum()
    temp.index = temp.index.map(lambda x: 'Consumption {} (TWh)'.format(x))
    detailed.update(temp.T / 10 ** 9)

    c = buildings.add_energy(consumption)
    emission = reindex_mi(param['carbon_emission'].T.rename_axis('Energy', axis=0), c.index) * c
    emission = emission.reindex(consumption.columns, axis=1)

    detailed['Emission (MtCO2)'] = emission.sum() / 10 ** 12

    temp = emission.groupby('Energy').sum()
    temp.index = temp.index.map(lambda x: 'Emission {} (MtCO2)'.format(x))
    detailed.update(temp.T / 10**9)

    # emission = (consumption.groupby(buildings.energy).sum() * param['carbon_emission'].T).dropna(axis=1, how='all')
    detailed['Cumulated emission (MtCO2)'] = detailed['Emission (MtCO2)'].cumsum()
    detailed['Stock (Million)'] = stock.sum() / 10 ** 6
    #detailed['Sizing factor (%)'] = pd.Series(detailed['Stock (Million)'].shape[0] * [param['sizing_factor']],
    #                                          index=detailed['Stock (Million)'].index)
    detailed['Surface (Million m2)'] = pd.DataFrame(buildings.surface_yrs).sum() / 10**6

    detailed['Surface (m2/person)'] = (detailed['Surface (Million m2)'] / (param['population'] / 10**6)).dropna()
    detailed['Consumption standard (kWh/m2)'] = (detailed['Consumption standard (TWh)'] * 10 ** 9) / (
                detailed['Surface (Million m2)'] * 10 ** 6)
    detailed['Consumption (kWh/m2)'] = (detailed['Consumption (TWh)'] * 10 ** 9) / (
                detailed['Surface (Million m2)'] * 10 ** 6)

    detailed['Heating intensity (%)'] = pd.Series(buildings.heating_intensity_avg)
    temp = pd.DataFrame(buildings.heating_intensity_tenant)
    temp.index = temp.index.map(lambda x: 'Heating intensity {} (%)'.format(x))
    detailed.update(temp.T)

    detailed['Energy poverty (Million)'] = pd.Series(buildings.energy_poverty) / 10 ** 6

    temp = pd.DataFrame(buildings.certificate_nb)
    temp.index = temp.index.map(lambda x: 'Stock {} (Million)'.format(x))
    detailed.update(temp.T / 10 ** 6)
    try:
        detailed['Stock efficient (Million)'] = detailed['Stock A (Million)'] + detailed['Stock B (Million)']
    except KeyError:
        detailed['Stock efficient (Million)'] = detailed['Stock B (Million)']

    detailed['Stock low-efficient (Million)'] = detailed['Stock F (Million)'] + detailed['Stock G (Million)']

    temp = pd.DataFrame(buildings.retrofit_rate).dropna(how='all')
    temp = temp.groupby([i for i in temp.index.names if i not in ['Heating system final']]).mean()
    t = temp.xs(False, level='Heater replacement')

    s_temp = pd.DataFrame(buildings.stock_yrs)
    s_temp = s_temp.groupby([i for i in s_temp.index.names if i != 'Income tenant']).sum()

    fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))
    for year in t.columns:
        df = pd.concat((s_temp[year - 1], t[year]), axis=1, keys=['Stock', 'Retrofit rate']).sort_values('Retrofit rate')
        df['Stock'] = df['Stock'].cumsum() / 10**6
        df = df.set_index('Stock').squeeze().sort_values()
        df.plot(ax=ax)
    format_ax(ax, format_y=lambda x, _: '{:.0%}'.format(x), y_label='Retrofit rate (%)')
    ax.set_xlabel('Cumulated buildings stock (Million)')
    format_legend(ax, labels=t.columns)
    save_fig(fig, save=os.path.join(buildings.path, 'retrofit_rate.png'))

    #Weighted average with stock to calculate real retrofit rate
    detailed['Retrofit rate (%)'] = ((t * s_temp).sum() / s_temp.sum()).drop(2018)
    t_grouped = (t * s_temp).groupby(['Housing type', 'Occupancy status']).sum() / s_temp.groupby(['Housing type',
                                                                                                   'Occupancy status']).sum()
    t_grouped = t_grouped.drop(2018, axis=1)
    t_grouped.index = t_grouped.index.map(lambda x: 'Retrofit rate {} - {} (%)'.format(x[0], x[1]))
    detailed.update(t_grouped.T)

    detailed['Non-weighted retrofit rate (%)'] = t.mean()
    t = t.groupby(['Housing type', 'Occupancy status']).mean()
    t.index = t.index.map(lambda x: 'Non-weighted retrofit rate {} - {} (%)'.format(x[0], x[1]))
    detailed.update(t.T)

    t = temp.xs(True, level='Heater replacement')
    s_temp = pd.DataFrame(buildings.stock_yrs)
    s_temp = s_temp.groupby([i for i in s_temp.index.names if i != 'Income tenant']).sum()
    detailed['Retrofit rate w/ heater (%)'] = ((t * s_temp).sum() / s_temp.sum()).drop(2018)

    t_grouped = (t * s_temp).groupby(['Housing type', 'Occupancy status']).sum() / s_temp.groupby(['Housing type',
                                                                                           'Occupancy status']).sum()
    t_grouped = t_grouped.drop(2018, axis=1)
    t_grouped.index = t_grouped.index.map(lambda x: 'Retrofit rate heater {} - {} (%)'.format(x[0], x[1]))
    detailed.update(t_grouped.T)

    detailed['Non-weighted retrofit rate w/ heater (%)'] = t.mean()
    t = t.groupby(['Housing type', 'Occupancy status']).mean()
    t.index = t.index.map(lambda x: 'Non-weighted retrofit rate heater {} - {} (%)'.format(x[0], x[1]))
    detailed.update(t.T)

    detailed['Retrofit (Thousand)'] = pd.Series(
        {year: item.sum().sum() for year, item in buildings.certificate_jump_yrs.items()}) / 10**3
    #We need them by income for freerider ratios per income deciles
    temp = pd.DataFrame(
        {year: item.sum(axis=1) for year, item in buildings.certificate_jump_yrs.items()})
    t = temp.groupby('Income owner').sum()
    t.index = t.index.map(lambda x: 'Retrofit {} (Thousand)'.format(x))
    detailed.update(t.T / 10**3)
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

    detailed['Efficient retrofits (Thousand)'] = pd.Series(buildings.efficient_renovation_yrs) / 10**3
    detailed['Global retrofits (Thousand)'] = pd.Series(buildings.global_renovation_yrs).T / 10**3
    detailed['Bonus best retrofits (Thousand)'] = pd.Series(buildings.bonus_best_yrs).T / 10**3
    detailed['Bonus worst retrofits (Thousand)'] = pd.Series(buildings.bonus_worst_yrs).T / 10**3
    detailed['Percentage of global retrofits'] = detailed['Global retrofits (Thousand)']/detailed[
        'Retrofit (Thousand)']
    detailed['Percentage of bonus best retrofits'] = detailed['Bonus best retrofits (Thousand)']/detailed[
        'Retrofit (Thousand)']
    detailed['Percentage of bonus worst retrofits'] = detailed['Bonus worst retrofits (Thousand)']/detailed[
        'Retrofit (Thousand)']

    # for replacement output need to be presented by technologies (what is used) and by agent (who change)
    replacement_heater = buildings.replacement_heater
    temp = pd.DataFrame({year: item.sum() for year, item in replacement_heater.items()})
    t = temp.copy()
    t.index = t.index.map(lambda x: 'Replacement heater {} (Thousand)'.format(x))
    detailed.update((t / 10 ** 3).T)
    detailed['Replacement heater (Thousand)'] = temp.sum() / 10 ** 3

    """t = temp * pd.DataFrame(buildings.cost_heater)
    t.index = t.index.map(lambda x: 'Investment heater {} (Million)'.format(x))
    detailed.update((t / 10**6).T)

    temp = pd.DataFrame({year: item.sum(axis=1) for year, item in replacement_heater.items()})
    t = temp.groupby('Income owner').sum()
    t.index = t.index.map(lambda x: 'Replacement heater {} (Thousand)'.format(x))
    detailed.update((t / 10 ** 3).T)
    t = temp.groupby(['Housing type', 'Occupancy status']).sum()
    t.index = t.index.map(lambda x: 'Replacement heater {} - {} (Thousand)'.format(x[0], x[1]))
    detailed.update((t / 10 ** 3).T)
    """
    replacement_insulation = buildings.replacement_insulation
    temp = pd.DataFrame({year: item.sum(axis=1) for year, item in replacement_insulation.items()})
    detailed['Replacement insulation (Thousand)'] = temp.sum() / 10 ** 3
    t = temp.groupby('Income owner').sum()
    t.index = t.index.map(lambda x: 'Replacement insulation {} (Thousand)'.format(x))
    detailed.update(t.T / 10**3)
    t = temp.groupby(['Housing type', 'Occupancy status']).sum()
    t.index = t.index.map(lambda x: 'Replacement insulation {} - {} (Thousand)'.format(x[0], x[1]))
    detailed.update((t / 10**3).T)
    t.index = t.index.str.replace('Thousand', '%')
    s = stock.groupby(['Housing type', 'Occupancy status']).sum()
    s.index = s.index.map(lambda x: 'Replacement insulation {} - {} (%)'.format(x[0], x[1]))
    t = t / s
    detailed.update(t.T)

    for i in ['Wall', 'Floor', 'Roof', 'Windows']:
        temp = pd.DataFrame(
            {year: item.xs(True, level=i, axis=1).sum(axis=1) for year, item in replacement_insulation.items()})
        detailed['Replacement {} (Thousand)'.format(i)] = temp.sum() / 10**3

        cost = pd.DataFrame({key: item.loc[:, i] for key, item in buildings.cost_component.items()})
        t = reindex_mi(cost, temp.index) * temp
        # only work because existing surface does not change over time
        detailed['Investment {} (Billion euro)'.format(i)] = (t * reindex_mi(param['surface'], t.index)).sum().loc[
                                                                 t.columns] / 10 ** 9

        detailed['Embodied energy {} (TWh PE)'.format(i)] = (temp * reindex_mi(param['surface'], temp.index) *
                                                             param['embodied_energy_renovation'][i]).sum().loc[
                                                                temp.columns] / 10 ** 9
        detailed['Carbon footprint {} (MtCO2)'.format(i)] = (temp * reindex_mi(param['surface'], temp.index) *
                                                             param['carbon_footprint_renovation'][i]).sum().loc[
                                                                temp.columns] / 10 ** 9

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

    detailed['Cost factor insulation (%)'] = pd.Series(buildings.factor_yrs, dtype=float)

    temp = pd.DataFrame({year: item.sum() for year, item in buildings.investment_heater.items()})
    detailed['Investment heater (Billion euro)'] = temp.sum() / 10**9
    temp.index = temp.index.map(lambda x: 'Investment {} (Billion euro)'.format(x))
    detailed.update(temp.T / 10 ** 9)
    investment_heater = pd.DataFrame({year: item.sum(axis=1) for year, item in buildings.investment_heater.items()})

    #representative insulation investment: weighted average with number of insulation actions as weights
    investment_insulation_repr = pd.DataFrame(buildings.investment_insulation_repr)
    gest = pd.DataFrame({year: item.sum(axis=1) for year, item in replacement_insulation.items()})
    gest = reindex_mi(gest, investment_insulation_repr.index)
    temp = gest * investment_insulation_repr

    t = temp.groupby('Income owner').sum() / gest.groupby('Income owner').sum()
    t.index = t.index.map(lambda x: 'Investment per insulation action {} (euro)'.format(x))
    detailed.update(t.T)

    t = temp.groupby(['Housing type', 'Occupancy status']).sum() / gest.groupby(['Housing type',
                                                                                 'Occupancy status']).sum()
    t.index = t.index.map(lambda x: 'Investment per insulation action {} - {} (euro)'.format(x[0], x[1]))
    detailed.update(t.T)

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
    for i in subsidies.index:
        detailed['{} (Billion euro)'.format(i.capitalize().replace('_', ' '))] = subsidies.loc[i, :] / 10 ** 9


    taxes_expenditures = buildings.taxes_expenditure_details
    taxes_expenditures = pd.DataFrame(
        {key: pd.Series({year: data.sum() for year, data in item.items()}, dtype=float) for key, item in
         taxes_expenditures.items()}).T
    taxes_expenditures.index = taxes_expenditures.index.map(
        lambda x: '{} (Billion euro)'.format(x.capitalize().replace('_', ' ').replace('Cee', 'Cee tax')))
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

    """index = buildings.utility_insulation_extensive.index
    utility_dict = {i: pd.DataFrame({year: item.loc[i, :] for year, item in buildings.utility_yrs.items()}).T for i in index}
    index = [('Single-family', 'Owner-occupied', False), ('Multi-family', 'Owner-occupied', False)]
    path_utility = os.path.join(buildings.path, 'utility')
    os.mkdir(path_utility)
    for i in index:
        utility_dict[i].to_csv(os.path.join(path_utility, 'utility_{}_{}.csv'.format(i[0].lower(), i[1].lower())))
        make_area_plot(utility_dict[i], '{} - {}'.format(i[0], i[1]),
                       save=os.path.join(path_utility, 'utility_{}_{}.png'.format(i[0].lower(), i[1].lower())))

    pd.DataFrame(buildings.market_share_yrs).to_csv(os.path.join(path_utility, 'market_share.csv'))"""

    """
    b = pd.DataFrame(buildings.bill_saved_repr)
    s = pd.DataFrame(buildings.subsidies_insulation_repr)
    i = pd.DataFrame(buildings.investment_insulation_repr)
    r = pd.DataFrame(buildings.retrofit_rate)
    """

    df = pd.DataFrame([detailed.loc['Replacement {} (Thousand)'.format(i), :] for i in generic_input['index']['Insulation']]).T.dropna()
    df.columns = generic_input['index']['Insulation']
    make_area_plot(df, 'Replacement (Thousand)',
                   save=os.path.join(buildings.path, 'replacement_insulation.png'), total=False,
                   format_y=lambda y, _: '{:.0f}'.format(y), colors=generic_input['colors'], loc='left', left=1.1)

    df = pd.DataFrame([detailed.loc['Replacement heater {} (Thousand)'.format(i), :] for i in generic_input['index']['Heater']]).T.dropna()
    df.columns = generic_input['index']['Heater']
    make_area_plot(df, 'Replacement (Thousand)',
                   save=os.path.join(buildings.path, 'replacement_heater.png'), total=False,
                   format_y=lambda y, _: '{:.0f}'.format(y),
                   colors=generic_input['colors'], loc='left', left=1.25)

    subset = pd.concat((subsidies, -taxes_expenditures), axis=0).T
    subset = subset.loc[:, (subset != 0).any(axis=0)]
    if 'over_cap' in subset.columns:
        subset.drop('over_cap', inplace=True, axis=1)
    subset.columns = [c.split(' (Billion euro)')[0].capitalize().replace('_', ' ') for c in subset.columns]
    subset.dropna(inplace=True, how='all')
    if not subset.empty:
        make_area_plot(subset, 'Billion euro', save=os.path.join(buildings.path, 'policies.png'),
                       colors=generic_input['colors'], format_y=lambda y, _: '{:.0f}'.format(y / 10 ** 9),
                       loc='left', left=1.1)

    # graph public finance
    subset = detailed.loc[['VTA (Billion euro)', 'Taxes expenditure (Billion euro)', 'Subsidies heater (Billion euro)',
                           'Subsidies insulation (Billion euro)'], :].T
    subset['Subsidies heater (Billion euro)'] = -subset['Subsidies heater (Billion euro)']
    subset['Subsidies insulation (Billion euro)'] = -subset['Subsidies insulation (Billion euro)']
    subset.dropna(how='any', inplace=True)
    subset.columns = [c.split(' (Billion euro)')[0] for c in subset.columns]
    if not subset.empty:
        make_area_plot(subset, 'Billion euro', save=os.path.join(buildings.path, 'public_finance.png'),
                       colors=generic_input['colors'],
                       format_y=lambda y, _: '{:.0f}'.format(y), loc='left', left=1.1)

    df = investment_total.groupby('Income owner').sum().loc[generic_input['index']['Income owner']].T
    make_area_plot(df, 'Investment (Billion euro)', colors=generic_input['colors'],
                   save=os.path.join(buildings.path, 'investment_income.png'), total=False, loc='left',
                   format_y=lambda y, _: '{:.0f}'.format(y / 10 ** 9))

    df = investment_total.groupby(['Housing type', 'Occupancy status']).sum().T
    df.columns = df.columns.map(lambda x: '{} - {}'.format(x[0], x[1]))
    df = df.loc[:, generic_input['index']['Decision maker']]
    make_area_plot(df, 'Investment (Billion euro)',
                   save=os.path.join(buildings.path, 'investment_decision_maker.png'), total=False,
                   format_y=lambda y, _: '{:.0f}'.format(y / 10 ** 9),
                   colors=generic_input['colors'], loc='left', left=1.25)

    # graph consumption
    t = detailed.loc[['Embodied energy renovation (TWh PE)', 'Embodied energy construction (TWh PE)'], :]
    t.index = ['Renovation', 'Construction']
    temp = consumption.groupby('Existing').sum().rename(index={True: 'Existing', False: 'Construction'}).T
    if 'Construction' in temp.columns:
        temp = temp.loc[:, ['Existing', 'Construction']]
        temp.columns = ['Existing', 'New']
    else:
        temp = temp.loc[:, 'Existing']
    temp = pd.concat((temp / 10**9, t.T), axis=1).dropna(how='any')
    make_area_plot(temp, 'Consumption (TWh)', colors=generic_input['colors'],
                   save=os.path.join(buildings.path, 'consumption.png'), total=False,
                   format_y=lambda y, _: '{:.0f}'.format(y), loc='left', left=1.1)

    # graph emissions
    t = detailed.loc[['Carbon footprint renovation (MtCO2)', 'Carbon footprint construction (MtCO2)'], :]
    t.index = ['Renovation', 'Construction']
    temp = emission.groupby('Existing').sum().rename(index={True: 'Existing', False: 'Construction'}).T
    if 'Construction' in temp.columns:
        temp = temp.loc[:, ['Existing', 'Construction']]
        temp.columns = ['Existing', 'New']
    else:
        temp = temp.loc[:, 'Existing']
    temp = pd.concat((temp / 10 ** 12, t.T), axis=1).dropna(how='any')
    make_area_plot(temp, 'Emission (MtCO2)', colors=generic_input['colors'],
                   save=os.path.join(buildings.path, 'emission.png'), total=False,
                   format_y=lambda y, _: '{:.0f}'.format(y), loc='left', left=1.1)

    df = stock.groupby('Performance').sum().T.sort_index(axis=1, ascending=False)
    make_area_plot(df, 'Dwelling stock (Millions)', colors=generic_input['colors'],
                   format_y=lambda y, _: '{:.0f}'.format(y / 10 ** 6),
                   save=os.path.join(buildings.path, 'stock_performance.png'), total=False,
                   loc='left')

    consumption = pd.concat((consumption, certificate, energy), axis=1).set_index(['Performance', 'Energy'],
                                                                                  append=True)

    df = consumption.groupby('Performance').sum().T.sort_index(axis=1, ascending=False)
    make_area_plot(df, 'Energy consumption (TWh)', colors=generic_input['colors'],
                   format_y=lambda y, _: '{:.0f}'.format(y / 10 ** 9),
                   save=os.path.join(buildings.path, 'consumption_performance.png'), loc='left')

    df = consumption.groupby('Energy').sum().T.loc[:, generic_input['index']['Heating energy']]
    make_area_plot(df, 'Energy consumption (TWh)', colors=generic_input['colors'],
                   format_y=lambda y, _: '{:.0f}'.format(y / 10 ** 9),
                   save=os.path.join(buildings.path, 'consumption_energy.png'),
                   total=False, loc='left', left=1.1)

    df = consumption.groupby('Income tenant').sum().T.loc[:, generic_input['index']['Income tenant']]
    make_area_plot(df, 'Energy consumption (TWh)', colors=generic_input['colors'],
                   format_y=lambda y, _: '{:.0f}'.format(y / 10 ** 9),
                   save=os.path.join(buildings.path, 'consumption_income.png'), loc='left', total=False)

    return stock, detailed


def indicator_policies(result, folder, config, discount_rate=0.032, years=30):

    folder_policies = os.path.join(folder, 'policies')
    os.mkdir(folder_policies)

    if 'Discount rate' in config.keys():
        discount_rate = float(config['Discount rate'])

    if 'Lifetime' in config.keys():
        years = int(config['Lifetime'])

    policy_name = config['Policy name'].replace('_', ' ').capitalize()

    def double_difference(ref, scenario, values=None, discount_rate=discount_rate, years=years):
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
        discount = pd.Series([1 / (1 + discount_rate) ** i for i in range(years)])

        if values is None:
            result = double_diff * discount.sum()

        else:
            values = values.reindex(range(min(double_diff.index), max(double_diff.index + years)), method='pad')

            matrix_discount = pd.DataFrame([pd.Series([1 / (1 + discount_rate) ** (i - start) for i in range(start, start + years)],
                                           index=range(start, start + years)) for start in double_diff.index], index=double_diff.index)
            result = (double_diff * (matrix_discount * values).T).T
            result = result.sum(axis=1)

        discount = pd.Series([1 / (1 + discount_rate) ** i for i in range(result.shape[0])],
                             index=result.index)
        return (result * discount).sum()

    # Getting inputs needed
    energy_prices = pd.read_csv(config['energy_prices'], index_col=[0]) * 10 ** 9  # euro/kWh to euro/TWh
    carbon_value = pd.read_csv(config['carbon_value'], index_col=[0]).squeeze()  # euro/tCO2
    carbon_emission = pd.read_csv(config['carbon_emission'], index_col=[0]) * 10 ** 3  # unit: gCO2/ kWh to tCO2/ TWh
    # euro/tCO2 * tCO2/TWh  = euro/TWh
    carbon_emission_value = (carbon_value * carbon_emission.T).T  # euro/TWh
    carbon_emission_value.dropna(how='all', inplace=True)

    scenarios = [s for s in result.keys() if s != 'Reference' and s != 'ZP']

    # Calculating simple and double differences for needed variables, and storing them in agg
    # Double difference = Scenario - Reference
    comparison = {}
    for s in scenarios:
        if 'AP' in s:
            ref = result['Reference']
        else:
            ref = result['ZP']

        rslt = {}
        for var in ['Consumption standard (TWh)', 'Consumption (TWh)', 'Energy poverty (Million)',
                    'Health expenditure (Billion euro)', 'Social cost of mortality (Billion euro)',
                    'Loss of well-being (Billion euro)', 'Health cost (Billion euro)', 'Emission (MtCO2)']:
            rslt[var] = double_difference(ref.loc[var, :], result[s].loc[var, :],
                                          values=None)

        for energy in generic_input['index']['Heating energy']:
            var = 'Consumption {} (TWh)'.format(energy)
            rslt[var] = double_difference(ref.loc[var, :], result[s].loc[var, :],
                                          values=None)
            rslt['Energy expenditures {} (Billion euro)'.format(energy)] = double_difference(
                ref.loc[var, :],
                result[s].loc[var, :],
                values=energy_prices[energy]) / (10 ** 9)
            # On a des euros

            rslt['Emission {} (tCO2)'.format(energy)] = double_difference(ref.loc[var, :],
                                                                          result[s].loc[var, :],
                                                                          values=carbon_emission[energy])

            rslt['Carbon value {} (Billion euro)'.format(energy)] = double_difference(ref.loc[var, :],
                                                                                      result[s].loc[var, :],
                                                                                      values=carbon_emission_value[energy]) / (
                                                                            10 ** 9)

        # Simple diff = scenario - ref
        for var in ['Subsidies total (Billion euro)', 'VTA (Billion euro)', 'Investment total (Billion euro)',
                    'Health expenditure (Billion euro)']:
            discount = pd.Series(
                [1 / (1 + discount_rate) ** i for i in range(ref.loc[var, :].shape[0])],
                index=ref.loc[var, :].index)
            if var == 'Health expenditure (Billion euro)':
                rslt['Simple difference ' + var] = ((result[s].loc[var, :] - ref.loc[var, :]) * discount.T).sum()
            else:
                rslt[var] = ((result[s].loc[var, :] - ref.loc[var, :]) * discount.T).sum()


        var = 'Carbon footprint (MtCO2)'
        discount = pd.Series([1 / (1 + discount_rate) ** i for i in range(ref.loc[var, :].shape[0])],
                             index=ref.loc[var, :].index)
        rslt['Carbon footprint (Billion euro)'] = ((result[s].loc[var,
                                                    :] - ref.loc[var, :]) * discount.T * carbon_value).sum() / 10**3

        var = '{} (Billion euro)'.format(policy_name)
        discount = pd.Series([1 / (1 + discount_rate) ** i for i in range(ref.loc[var, :].shape[0])],
                             index=ref.loc[var, :].index)

        if var in result[s].index: #model calibrated without Mpr so can be missing
            rslt[var] = (((result[s].loc[var, :]).fillna(0) - ref.loc[var, :]) * discount.T).sum()
            # We had NaN for year t with AP-t scnarios, so replaced these with 0... is it ok?
        else:
            rslt[var] = (- ref.loc[var, :] * discount.T).sum()
        comparison[s] = rslt

    comparison = pd.DataFrame(comparison)

    # Efficiency: AP and AP-t scenarios
    efficiency_scenarios = list(set(comparison.columns).intersection(['AP-{}'.format(y) for y in range(2018, 2050)]))
    indicator = dict()
    if efficiency_scenarios:
        # We want efficiency only for concerned scenario policy (that is cut at t-1)
        comp_efficiency = comparison.loc[:, efficiency_scenarios]

        policy_cost = comp_efficiency.loc['{} (Billion euro)'.format(policy_name)]
        indicator.update({'{} (Billion euro)'.format(policy_name): policy_cost})
        indicator.update({'Consumption (TWh)': comp_efficiency.loc['Consumption (TWh)']})
        indicator.update({'Consumption standard (TWh)': comp_efficiency.loc['Consumption standard (TWh)']})
        indicator.update({'Emission (MtCO2)': comp_efficiency.loc['Emission (MtCO2)']})
        indicator.update({"Cost effectiveness (euro/kWh)": - policy_cost / comp_efficiency.loc['Consumption (TWh)']})
        indicator.update({"Cost effectiveness standard (euro/kWh)": - policy_cost / comp_efficiency.loc['Consumption standard (TWh)']})
        indicator.update({"Cost effectiveness carbon (euro/tCO2)": - policy_cost / comp_efficiency.loc['Emission (MtCO2)'] * 10**3})
        indicator.update({"Leverage (%)": comp_efficiency.loc['Investment total (Billion euro)'] / policy_cost})
        indicator.update({"Investment / energy savings (€/kWh)": comp_efficiency.loc['Investment total (Billion euro)'] / comp_efficiency.loc['Consumption (TWh)']})
        indicator = pd.DataFrame(indicator).T

        # Retrofit ratio = freerider ratio
        # And impact on retrofit rate : difference in retrofit rate / cost of subvention
        for s in efficiency_scenarios:
            year = int(s[-4:])
            if year in result['Reference'].columns:
                indicator.loc['Freeriding retrofit (Thousand)', s] = result[s].loc['Retrofit (Thousand)', year]
                indicator.loc['Non-freeriding retrofit (Thousand)', s] = result['Reference'].loc['Retrofit (Thousand)', year] - (
                    result[s].loc['Retrofit (Thousand)', year])
                indicator.loc['Freeriding retrofit ratio (%)', s] = result[s].loc['Retrofit (Thousand)', year] / (
                    result['Reference'].loc['Retrofit (Thousand)', year])
                decile = ['D{}'.format(i) for i in range(1, 11)]
                df = result[s].loc[['Retrofit {} (Thousand)'.format(d) for d in decile], year] / \
                     result['Reference'].loc[['Retrofit {} (Thousand)'.format(d) for d in decile], year]
                df.index = ['Freeriding retrofit ratio {} (%)'.format(d) for d in decile]
                df.name = s
                df = pd.DataFrame(df)
                #This part to be improved
                if set(list(df.index)).issubset(list(indicator.index)):
                    indicator.loc[list(df.index), s] = df[s]
                else:
                    indicator = pd.concat((indicator, df), axis=0)

                indicator.loc['Retrofit rate difference (%)', s] = result['Reference'].loc['Retrofit rate (%)', year] - (
                    result[s].loc['Retrofit rate (%)', year])
                indicator.loc['Impact on retrofit rate (%)', s] = (result['Reference'].loc['Retrofit rate (%)', year] - (
                    result[s].loc['Retrofit rate (%)', year])) / comparison.loc['{} (Billion euro)'.format(policy_name), s]
    else:
        indicator = pd.DataFrame(indicator).T

        # Effectiveness : AP/AP-1 and ZP/ ZP+1 scenarios

    def socioeconomic_npv(data, scenarios, save=None, factor_cofp=0.2, embodied_emission=True, cofp=True):
        """Calculate socioeconomic NPV.
        Double difference is calculated with : scenario - reference
        If the scenario requires more investment than the reference, then the difference of investments is
        positive, and it is taken into account in the NPV as a negative impact: - Investment total
        If the scenario results in less energy consumed then the reference, then energy savings is positive,
        and taken into account in the NPV as a positive account.

        Parameters
        ----------
        data: pd.DataFrame
        scenarios: list
        save: str, default None
        factor_cofp: float, default 0.2
        embodied_emission: bool
        cofp: bool

        Returns
        -------
        pd.DataFrame
        """
        npv = {}
        for s in scenarios:
            df = data.loc[:, s]
            temp = dict()
            temp.update({'Investment': df['Investment total (Billion euro)']})
            if embodied_emission:
                temp.update({'Embodied emission additional': - df['Carbon footprint (Billion euro)']})
            if cofp:
                temp.update({'Cofp': (df['Subsidies total (Billion euro)'] - df['VTA (Billion euro)'] +
                                      df['Simple difference Health expenditure (Billion euro)']
                                      ) * factor_cofp})

            temp.update({'Energy saving': sum(df['Energy expenditures {} (Billion euro)'.format(i)]
                                              for i in generic_input['index']['Heating energy'])})
            temp.update({'Emission saving': sum(df['Carbon value {} (Billion euro)'.format(i)]
                                                for i in generic_input['index']['Heating energy'])})
            temp.update({'Well-being benefit': df['Loss of well-being (Billion euro)']})
            temp.update({'Health savings': df['Health expenditure (Billion euro)']})
            temp.update({'Mortality reduction benefit': df['Social cost of mortality (Billion euro)']})
            temp = - pd.Series(temp) #minus sign for convention

            if save:
                waterfall_chart(temp, title=s,
                                save=os.path.join(save, 'npv_{}.png'.format(s.lower().replace(' ', '_'))),
                                colors=generic_input['colors'])

            npv[s] = temp

        npv = pd.DataFrame(npv)
        if save:
            assessment_scenarios(npv.T, save=os.path.join(save, 'npv.png'.lower().replace(' ', '_')), colors=generic_input['colors'])

        npv.loc['NPV', :] = npv.sum()
        return npv

    effectiveness_scenarios = [s for s in comparison.columns if s not in efficiency_scenarios]
    if effectiveness_scenarios:
        se_npv = socioeconomic_npv(comparison, effectiveness_scenarios, save=folder_policies)
        if indicator is not None:
            if set(list(se_npv.index)).issubset(list(indicator.index)):
                indicator.loc[list(se_npv.index), s] = se_npv[s]
            else:
                indicator = pd.concat((indicator, se_npv), axis=0)
        else:
            indicator = se_npv
        # Percentage of objectives accomplished

        # Objectives in param (generic_input), we need to make this cleaner but for now:

        """"* result['Reference'].loc['Sizing factor (%)'].iloc[0]"""
        # TODO: have sizing factor in generic input
        comparison_results_energy = pd.DataFrame([result[s].loc['Consumption (TWh)'] for s in effectiveness_scenarios],
                                                 index=effectiveness_scenarios).T
        comparison_results_emissions = pd.DataFrame([result[s].loc['Emission (MtCO2)'] for s in effectiveness_scenarios],
                                                    index=effectiveness_scenarios).T

        # Selecting years with corresponding objectives and calculating the % of objective accomplished
        for y in generic_input['consumption_total_objectives'].index:
            if y in comparison_results_energy.index:
                indicator.loc['Consumption reduction {} (TWh) '.format(y), :] = (comparison_results_energy.iloc[0] -
                                                                                 comparison_results_energy.loc[y]).T

                indicator.loc['Consumption reduction Obj {} (TWh)'.format(y), :] = (comparison_results_energy.iloc[0] -
                                                                                    generic_input['consumption_total_objectives'].loc[y]).T

                indicator.loc['Percentage of {} consumption objective (%)'.format(y), :] = (comparison_results_energy.iloc[0] -
                                                                                            comparison_results_energy.loc[y]).T / (
                                                                                            comparison_results_energy.iloc[0] -
                                                                                            generic_input['consumption_total_objectives'].loc[y]).T

        for y in generic_input['emissions_total_objectives'] .index:
            if y in comparison_results_emissions.index:
                indicator.loc['Emission reduction {} (MtCO2) '.format(y), :] = (comparison_results_emissions.iloc[0] -
                                                                                comparison_results_emissions.loc[y]).T

                indicator.loc['Emission reduction Obj {} (MtCO2)'.format(y), :] = (comparison_results_emissions.iloc[0] -
                                                                                   generic_input['emissions_total_objectives'] .loc[y]).T

                indicator.loc['Percentage of {} emission objective (%)'.format(y), :] = (comparison_results_emissions.iloc[0] -
                                                                                         comparison_results_emissions.loc[y]).T / (
                                                                                         comparison_results_emissions.iloc[0] -
                                                                                         generic_input['emissions_total_objectives'] .loc[y]).T
        # low_eff_var = 'Stock low-efficient (Million)'
        # Objective is zero in 2030 - introduce it in params to make it resilient
        comparison_results_low_eff = pd.DataFrame([result[s].loc['Stock low-efficient (Million)']
                                                   for s in effectiveness_scenarios], index=effectiveness_scenarios).T
        for y in generic_input['low_eff_objectives'].index:
            if y in comparison_results_low_eff.index:
                indicator.loc['Low-efficient stock reduction {} (Million) '.format(y), :] = (comparison_results_low_eff.iloc[0] -
                                                                                             comparison_results_low_eff.loc[y]).T

                indicator.loc['Low-efficient stock reduction objective {} (Million) '.format(y), :] = (comparison_results_low_eff.iloc[0] -
                                                                                                       generic_input['low_eff_objectives'].loc[y]).T

                indicator.loc['Percentage of {} low-efficient objective (%) '.format(y), :] = (comparison_results_low_eff.iloc[0] -
                                                                                               comparison_results_low_eff.loc[y]).T / (
                                                                                               comparison_results_low_eff.iloc[0] -
                                                                                               generic_input['low_eff_objectives'].loc[y]).T
        # Energy poverty
        # No objective so simply showing the reduction between first and last year
        # first year - last year
        energy_poverty = pd.DataFrame([result[s].loc['Energy poverty (Million)'] for s in effectiveness_scenarios],
                                      index=effectiveness_scenarios).T
        indicator.loc['Energy poverty (Thousand)'] = (energy_poverty.iloc[0] - energy_poverty.iloc[-1]) * 10**3

        # Counting the number of years after 2030 when number of retrofit >= objective of 700 000
        """
        retrofit_obj = pd.Series([700], index=[2030], name='Objectives')
        comparison_results_retrofit = pd.DataFrame([result[s].loc['New efficient (Thousand)']
                                                    for s in effectiveness_scenarios], index=effectiveness_scenarios).T
        #à revoir deux boucles for
        for year in retrofit_obj.index:
            comparison_results_retrofit = comparison_results_retrofit[comparison_results_retrofit.index > year]
            for s in comparison_results_retrofit.columns:
                comparison_results_retrofit[s] = [True if retrofit > retrofit_obj.loc[year] else False for retrofit in
                                                  comparison_results_retrofit[s]]
                comparison.loc['Percentage of {} retrofit objective - {}'. format(year, policy_name), :] = (comparison_results_retrofit.sum() / len(comparison_results_retrofit.index)).T
        """

    comparison.round(2).to_csv(os.path.join(folder_policies, 'comparison.csv'))
    if indicator is not None:
        indicator.round(2).to_csv(os.path.join(folder_policies, 'indicator.csv'))

    return comparison, indicator


def grouped_output(result, stocks, folder, config_runs=None, config_sensitivity=None):
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

    folder_img = os.path.join(folder, 'img')
    os.mkdir(folder_img)

    variables = {'Consumption (TWh)': ('consumption_hist.png', lambda y, _: '{:,.0f}'.format(y),
                                       generic_input['consumption_total_hist'],
                                       generic_input['consumption_total_objectives']),
                 'Consumption standard (TWh)': ('consumption_standard.png', lambda y, _: '{:,.0f}'.format(y)),
                 'Heating intensity (%)': ('heating_intensity.png', lambda y, _: '{:,.0%}'.format(y)),
                 'Emission (MtCO2)': ('emission.png', lambda y, _: '{:,.0f}'.format(y)),
                 'Energy poverty (Million)': ('energy_poverty.png', lambda y, _: '{:,.1f}'.format(y)),
                 'Stock low-efficient (Million)': ('stock_low_efficient.png', lambda y, _: '{:,.0f}'.format(y)),
                 'Stock efficient (Million)': ('stock_efficient.png', lambda y, _: '{:,.0f}'.format(y)),
                 'Retrofit >= 1 EPC (Thousand)': ('retrofit.png', lambda y, _: '{:,.0f}'.format(y),
                                                  generic_input['retrofit_comparison']),
                 'Efficient retrofits (Thousand)': ('renovation_efficient.png', lambda y, _: '{:,.0f}'.format(y)),
                 'Global retrofits (Thousand)': ('renovation_global.png', lambda y, _: '{:,.0f}'.format(y)),
                 'Investment total (Billion euro)': ('investment_total.png', lambda y, _: '{:,.0f}'.format(y)),
                 'Subsidies total (Billion euro)': ('subsidies_total.png', lambda y, _: '{:,.0f}'.format(y)),
                 'Energy expenditures (Billion euro)': (
                 'energy_expenditures.png', lambda y, _: '{:,.0f}'.format(y)),
                 'Health cost (Billion euro)': ('health_cost.png', lambda y, _: '{:,.0f}'.format(y)),
                 'Replacement insulation (Thousand)': ('replacement_insulation_total.png', lambda y, _: '{:,.0f}'.format(y)),
                 'Replacement heater (Thousand)': ('replacement_heater.png', lambda y, _: '{:,.0f}'.format(y))
                 }

    for variable, infos in variables.items():
        temp = pd.DataFrame({scenario: output.loc[variable, :] for scenario, output in result.items()})
        try:
            temp = pd.concat((temp, infos[2]), axis=1)
            temp.sort_index(inplace=True)
        except IndexError:
            continue

        try:
            scatter = infos[3]
        except IndexError:
            scatter = None

        make_plot(temp, variable, save=os.path.join(folder_img, '{}'.format(infos[0])), format_y=infos[1], scatter=scatter)

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
        'Investment per insulation action {} (euro)': [('Income owner', lambda y, _: '{:,.0f}'.format(y)),
                                                       ('Decision maker', lambda y, _: '{:,.0f}'.format(y), 2)],
        'Investment total {} (Billion euro)': [
            ('Decision maker', lambda y, _: '{:,.0f}'.format(y), 2)],
        'Replacement {} (Thousand)': [
            ('Insulation', lambda y, _: '{:,.0f}'.format(y), 2, None, generic_input['retrofit_hist'])],
        'Replacement insulation {} (Thousand)': [
            ('Decision maker', lambda y, _: '{:,.0f}'.format(y), 2)],
        'Replacement insulation {} (%)': [
            ('Decision maker', lambda y, _: '{:,.0%}'.format(y), 2)],
        'Retrofit rate {} (%)': [
            ('Decision maker', lambda y, _: '{:,.0%}'.format(y), 2)],
        'Non-weighted retrofit rate {} (%)': [
            ('Decision maker', lambda y, _: '{:,.0%}'.format(y), 2)],
        'Retrofit rate heater {} (%)': [
            ('Decision maker', lambda y, _: '{:,.0%}'.format(y), 2)],
        'Non-weighted retrofit rate heater {} (%)': [
            ('Decision maker', lambda y, _: '{:,.0%}'.format(y), 2)],
    }

    def details_graphs(data, v, inf, folder_img):
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

        make_grouped_subplots(temp, format_y=inf[1], n_columns=n_columns, save=os.path.join(folder_img, n), scatter=scatter,
                              order=generic_input['index'][inf[0]])

    for var, infos in variables_detailed.items():
        for info in infos:
            details_graphs(result, var, info, folder_img)

    if 'Reference' in result.keys() and len(result.keys()) > 1 and config_runs is not None:
        indicator_policies(result, folder, config_runs)
