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
import seaborn as sns
from project.input.resources import resources_data
from project.utils import make_plot, make_grouped_subplots, make_area_plot, waterfall_chart, \
    make_uncertainty_plot, format_table, select, make_clusterstackedbar_plot, plot_ldmi_method, make_stacked_bar_subplot
from project.utils import stack_catplot, make_relplot, make_stackedbar_plot, make_scatter_plot, get_pandas, get_series
from itertools import product
from PIL import Image
from numpy import log
import re
from project.utils import calculate_annuities


def decomposition_analysis(output, save=None):
    """Estimates emissions reduction attribution across different channels with the LDMI method."""

    start, end = output.columns[0], output.columns[-1]

    delta_emission = output.loc['Emission (MtCO2)', end] - output.loc['Emission (MtCO2)', start]

    heater_vector, energy_vector = resources_data['index']['Heating system'], resources_data['index']['Energy']

    # select only heater_vector that have 'Surface {} (Million m2)' in output index
    heater_vector = [i for i in heater_vector if 'Surface {} (Million m2)'.format(i) in output.index]

    # Select rows
    rows = []
    rows += ['Surface (Million m2)']
    rows += ['Surface {} (Million m2)'.format(i) for i in heater_vector]
    rows += ['Consumption standard {} (TWh)'.format(i) for i in heater_vector]
    rows += ['Consumption {} (TWh)'.format(i) for i in heater_vector]
    rows += ['Emission content {} (gCO2/kWh)'.format(i) for i in energy_vector]
    rows = [i for i in rows if i in output.index]
    data = output.loc[rows, [start, end]]

    # Prepare indicators
    for i in heater_vector:
        data.loc['Share surface {} (%)'.format(i), :] = data.loc['Surface {} (Million m2)'.format(i), :] / data.loc[
            'Surface (Million m2)', :]
        data.loc['Consumption standard {} (TWh/m2)'.format(i), :] = data.loc['Consumption standard {} (TWh)'.format(i), :] / data.loc[
            'Surface {} (Million m2)'.format(i), :]
        data.loc['Heating intensity {} (%)'.format(i), :] = data.loc['Consumption {} (TWh)'.format(i), :] / data.loc[
            'Consumption standard {} (TWh)'.format(i), :]
        data.loc['Emission content {} (gCO2/kWh)'.format(i), :] = data.loc['Emission content {} (gCO2/kWh)'.format(i.split('-')[0]), :]
        data.loc['Emission {} (MtCO2)'.format(i), :] = data.loc['Consumption {} (TWh)'.format(i), :] * data.loc[
            'Emission content {} (gCO2/kWh)'.format(i), :] / 1000
    # in case heating system is not used at the end of the period
    data.fillna(0, inplace=True)

    # Calculate individual effect
    channels_heater = ['Surface (Million m2)', 'Share surface {} (%)', 'Consumption standard {} (TWh/m2)',
                       'Heating intensity {} (%)', 'Emission content {} (gCO2/kWh)']
    result = {}
    manual_treatment = []
    for i in heater_vector:
        # no more emission if no more surface
        if data.loc['Surface {} (Million m2)'.format(i), end] == 0:
            result['Surface {} (Million m2)'.format(i)] = - data.loc['Emission {} (MtCO2)'.format(i), start]
            manual_treatment += [i]

    for channel in channels_heater:
        result[channel.split(' {}')[0]] = sum([log(data.loc[channel.format(i), end] / data.loc[channel.format(i), start]) *
                         (data.loc['Emission {} (MtCO2)'.format(i), end] - data.loc['Emission {} (MtCO2)'.format(i), start]) /
                         log(data.loc['Emission {} (MtCO2)'.format(i), end] / data.loc['Emission {} (MtCO2)'.format(i), start]) for i in heater_vector if i not in manual_treatment])

    assert abs(sum(result.values()) - delta_emission) < 0.01, 'Error in decomposition analysis'

    rename = {'Surface (Million m2)': 'Surface',
            'Share surface': 'Switch\nheater',
            'Consumption standard': 'Insulation',
            'Heating intensity': 'Heating\nintensity',
            'Emission content': 'Carbon\ncontent'}
    result = pd.Series({rename[k]: v for k, v in result.items()})
    emission = output.loc['Emission (MtCO2)', [start, end]]

    colors = {'Switch\nheater': 'royalblue',
              'Insulation': 'firebrick',
              'Heating\nintensity': 'darkorange',
              'Carbon\ncontent': 'forestgreen',
              'Surface': 'grey'}

    plot_ldmi_method(result, emission, save=save, colors=colors)


def plot_scenario(output, stock, buildings, detailed_graph=False):
    path = os.path.join(buildings.path, 'img')
    if not os.path.isdir(path):
        os.mkdir(path)

    if buildings.quintiles:
        resources_data['index']['Income tenant'] = resources_data['quintiles']
        resources_data['index']['Income owner'] = resources_data['quintiles']

    # decomposition analysis
    # decomposition_analysis(output, save=os.path.join(path, 'decomposition.png'))

    # energy consumption
    df = output.loc[['Consumption {} (TWh)'.format(i) for i in resources_data['index']['Energy']], :].T
    df.columns = resources_data['index']['Energy']
    make_area_plot(df, 'Energy consumption (TWh)', colors=resources_data['colors'],
                   format_y=lambda y, _: '{:.0f}'.format(y),
                   save=os.path.join(path, 'consumption_energy.png'),
                   total=False, loc='left', left=1.2) # scatter=resources_data['consumption_total_objectives']

    df = output.loc[['Consumption {} (TWh)'.format(i) for i in resources_data['index']['Heater']], :].T
    df.columns = resources_data['index']['Heater']
    df = df.loc[:, (df > 0).all()]
    make_area_plot(df, 'Energy consumption (TWh)', colors=resources_data['colors'],
                   format_y=lambda y, _: '{:.0f}'.format(y),
                   save=os.path.join(path, 'consumption_heater.png'),
                   total=False, loc='left', left=1.2) # scatter=resources_data['consumption_total_objectives']

    saving = {'Consumption saving insulation (TWh/year)': 'Saving insulation',
              'Consumption saving heater (TWh/year)': 'Saving heater',
              'Consumption saving prices effect (TWh/year)': 'Saving prices'}
    # impossible because negative value:

    temp = output.loc[saving.keys(), :].fillna(0).cumsum(axis=1)
    if (temp > 0).all().all():

        temp.index = saving.values()
        df = pd.concat((df, temp.T), axis=1)
        make_area_plot(df, 'Energy consumption (TWh)', colors=resources_data['colors'],
                       format_y=lambda y, _: '{:.0f}'.format(y),
                       save=os.path.join(path, 'consumption_heater_saving.png'),
                       total=False, loc='left', left=1.2)

    # consumption existing vs new
    if detailed_graph:
        t = output.loc[['Embodied energy renovation (TWh PE)', 'Embodied energy construction (TWh PE)'], :]
        t.index = ['Renovation', 'Construction']
        if 'Consumption New (TWh)' in output.index:
            l = ['Existing', 'New']
        else:
            l = ['Existing']
        temp = output.loc[['Consumption {} (TWh)'.format(i) for i in l], :]
        temp.index = l
        temp = pd.concat((temp.T, t.T), axis=1).fillna(0)
        make_area_plot(temp.loc[buildings.first_year + 1:, :], 'Consumption (TWh)', colors=resources_data['colors'],
                       save=os.path.join(path, 'consumption.png'), total=False,
                       format_y=lambda y, _: '{:.0f}'.format(y), loc='left', left=1.1)

    # emission
    t = output.loc[['Carbon footprint renovation (MtCO2)', 'Carbon footprint construction (MtCO2)'], :]
    t.index = ['Renovation', 'Construction']
    if 'Emission New (MtCO2)' in output.index:
        l = ['Existing', 'New']
    else:
        l = ['Existing']
    temp = output.loc[['Emission {} (MtCO2)'.format(i) for i in l], :]
    temp.index = l
    temp = pd.concat((temp.T, t.T), axis=1).fillna(0)
    make_area_plot(temp.loc[buildings.first_year + 1:, :], 'Emission (MtCO2)', colors=resources_data['colors'],
                   save=os.path.join(path, 'emission.png'), total=False,
                   format_y=lambda y, _: '{:.0f}'.format(y), loc='left', left=1.2
                   ) # scatter=resources_data['emissions_total_objectives']

    # building stock performance
    df = stock.groupby('Performance').sum().T.sort_index(axis=1, ascending=False)
    make_area_plot(df, 'Dwelling stock (Million)', colors=resources_data['colors'],
                   format_y=lambda y, _: '{:.0f}'.format(y / 10 ** 6),
                   save=os.path.join(path, 'stock_performance.png'), total=False,
                   loc='left')

    # building stock heating system
    df = output.loc[['Stock {} (Million)'.format(i) for i in resources_data['index']['Heater']], :].T
    df.columns = resources_data['index']['Heater']
    make_area_plot(df, 'Dwelling stock (Million)', colors=resources_data['colors'],
                   format_y=lambda y, _: '{:.0f}'.format(y),
                   save=os.path.join(path, 'stock_heater.png'), total=False,
                   loc='left', left=1.1)

    # replacement insulation
    df = pd.DataFrame(
        [output.loc['Replacement {} (Thousand households)'.format(i), :] for i in resources_data['index']['Insulation']]).T.dropna()
    df.columns = resources_data['index']['Insulation']
    make_area_plot(df, 'Replacement (Thousand households)',
                   save=os.path.join(path, 'replacement_insulation.png'), total=False,
                   format_y=lambda y, _: '{:.0f}'.format(y), colors=resources_data['colors'], loc='left', left=1.1)

    i = ['Switch heater only (Thousand households)',
         'Renovation with heater replacement (Thousand households)',
         'Renovation endogenous (Thousand households)',
         'Renovation obligation (Thousand households)'
         ]
    colors = ['royalblue', 'tomato', 'darksalmon', 'grey']

    df = output.loc[i, :].T
    df.dropna(inplace=True)
    df.columns = [i.split(' (')[0] for i in df.columns]
    make_area_plot(df, 'Retrofit (Thousand households)',
                   save=os.path.join(path, 'retrofit.png'), total=False,
                   format_y=lambda y, _: '{:.0f}'.format(y),
                   loc='left', left=1.3, colors=colors)

    i = ['Switch decarbonize (Thousand households)',
         'Insulation (Thousand households)',
         'Insulation and switch decarbonize (Thousand households)'
         ]
    colors = ['royalblue', 'darksalmon', 'tomato']

    df = output.loc[i, :].T
    df.dropna(inplace=True)
    df.columns = [i.split(' (')[0] for i in df.columns]
    make_area_plot(df, 'Retrofit (Thousand households)',
                   save=os.path.join(path, 'retrofit_decarbonize_options.png'), total=False,
                   format_y=lambda y, _: '{:.0f}'.format(y),
                   loc='left', left=1.3, colors=colors)

    df = output.loc[['Renovation {} (Thousand households)'.format(i) for i in resources_data['index']['Decision maker']], :].T
    df.dropna(inplace=True)
    df.columns = resources_data['index']['Decision maker']
    make_area_plot(df, 'Renovation (Thousand households)',
                   save=os.path.join(path, 'renovation_decision_maker.png'), total=False,
                   format_y=lambda y, _: '{:.0f}'.format(y),
                   colors=resources_data['colors'], loc='left', left=1.25)

    df = output.loc[['Retrofit measures {} (Thousand households)'.format(i) for i in resources_data['index']['Count']], :].T
    df.dropna(inplace=True)
    df.columns = resources_data['index']['Count']
    make_area_plot(df, 'Retrofit measures (Thousand households)',
                   save=os.path.join(path, 'retrofit_measures.png'), total=False,
                   format_y=lambda y, _: '{:.0f}'.format(y),
                   colors=resources_data['colors'], loc='left', left=1.25)

    # switch heating system
    c = ['Switch {} (Thousand households)'.format(i) for i in resources_data['index']['Heating system']]
    c = [i for i in c if i in output.index]
    df = output.loc[c, :].T.dropna()
    df.columns = df.columns.map(lambda x: x.split('Switch ')[1].split(' (Thousand households)')[0])
    make_area_plot(df, 'Switch (Thousand households)',
                   save=os.path.join(path, 'switch_heater.png'), total=False,
                   format_y=lambda y, _: '{:.0f}'.format(y),
                   colors=resources_data['colors'], loc='left', left=1.25)
    if detailed_graph is True:
        df = pd.DataFrame(
            [output.loc['Switch Multi-family {} (Thousand households)'.format(i), :] for i in resources_data['index']['Heating system']]).T.dropna()
        df.columns = resources_data['index']['Heating system']
        make_area_plot(df, 'Switch (Thousand households)',
                       save=os.path.join(path, 'switch_heater_mf.png'), total=False,
                       format_y=lambda y, _: '{:.0f}'.format(y),
                       colors=resources_data['colors'], loc='left', left=1.25)

        df = pd.DataFrame([output.loc['Switch Single-family {} (Thousand households)'.format(i), :] for i in
                           resources_data['index']['Heating system']]).T.dropna()
        df.columns = resources_data['index']['Heating system']
        make_area_plot(df, 'Switch (Thousand households)',
                       save=os.path.join(path, 'switch_heater_sf.png'), total=False,
                       format_y=lambda y, _: '{:.0f}'.format(y),
                       colors=resources_data['colors'], loc='left', left=1.25)

    # emission saving
    variables = {
        'Emission saving heater (MtCO2/year)': 'Switch heater induced',
        'Emission saving insulation (MtCO2/year)': 'Insulation induced',
        'Emission saving carbon content (MtCO2/year)': 'Carbon content',
        'Emission saving prices (MtCO2/year)': 'Prices induced',
    }
    colors = {
        'Prices induced': 'grey',
        'Switch heater induced': 'royalblue',
        'Insulation induced': 'firebrick',
        'Carbon content': 'forestgreen',
    }
    df = output.loc[variables.keys(), :].dropna(axis=1, how='all').rename_axis('Attribute', axis=0)
    df = df.cumsum(axis=1).round(3)
    df = df.rename(index=variables)
    make_stackedbar_plot(df.T, 'Emission saving (MtCO2)', ncol=2, ymin=None, format_y=lambda y, _: '{:.0f}'.format(y),
                         colors=colors, save=os.path.join(path, 'emission_saving_decomposition.png'), left=1.2)
    # consumption saving
    variables = {
        'CBA Consumption saving insulation (TWh)': 'Insulation induced',
        'CBA Consumption saving switch fuel (TWh)': 'Switch heater induced',
        'CBA Rebound EE (TWh)': 'Rebound effect',
        'CBA Consumption saving prices (TWh)': 'Prices induced',
    }
    colors = {
        'Prices induced': 'grey',
        'Switch heater induced': 'royalblue',
        'Insulation induced': 'firebrick',
        'Rebound effect': 'darkorange'
    }
    df = output.loc[variables.keys(), :].dropna(axis=1, how='all').rename_axis('Attribute', axis=0)
    df = df.cumsum(axis=1).round(3)
    df = df.rename(index=variables)
    lineplot = df.loc[['Insulation induced', 'Switch heater induced', 'Prices induced']].sum().rename('Total')
    make_stackedbar_plot(df.T, 'Consumption saving (TWh)', ncol=2, ymin=None, format_y=lambda y, _: '{:.0f}'.format(y),
                         colors=colors, save=os.path.join(path, 'consumption_saving_decomposition.png'), left=1.2,
                         lineplot=lineplot)

    # total running cost
    """
    variables = {'Cost energy (Billion euro)': 'Energy expenditure',
                 'Loss thermal comfort (Billion euro)': 'Comfort',
                 'Cost emission (Billion euro)': 'Direct emission',
                 'Cost heath (Billion euro)': 'Health cost',
                 'Cost heater (Billion euro)': 'Annuities heater',
                 'Cost insulation (Billion euro)': 'Annuities insulation'}
    df = output.loc[variables.keys(), :].dropna(axis=1, how='all').rename_axis('Attribute', axis=0)
    df = df.rename(index=variables)
    lineplot = df.sum().rename('Total')
    make_stackedbar_plot(df.T, 'Running cost (Billion euro)', ncol=2, ymin=None, format_y=lambda y, _: '{:.0f}'.format(y),
                         colors=resources_data['colors'], save=os.path.join(path, 'running_cost.png'), left=1.2,
                         lineplot=lineplot)
    
    # cost-benefit analysis
    variables = {'CBA Consumption saving EE (Billion euro)': 'Saving EE',
                 'CBA Consumption saving prices (Billion euro)': 'Saving price',
                 'CBA Thermal comfort EE (Billion euro)': 'Comfort EE',
                 'CBA Emission direct (Billion euro)': 'Direct emission',
                 'CBA Thermal loss prices (Billion euro)': 'Comfort prices',
                 'CBA Annuities heater (Billion euro)': 'Annuities heater',
                 'CBA Annuities insulation (Billion euro)': 'Annuities insulation',
                 'CBA Carbon Emission indirect (Billion euro)': 'Indirect emission',
                 'CBA Health cost (Billion euro)': 'Health cost'
    }
    df = output.loc[variables.keys(), :].dropna(axis=1, how='all').rename_axis('Attribute', axis=0)
    df = df.rename(index=variables)
    lineplot = output.loc['Cost-benefits analysis (Billion euro)', :].dropna()
    make_stackedbar_plot(df.T, 'Cost-benefits analysis (Billion euro)', ncol=3, ymin=None, format_y=lambda y, _: '{:.1f}'.format(y),
                         hline=0, lineplot=lineplot, colors=resources_data['colors'],
                         save=os.path.join(path, 'cost_benefit_analysis.png'), left=1.2)
    """
    # subsidies
    non_subsidies = ['subsidies_cap', 'obligation']
    temp = ['{} (Billion euro)'.format(i.capitalize().replace('_', ' ')) for i in buildings.policies + ['reduced_vat'] if i not in non_subsidies]
    subsidies = output.loc[[i for i in temp if i in output.index], :]
    subsidies = subsidies.loc[~(subsidies == 0).all(axis=1)]

    subset = subsidies.copy()

    temp = ['Cee tax (Billion euro)', 'Carbon tax (Billion euro)']
    taxes_expenditures = output.loc[[i for i in temp if i in output.index], :]

    if not taxes_expenditures.empty:
        subset = pd.concat((subsidies, -taxes_expenditures), axis=0)

    subset.fillna(0, inplace=True)
    subset = subset.loc[:, (subset != 0).any(axis=0)].T
    subset.columns = [c.split(' (Billion euro)')[0].capitalize().replace('_', ' ') for c in subset.columns]
    color = [i for i in subset.columns if i not in resources_data['colors']]
    resources_data['colors'].update(dict(zip(color, sns.color_palette(n_colors=len(color)))))
    subset = subset.sort_index(axis=1)

    historic = resources_data['policies_hist'].loc[:, [i for i in subset.columns if i in resources_data['policies_hist'].columns]] / 1000

    if not subset.empty:
        if historic is not None:

            if 'Mpr multifamily' in subset.columns and 'Mpr serenite' in subset.columns:
                subset['Mpr serenite'] += subset['Mpr multifamily']

            df = format_table(subset.rename_axis('Policies', axis=1).T, name='Years')
            df = df[df['Policies'].isin(historic.columns)]
            df_historic = format_table(historic.rename_axis('Policies', axis=1).T, name='Years')
            df = pd.concat((df, df_historic), axis=0, keys=['Simulated', 'Realized'], names=['Source']).reset_index('Source')
            df.reset_index(drop=True, inplace=True)

            yrs = [str(i) for i in subset.index if i in resources_data['policies_hist'].index and i != buildings.first_year]
            df = df[df['Years'].isin(yrs)]

            palette = {s: resources_data['colors'][s] for s in df['Policies'].unique()}
            stack_catplot(x='Years', y='Data', cat='Source', stack='Policies', data=df, palette=palette,
                          y_label='Policies amount (Billion euro)',
                          save=os.path.join(path, 'policies_validation_hash.png'),
                          format_y=lambda y, _: '{:.0f}'.format(y))

            temp = df.copy()
            temp.set_index(['Years', 'Source', 'Policies'], inplace=True)
            temp = temp.squeeze().unstack('Years')

            make_clusterstackedbar_plot(temp, 'Policies', colors=resources_data['colors'],
                                        format_y=lambda y, _: '{:.0f} B€'.format(y),
                                        save=os.path.join(path, 'policies_validation.png'),
                                        rotation=90,
                                        fonttick=20)

            subset = subset.iloc[1:, :]
            make_area_plot(subset, 'Policies cost (Billion euro)', save=os.path.join(path, 'policies.png'),
                           colors=resources_data['colors'], format_y=lambda y, _: '{:.0f} B€'.format(y),
                           loc='left', left=1.2)

        else:
            subset = subset.iloc[1:, :]
            make_area_plot(subset, 'Policies cost (Billion euro)', save=os.path.join(path, 'policies.png'),
                           colors=resources_data['colors'], format_y=lambda y, _: '{:.0f} B€'.format(y),
                           scatter=historic, loc='left', left=1.2)

    # graph public finance
    subset = output.loc[['VAT (Billion euro)', 'Taxes expenditure (Billion euro)', 'Subsidies heater (Billion euro)',
                         'Subsidies insulation (Billion euro)'], :].T
    subset['Subsidies heater (Billion euro)'] = -subset['Subsidies heater (Billion euro)']
    subset['Subsidies insulation (Billion euro)'] = -subset['Subsidies insulation (Billion euro)']
    subset.dropna(how='any', inplace=True)
    subset.columns = [c.split(' (Billion euro)')[0] for c in subset.columns]
    if not subset.empty:
        make_area_plot(subset, 'Public finance (Billion euro)', save=os.path.join(path, 'public_finance.png'),
                       colors=resources_data['colors'],
                       format_y=lambda y, _: '{:.0f}'.format(y), loc='left', left=1.1)

    # graph investment total and financing
    subset = output.loc[['Investment insulation (Billion euro)', 'Investment heater (Billion euro)',
                         'Financing insulation (Billion euro)', 'Financing heater (Billion euro)'], :].T
    subset.dropna(how='any', inplace=True)
    subset.columns = [c.split(' (Billion euro)')[0] for c in subset.columns]
    if not subset.empty:
        make_area_plot(subset, 'Investment (Billion euro)', save=os.path.join(path, 'investment.png'),
                       format_y=lambda y, _: '{:.0f}'.format(y), loc='left', left=1.2,
                       colors=['firebrick', 'royalblue', 'darksalmon', 'lightblue'])

    subset = output.loc[['Saving total (Billion euro)', 'Debt total (Billion euro)',
                         'Subsidies total (Billion euro)'], :].T
    subset.dropna(how='any', inplace=True)
    subset.columns = [c.split(' (Billion euro)')[0] for c in subset.columns]
    if not subset.empty:
        make_area_plot(subset, 'Financing (Billion euro)', save=os.path.join(path, 'financing.png'),
                       format_y=lambda y, _: '{:.0f}'.format(y), loc='left', left=1.2,
                       colors=['darkred', 'darkgrey', 'darkgreen'])

    subset = output.loc[['Saving insulation (Thousand euro/household)', 'Debt insulation (Thousand euro/household)',
                         'Subsidies insulation (Thousand euro/household)'], :].T
    subset.dropna(how='any', inplace=True)
    subset.columns = [c.split(' (Thousand euro/household)')[0] for c in subset.columns]
    if not subset.empty:
        make_area_plot(subset, 'Financing (Thousand euro per household)', save=os.path.join(path, 'financing_households.png'),
                       format_y=lambda y, _: '{:.0f}'.format(y), loc='left', left=1.2,
                       colors=['darkred', 'darkgrey', 'darkgreen'])

    # balance
    if False:
        subset = output.loc[['Balance Tenant private - {} (euro/year.household)'.format(i) for i in resources_data['index']['Income tenant']], :].T
        subset.dropna(how='any', inplace=True)
        subset.columns = resources_data['index']['Income tenant']
        if not subset.empty:
            make_plot(subset, 'Balance Tenant private (euro per year)',
                      save=os.path.join(path, 'balance_tenant.png'),
                      format_y=lambda y, _: '{:.0f}'.format(y),
                      colors=resources_data['colors'], ymin=None)

        subset = output.loc[['Balance Owner-occupied - {} (euro/year.household)'.format(i) for i in resources_data['index']['Income tenant']], :].T
        subset.dropna(how='any', inplace=True)
        subset.columns = resources_data['index']['Income tenant']
        if not subset.empty:
            make_plot(subset, 'Balance Owner occupied (euro per year)',
                      save=os.path.join(path, 'balance_owner.png'),
                      format_y=lambda y, _: '{:.0f}'.format(y),
                      colors=resources_data['colors'], ymin=None)


def plot_compare_scenarios(result, folder, quintiles=None, order_scenarios=None, reference='Reference', colors=None,
                           scenario_assessment=None, social_discount_rate=0.032):
    """Grouped scenarios output.

    Parameters
    ----------
    result: dict
    folder: str
    quintiles: bool, default None
    scenario_assessment: str, default None


    Returns
    -------

    """
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

    def details_graphs(data, v, inf, folder_img, colors=None, order_scenarios=None):

        order = None
        if inf.get('groupby') is not None:
            n = (v.split(' {}')[0] + '_by_' + inf['groupby'] + '.png').replace(' ', '_').lower()
            dict_data = grouped(data, [v.format(i) for i in resources_data['index'][inf['groupby']]])
            replace = {v.format(i): i for i in resources_data['index'][inf['groupby']]}
            dict_data = {replace[key]: item for key, item in dict_data.items()}
            dict_data = {k: i for k, i in dict_data.items() if not (i == 0).all().all()}
            order = [i for i in resources_data['index'][inf['groupby']] if i in dict_data.keys()]
        elif inf.get('variables') is not None:
            n = inf.get('name')
            dict_data = grouped(data, inf['variables'])
            dict_data = {k.split(' (')[0]: i for k, i in dict_data.items()}
            dict_data = {k: i for k, i in dict_data.items() if not (i == 0).all().all()}
        else:
            raise NotImplemented('groupby or variables must be defined')

        dict_data = {key: item.astype(float).interpolate(limit_area='inside') for key, item in dict_data.items()}

        n_columns = len(dict_data.keys())
        if inf.get('n_columns') is not None:
            n_columns = inf['n_columns']

        if inf.get('exogenous') is not None:
            for key in dict_data.keys():
                dict_data[key] = pd.concat((dict_data[key], inf['exogenous'][key]), axis=1)
                dict_data[key].sort_index(inplace=True)

        if order_scenarios is not None:
            for key in dict_data.keys():
                order_temp = [i for i in dict_data[key].columns if i not in order_scenarios] + \
                             [i for i in order_scenarios if i in dict_data[key].columns]
                dict_data[key] = dict_data[key].loc[:, order_temp]

        if dict_data:
            make_grouped_subplots(dict_data, format_y=inf.get('format_y'), n_columns=n_columns, save=os.path.join(folder_img, n),
                                  order=order, scatter=inf.get('scatter'), colors=colors)

    if quintiles:
        resources_data['index']['Income tenant'] = resources_data['quintiles']
        resources_data['index']['Income owner'] = resources_data['quintiles']

    if colors is None:
        colors = dict(zip(result.keys(), sns.color_palette(n_colors=len(result.keys()))))
    colors.update({'Historic': "#000000"}) #(0, 0, 0)
    colors_add = sns.color_palette("husl", 10)

    folder_img = os.path.join(folder, 'img')
    if not os.path.isdir(folder_img):
        os.mkdir(folder_img)

    # ini
    emission_ini = result.get(reference).loc['Emission (MtCO2)', :].iloc[0]
    consumption_ini = result.get(reference).loc['Consumption (TWh)', :].iloc[0]
    # consumption_pe_ini = result.get(reference).loc['Consumption PE (TWh)', :].iloc[0]
    energy_poverty_ini = result.get(reference).loc['Energy poverty (Million)', :].iloc[0]

    start = result.get(reference).columns[0]
    end = result.get(reference).columns[-1]

    # ----------------
    # graph distributive impact - cost for households

    # post-treatment
    for k in result.keys():
        subsidies = result[k].loc['Subsidies total (Billion euro)', :]
        annuities_subsidies = calculate_annuities(subsidies, lifetime=10, discount_rate=social_discount_rate)
        subsidies = annuities_subsidies.rolling(window=10, min_periods=1).sum()
        subsidies.fillna(0, inplace=True)

        taxes = result[k].loc['Taxes expenditure (Billion euro)', :]
        bill_rebates = 0
        if 'Bill rebate (Billion euro)' in result[k].index:
            bill_rebates = result[k].loc['Bill rebate (Billion euro)', :]

        government_spending = subsidies - taxes + bill_rebates

        stock = result[reference].loc['Stock (Million)', :]
        cost_public_households = (government_spending / stock) * 1e3
        result[k].loc['Cost public households (euro)', :] = cost_public_households

        #result[k].loc['Cost public households (euro)', :] = subsidies_households

        levels = ['Housing type', 'Occupancy status', 'Income tenant']
        idx = list(product(*[resources_data['index'][i] for i in levels]))
        for j in ['', ' insulation', ' heater']:
            idx_temp = ['Annuities{} {} - {} - {}'.format(j, i[0], i[1], i[2]) for i in idx]
            annuities = result[k].loc[idx_temp, :].fillna(0)
            annuities_cumulated = annuities.rolling(window=10, min_periods=1, axis=1).sum()
            annuities_cumulated.index = ['Annuities{} cumulated {} - {} - {} (euro)'.format(j, i[0], i[1], i[2]) for i in idx]
            result[k] = pd.concat((result[k], annuities_cumulated), axis=0)

        idx_temp = ['Annuities subsidies heater {} - {} - {}'.format(i[0], i[1], i[2]) for i in idx]
        temp = result[k].loc[idx_temp, :].fillna(0)
        temp = temp.rolling(window=10, min_periods=1, axis=1).sum()
        temp.index = ['Annuities subsidies heater cumulated {} - {} - {} (euro)'.format(i[0], i[1], i[2]) for i in
                      idx]
        result[k] = pd.concat((result[k], temp), axis=0)

        idx_temp = ['Annuities cost heater {} - {} - {}'.format(i[0], i[1], i[2]) for i in idx]
        temp = result[k].loc[idx_temp, :].fillna(0)
        temp = temp.rolling(window=10, min_periods=1, axis=1).sum()
        temp.index = ['Annuities cost heater cumulated {} - {} - {} (euro)'.format(i[0], i[1], i[2]) for i in
                      idx]
        result[k] = pd.concat((result[k], temp), axis=0)

        idx_temp = ['Annuities subsidies insulation {} - {} - {}'.format(i[0], i[1], i[2]) for i in idx]
        temp = result[k].loc[idx_temp, :].fillna(0)
        temp = temp.rolling(window=10, min_periods=1, axis=1).sum()
        temp.index = ['Annuities subsidies insulation cumulated {} - {} - {} (euro)'.format(i[0], i[1], i[2]) for i in
                      idx]
        result[k] = pd.concat((result[k], temp), axis=0)

        idx_temp = ['Annuities cost insulation {} - {} - {}'.format(i[0], i[1], i[2]) for i in idx]
        temp = result[k].loc[idx_temp, :].fillna(0)
        temp = temp.rolling(window=10, min_periods=1, axis=1).sum()
        temp.index = ['Annuities cost insulation cumulated {} - {} - {} (euro)'.format(i[0], i[1], i[2]) for i in
                      idx]
        result[k] = pd.concat((result[k], temp), axis=0)
        # result[k].to_csv(os.path.join(folder, 'output_{}.csv'.format(k)))

    # graph distributive impact - cost for households by scenario
    counterfactual = reference
    #scenario_assessment, counterfactual = 'BanNoPolicyHeater', 'NoPolicyHeater'
    if scenario_assessment is not None:
        levels = ['Housing type', 'Occupancy status', 'Income tenant']
        idx = list(product(*[resources_data['index'][i] for i in levels]))
        cost_insulation = ['Annuities cost insulation cumulated {} - {} - {} (euro)'.format(i[0], i[1], i[2]) for i in idx]
        subsidies_insulation = ['Annuities subsidies insulation cumulated {} - {} - {} (euro)'.format(i[0], i[1], i[2]) for i in idx]
        cost_heater = ['Annuities cost heater cumulated {} - {} - {} (euro)'.format(i[0], i[1], i[2]) for i in idx]
        subsidies_heater = ['Annuities subsidies heater cumulated {} - {} - {} (euro)'.format(i[0], i[1], i[2]) for i in idx]

        energy = ['Energy expenditures {} - {} - {} (euro)'.format(i[0], i[1], i[2]) for i in idx]
        stock = ['Stock {} - {} - {}'.format(i[0], i[1], i[2]) for i in idx]
        income = ['Income {} - {} - {} (euro)'.format(i[0], i[1], i[2]) for i in idx]

        idx = pd.MultiIndex.from_tuples(idx, names=levels)
        # average over time
        year = range(start, end, 1)
        year = [i for i in year if i in result.get(reference).columns]
        dict_cost_insulation, dict_subsidies_insulation, dict_cost_heater, dict_subsidies_heater, dict_energy, dict_taxes = {}, {}, {}, {}, {}, {}
        for k, i in result.items():
            temp = (i.loc[cost_insulation, year].set_axis(idx, axis=0)).sum(axis=1) / i.loc[stock, year].sum(axis=1).set_axis(idx, axis=0)
            dict_cost_insulation.update({k: temp.copy()})
            temp = - (i.loc[subsidies_insulation, year].set_axis(idx, axis=0)).sum(axis=1) / i.loc[stock, year].sum(axis=1).set_axis(idx, axis=0)
            dict_subsidies_insulation.update({k: temp.copy()})
            temp = (i.loc[cost_heater, year].set_axis(idx, axis=0)).sum(axis=1) / i.loc[stock, year].sum(axis=1).set_axis(idx, axis=0)
            dict_cost_heater.update({k: temp.copy()})
            temp = - (i.loc[subsidies_heater, year].set_axis(idx, axis=0)).sum(axis=1) / i.loc[stock, year].sum(axis=1).set_axis(idx, axis=0)
            dict_subsidies_heater.update({k: temp.copy()})
            temp = (i.loc[energy, year].set_axis(idx, axis=0)).sum(axis=1) / \
                   i.loc[stock, year].sum(axis=1).set_axis(idx, axis=0)
            dict_energy.update({k: temp.copy()})
            temp = pd.concat([i.loc['Cost public households (euro)', year]] * len(idx), keys=idx, axis=1).T
            temp = (temp * i.loc[stock, year].set_axis(idx, axis=0)).sum(axis=1) / i.loc[stock, year].sum(axis=1).set_axis(idx, axis=0)
            dict_taxes.update({k: temp.copy()})

        df_energy = pd.DataFrame(dict_energy)
        dict_cost_insulation = pd.DataFrame(dict_cost_insulation)
        dict_subsidies_insulation = pd.DataFrame(dict_subsidies_insulation)
        dict_cost_heater = pd.DataFrame(dict_cost_heater)
        dict_subsidies_heater = pd.DataFrame(dict_subsidies_heater)
        dict_taxes = pd.DataFrame(dict_taxes)
        df = pd.concat((dict_cost_insulation, dict_subsidies_insulation, dict_cost_heater, dict_subsidies_heater, df_energy, dict_taxes), axis=0,
                       keys=['Insulation', 'Subsidies insulation', 'Heater', 'Subsidies heater', 'Energy', 'Taxes'], names=['Type'])
        df.columns.names = ['Scenario']

        i = result[reference]
        df_income =  (i.loc[income, year].set_axis(idx, axis=0)).sum(axis=1) / i.loc[stock, year].sum(axis=1).set_axis(idx, axis=0)

        if not isinstance(scenario_assessment, list):
            scenario_assessment = [scenario_assessment]

        if len(scenario_assessment) == 1:
            df_diff = df.loc[:, scenario_assessment] - df.loc[:, counterfactual]
        else:
            df_diff = (df.loc[:, scenario_assessment].T - df.loc[:, counterfactual]).T

        # remove social housing from df_diff
        df_diff = df_diff.drop('Social-housing', level='Occupancy status')
        df_income = df_income.drop('Social-housing', level='Occupancy status')
        if not df_diff.empty:
            replace_legend = {'Insulation': 'Insulation cost',
                              'Subsidies insulation': 'Subsidies insulation',
                              'Heater': 'Heating system cost',
                              'Subsidies heater': 'Subsidies heating system',
                              'Taxes': 'Taxes to cover subsidies',
                              'Energy': 'Energy expenditures'}
            for s_assessment in scenario_assessment:
                temp = df_diff.loc[:, s_assessment].copy()
                # impact in euro
                make_stacked_bar_subplot(temp, format_y=lambda y, _: '{:.0f}€'.format(y),
                                         fonttick=18, color=resources_data['colors'],
                                         save=os.path.join(folder_img, 'cost_households_{}.png'.format(s_assessment)),
                                         subplot_groups=['Housing type', 'Occupancy status'],
                                         index_group='Income tenant', stack_group='Type',
                                         annotate='{:.0f}', annotate_bis=['Energy', 'Heater', 'Insulation'],
                                         replace_legend=replace_legend,
                                         figtitle='Households cost in {} compared to {}'.format(s_assessment, counterfactual))
                # impact in share of income
                make_stacked_bar_subplot(temp / df_income, format_y=lambda y, _: '{:.1%}'.format(y),
                                         fonttick=18, color=resources_data['colors'],
                                         save=os.path.join(folder_img, 'ratio_cost_households_{}.png'.format(s_assessment)),
                                         subplot_groups=['Housing type', 'Occupancy status'],
                                         index_group='Income tenant', stack_group='Type',
                                         annotate='{:.0%}', annotate_bis=['Energy', 'Heater', 'Insulation'],
                                         replace_legend=replace_legend,
                                         figtitle='Ratio households cost on income for in {} compared to {}'.format(s_assessment, counterfactual))
            # all scenarios same graphic in euro
            t = df_diff.groupby(['Housing type', 'Occupancy status', 'Income tenant']).sum()
            t = select(t, {'Occupancy status': ['Owner-occupied', 'Privately rented']})
            t = format_table(t, name='Scenario')
            t['Decision maker'] = t['Housing type'] + ' | ' + t['Occupancy status']
            make_relplot(t, x='Income tenant', y='Data', col='Decision maker', hue='Scenario',
                         palette=colors,
                         save=os.path.join(folder_img, 'cost_households_scenario_euro.png'),
                         title=None, format_y=lambda y, _: '{:.0f} €'.format(y))

            # all scenarios same graphic in share of income
            t = df_diff.groupby(['Housing type', 'Occupancy status', 'Income tenant']).sum()
            t = (t.T / df_income).T
            t = select(t, {'Occupancy status': ['Owner-occupied', 'Privately rented']})
            t = format_table(t, name='Scenario')
            t['Decision maker'] = t['Housing type'] + ' | ' + t['Occupancy status']
            make_relplot(t, x='Income tenant', y='Data', col='Decision maker', hue='Scenario',
                         palette=colors, format_y=lambda y, _: '{:.1%}'.format(y),
                         save=os.path.join(folder_img, 'cost_households_scenario_income.png'),
                         title=None)

            #

        if False:
            for n, g in df.groupby(levels_group):
                g = g.droplevel(levels_group, axis=0)
                if counterfactual is not None:
                    g = (g.T - g.loc[:, counterfactual]).T
                    g.drop(counterfactual, axis=1, inplace=True)
                g = g.stack('Scenario').unstack('Income tenant')
                # Creating a larger figure outside the function

                make_clusterstackedbar_plot(g, 'Type', colors=resources_data['colors'],
                                            format_y=lambda y, _: '{:.0f} €/year'.format(y),
                                            save=os.path.join(folder_img, 'cost_households_{}.png'.format('_'.join(n))), rotation=90,
                                            ymin=-60, ymax=60,
                                            legend=False, figtitle=' | '.join(n), display_total=True)

    # ----------------
    """
    data = select(df, {'Occupancy status': ['Owner-occupied', 'Privately rented']})
    data = format_table(data, name='Scenarios')
    data['Decision maker'] = data['Housing type'] + ' - ' + data['Occupancy status']

    make_relplot(data, x='Income tenant', y='Data', col='Decision maker', hue='Scenarios',
                 palette=colors,
                 save=os.path.join(folder_img, 'energy_income_ratio{}_{}.png'.format(k.replace(' ', '_'), year)),
                 title=None)
    """


    # make table summary
    vars = ['Stock (Million)', 'Surface (Million m2)', 'Consumption (TWh)', 'Consumption (kWh/m2)', 'Consumption PE (TWh)'] #
    vars += ['Consumption {} (TWh)'.format(i) for i in resources_data['index']['Energy']]
    vars += ['Energy poverty (Million)', 'Emission (MtCO2)']
    vars += ['Stock {} (Million)'.format(i) for i in resources_data['index']['Performance']]
    vars += ['Stock {} (Million)'.format(i) for i in resources_data['index']['Heating system']]
    vars += ['Carbon value (Billion euro)', 'Health cost (Billion euro)', 'Energy expenditures (Billion euro)']

    cum_vars = ['Emission (MtCO2)', 'Renovation (Thousand households)',
                'Investment insulation (Billion euro)', 'Subsidies insulation (Billion euro)',
                'Investment heater (Billion euro)', 'Subsidies heater (Billion euro)']
    avg_vars = ['Cumulated Renovation (Thousand households)',
                'Cumulated Investment insulation (Billion euro)',
                'Cumulated Subsidies insulation (Billion euro)',
                'Cumulated Investment heater (Billion euro)',
                'Cumulated Subsidies heater (Billion euro)']

    for yr in [i for i in [2030, 2050] if i in result.get(reference).columns]:
        summary = dict()
        for key, data in result.items():
            temp = data.loc[[v for v in vars if v in data.index], yr]

            t = data.loc[[v for v in cum_vars if v in data.index], :].sum(axis=1).rename(yr)
            t.index = t.index.map(lambda x: 'Cumulated {}'.format(x))
            temp = pd.concat((temp, t), axis=0)

            t = temp.loc[[v for v in avg_vars if v in temp.index]] / (yr - start)
            t.index = t.index.map(lambda x: x.replace('Cumulated', 'Annual average'))
            temp = pd.concat((temp, t), axis=0)

            temp['Consumption saving (%)'] = (consumption_ini - temp['Consumption (TWh)']) / consumption_ini
            # temp['Consumption PE saving (%)'] = (consumption_pe_ini - temp['Consumption PE (TWh)']) / consumption_pe_ini
            temp['Emission saving (%)'] = (emission_ini - temp['Emission (MtCO2)']) / emission_ini
            temp['Energy poverty reduction (%)'] = (energy_poverty_ini - temp['Energy poverty (Million)']) / energy_poverty_ini
            summary.update({key: temp})

        summary = pd.concat(summary, axis=1)
        if order_scenarios is not None:
            summary = summary.loc[:, order_scenarios]
        summary.to_csv(os.path.join(folder, 'summary_{}.csv'.format(yr)))

    # calculate cumulated annuities
    lifetime_insulation, lifetime_heater = 35, 20
    for scenario in result.keys():
        annuities_insulation = result.get(scenario).loc[['CBA Annuities insulation (Billion euro)'], :].T.dropna()
        annuities_insulation = annuities_insulation.rolling(window=lifetime_insulation, min_periods=1).sum()
        annuities_insulation = annuities_insulation.rename(
            columns={'CBA Annuities insulation (Billion euro)': 'CBA Annuities cumulated insulation (Billion euro)'})
        result[scenario] = pd.concat((result[scenario], annuities_insulation.T), axis=0)

        annuities_heater = result.get(scenario).loc[['CBA Annuities heater (Billion euro)'], :].T.dropna()
        annuities_heater = annuities_heater.rolling(window=lifetime_heater, min_periods=1).sum()
        annuities_heater = annuities_heater.rename(
            columns={'CBA Annuities heater (Billion euro)': 'CBA Annuities cumulated heater (Billion euro)'})
        result[scenario] = pd.concat((result[scenario], annuities_heater.T), axis=0)
    # ----------------

    # graph clusterstackedbar
    variables = [('Stock {} (Million)', 'Heater', 'M'),
                 ('Stock {} (Million)', 'Performance', 'M'),
                 ('Consumption {} (TWh)', 'Heater', 'TWh'),
                 ('Consumption {} (TWh)', 'Performance', 'TWh')]
    for v in variables:
        name = v[0].split(' {}')[0].lower()
        groupby = v[1]
        unit = v[2]
        v = v[0]
        years = list({start, 2030, end})
        years = [i for i in years if (i >= start) and (i <= end)]
        years.sort()
        temp = grouped(result, [v.format(i) for i in resources_data['index'][groupby]])
        temp = {k: i.loc[years, :] for k, i in temp.items()}
        replace = {v.format(i): i for i in resources_data['index'][groupby]}
        temp = {replace[key]: item for key, item in temp.items()}
        temp = pd.concat(temp).rename_axis([groupby, 'Years'], axis=0).rename_axis('Scenario', axis=1)
        temp = temp.stack('Scenario').unstack('Years')
        if not temp.empty:
            if len(temp.columns) > 1:
                make_clusterstackedbar_plot(temp, groupby, colors=resources_data['colors'],
                                            format_y=lambda y, _: '{:.0f} {}'.format(y, unit),
                                            save=os.path.join(folder_img, '{}_{}.png'.format(name, groupby.lower())),
                                            rotation=90, year_ini=start, order_scenarios=order_scenarios,
                                            fonttick=20)
    # ----------------

    # graph policies
    policies = []
    for _, df in result.items():
        policies += [i.split(' Single-family (Thousand households)')[0] for i in df.index if ('Single-family (Thousand households)' in i) and
                    ('insulation' not in i) and ('heater' not in i) and ('Renovation' not in i)]
    policies = list(set(policies))
    policies = [i for i in policies if 'Over cap' not in i]

    temp = grouped(result, ['{} (Billion euro)'.format(i) for i in policies])
    years = list({start + 1, 2030, end})
    years = [i for i in years if (i >= start) and (i <= end)]
    years.sort()
    temp = {k: i.loc[years, :] for k, i in temp.items()}
    temp = pd.concat(temp).rename_axis(['Policy', 'Years'], axis=0).rename_axis('Scenario', axis=1)
    temp = temp.stack('Scenario').unstack('Years')
    # remove policies with nan for a single year
    yrs = [i for i in years if i != start + 1]
    if all(item in temp.columns for item in yrs):
        to_drop = temp.loc[:, yrs]
        to_drop = to_drop.unstack('Scenario').stack('Years').T
        to_drop = to_drop.loc[to_drop.isna().all(axis=1), :].index
        temp.drop(to_drop, axis=0, level='Scenario', inplace=True)
        # modifiy index level 'Policy' to remove ' (Billion euro)'
        temp.index = temp.index.set_levels([i.split(' (Billion euro)')[0] for i in temp.index.levels[0]], level=0)
        if not temp.empty:
            if len(temp.columns) > 1:
                temp.fillna(0, inplace=True)
                order = None
                if order_scenarios is not None:
                    order = [i for i in order_scenarios if i in temp.index.get_level_values('Scenario')]

                make_clusterstackedbar_plot(temp, 'Policy',
                                            format_y=lambda y, _: '{:.0f} B€'.format(y),
                                            save=os.path.join(folder_img, 'policy_scenario_detailed.png'),
                                            rotation=90, year_ini=start + 1,
                                            order_scenarios=order,
                                            fonttick=20)

                agg = {'Mpr': 'Subsidy', 'Mpr multifamily': 'Subsidy', 'Mpr multifamily deep': 'Subsidy',
                       'Mpr multifamily updated': 'Subsidy',
                       'Mpr serenite': 'Subsidy', 'Mpr efficacite': 'Subsidy', 'Mpr performance': 'Subsidy',
                       'Cite': 'Subsidy'}
                # replace index with aggregated

                temp = temp.rename(index=agg).groupby(temp.index.names).sum()
                rename = {'Subsidy': 'Direct subsidies',
                          'Cee': 'White certificate',
                          'Reduced vat': 'Reduced VAT',
                          'Zero interest': 'Zero interest loan'}
                temp = temp.rename(index=rename)

                # ['Cee', 'Subsidy', 'Reduced vat', 'Zero interest loan']
                make_clusterstackedbar_plot(temp, 'Policy',
                                            format_y=lambda y, _: '{:.0f} B€'.format(y),
                                            save=os.path.join(folder_img, 'policy_scenario_aggregated.png'),
                                            rotation=90, year_ini=start + 1,
                                            order_scenarios=order,
                                            colors=resources_data['colors'],
                                            fonttick=20)
    # ----------------

    # graph emission saving
    variables = {
        'Emission saving heater (MtCO2/year)': 'Switch heater',
        'Emission saving insulation (MtCO2/year)': 'Insulation',
        'Emission saving prices (MtCO2/year)': 'Prices induced',
        'Emission saving carbon content (MtCO2/year)': 'Carbon content',
        'Emission saving natural replacement (MtCO2/year)': 'Natural replacement'
    }
    colors_temp = {'Switch heater': '#e76f51',
                   'Insulation': '#f4a261',
                   'Prices induced': '#e9c46a',
                   'Natural replacement': '#2a9d8f',
                   'Carbon content': '#264653'
                   }
    df = pd.DataFrame({k: i.loc[variables.keys(), :].sum(axis=1) for k, i in result.items()}).round(3)
    df = df.rename(index=variables)
    df /= emission_ini
    if order_scenarios is not None:
        df = df.loc[:, order_scenarios]
    make_stackedbar_plot(df.T, 'Emission saving to {} (% of initial emission {} - {:.0f} MtCO2)'.format(end, start, emission_ini),
                         ncol=2, ymin=None, format_y=lambda y, _: '{:.0%}'.format(y),
                         colors=colors_temp, save=os.path.join(folder_img, 'emission_saving_decomposition.png'),
                         rotation=90, left=1.2, fontxtick=20)

    # consumption saving
    variables = {
        'CBA Consumption saving EE (TWh)': 'Energy efficiency',
        'CBA Consumption saving prices (TWh)': 'Prices induced',
        'Consumption saving natural replacement (TWh/year)': 'Natural replacement',
        'CBA Rebound EE (TWh)': 'Rebound effect'
    }
    colors_temp = {
        'Energy efficiency': '#f4a261',
        'Prices induced': '#e9c46a',
        'Natural replacement': '#2a9d8f',
        'Rebound effect': '#000000'
    }
    df = pd.DataFrame({k: i.loc[variables.keys(), :].sum(axis=1) for k, i in result.items()}).round(3)
    df = df.rename(index=variables)
    df /= consumption_ini
    if order_scenarios is not None:
        df = df.loc[:, order_scenarios]
    make_stackedbar_plot(df.T, 'Consumption saving to {} (% of initial emission {} - {:.0f} TWh)'.format(end, start, consumption_ini),
                         ncol=2,
                         ymin=None, format_y=lambda y, _: '{:.0%}'.format(y),
                         colors=colors_temp, save=os.path.join(folder_img, 'consumption_saving_decomposition.png'),
                         rotation=90, left=1.2, fontxtick=20)

    emission_saving = pd.Series([result[s].loc['Emission (MtCO2)', start] - result[s].loc['Emission (MtCO2)', end] for s in result.keys()],
                                index=result.keys()) / emission_ini
    consumption_saving = pd.Series([result[s].loc['Consumption (TWh)', start] - result[s].loc['Consumption (TWh)', end] for s in result.keys()],
                                      index=result.keys()) / consumption_ini

    # ----------------
    # graph cost benefit analysis
    try:
        energy_poverty = pd.Series({k: i.loc['Energy poverty (Million)', end] for k, i in result.items()})
        npv = pd.Series({k: i.loc['NPV (Billion Euro)', end] for k, i in result.items()})
        df = pd.concat((consumption_saving, emission_saving, npv, pd.Series(colors), energy_poverty), axis=1,
                       keys=['Consumption saving (TWh)',
                             'Emission saving (MtCO2)',
                             'CBA diff (Billion euro)',
                             'colors',
                             'Energy poverty (Million)'
                             ])
        df.dropna(inplace=True)

        make_scatter_plot(df, 'CBA diff (Billion euro)', 'Consumption saving (TWh)',
                          'Cost benefit analysis to {} (Billion euro)'.format(end),
                          'Consumption saving to {} (%)'.format(end),
                          hlines=0,
                          format_x=lambda y, _: '{:.1f}'.format(y), ymin=0,
                          format_y=lambda x, _: '{:.0%}'.format(x),
                          save=os.path.join(folder_img, 'cba_consumption.png'),
                          col_colors='colors',
                          col_size='Energy poverty (Million)'
                          )

        make_scatter_plot(df, 'CBA diff (Billion euro)', 'Emission saving (MtCO2)',
                          'Cost benefit analysis to {} (Billion euro)'.format(end),
                          'Emission saving to {} (%)'.format(end),
                          hlines=0,
                          format_x=lambda y, _: '{:.1f}'.format(y), ymin=0,
                          format_y=lambda x, _: '{:.0%}'.format(x),
                          save=os.path.join(folder_img, 'cba_emission.png'),
                          col_colors='colors',
                          col_size='Energy poverty (Million)'
                          )
    except:
        pass

    # graph scatter plot - cba/runnning cost
    try:
        if False:
            subsidies_total = pd.Series(
                {k: i.loc['Subsidies total (Billion euro)', :].sum() for k, i in result.items()})
            variables = {'Cost energy (Billion euro)': 'Energy expenditure',
                         'Loss thermal comfort (Billion euro)': 'Comfort',
                         'Cost emission (Billion euro)': 'Direct emission',
                         'Cost heath (Billion euro)': 'Health cost',
                         'Cost heater (Billion euro)': 'Annuities heater',
                         'Cost insulation (Billion euro)': 'Annuities insulation',
                         'COFP (Billion euro)': 'COFP'
                         }
            # graph running cost
            df = pd.DataFrame({k: i.loc[variables.keys(), :].sum(axis=1) for k, i in result.items()}).round(3)
            df = df.rename(index=variables)
            cost_total = df.sum(axis=0).rename('Total')
            make_stackedbar_plot(df.T, 'Running cost to {} (Billion euro)'.format(end), ncol=3, ymin=None,
                                 format_y=lambda y, _: '{:.0f}'.format(y),
                                 hline=0, scatterplot=cost_total, colors=resources_data['colors'],
                                 save=os.path.join(folder_img, 'running_cost.png'),
                                 rotation=0, left=1.3)

            # colors
            diff = (df.T - df[reference]).T
            if not diff.empty and diff.shape[1] > 1:
                cost_diff_total = diff.T.sum(axis=1).rename('Total')
                make_stackedbar_plot(diff.drop(reference, axis=1).T,
                                     'Running cost compare to Reference to {} (Billion euro)'.format(end), ncol=3, ymin=None,
                                     format_y=lambda y, _: '{:.0f}'.format(y),
                                     hline=0, scatterplot=cost_diff_total.drop(reference), colors=resources_data['colors'],
                                     save=os.path.join(folder_img, 'running_cost_comparison.png'), rotation=0,
                                     left=1.3)

                df = pd.concat((consumption_saving, emission_saving, cost_total, cost_diff_total, pd.Series(colors), subsidies_total), axis=1,
                               keys=['Consumption saving (TWh)',
                                     'Emission saving (MtCO2)',
                                     'Running cost (Billion euro)',
                                     'Running cost diff (Billion euro)',
                                     'colors',
                                     'Subsidies (Billion euro)'
                                     ])
                df.dropna(inplace=True)

                make_scatter_plot(df, 'Consumption saving (TWh)', 'Running cost diff (Billion euro)',
                                  'Consumption saving to {} (TWh)'.format(end),
                                  'Running cost to {} (Billion euro)'.format(end),
                                  hlines=0,
                                  format_x=lambda x, _: '{:.0%}'.format(x), xmin=0,
                                  format_y=lambda y, _: '{:.1f}'.format(y),
                                  save=os.path.join(folder_img, 'running_cost_consumption.png'),
                                  col_colors='colors',
                                  col_size='Subsidies (Billion euro)'
                                  )

                make_scatter_plot(df, 'Emission saving (MtCO2)', 'Running cost diff (Billion euro)',
                                  'Emission saving to {}(MtCO2)'.format(end),
                                  'Running cost to {} (Billion euro)'.format(end),
                                  hlines=0,
                                  format_x=lambda x, _: '{:.0%}'.format(x), xmin=0,
                                  format_y=lambda y, _: '{:.1f}'.format(y),
                                  save=os.path.join(folder_img, 'running_cost_emission.png'),
                                  col_colors='colors',
                                  col_size='Subsidies (Billion euro)'
                                  )

        if False:
            # graph Annualized CBA
            variables = {'CBA Consumption saving EE (Billion euro)': 'Saving EE',
                         'CBA Consumption saving prices (Billion euro)': 'Saving price',
                         'CBA Thermal comfort EE (Billion euro)': 'Comfort EE',
                         'CBA Emission direct (Billion euro)': 'Direct emission',
                         'CBA Thermal loss prices (Billion euro)': 'Comfort prices',
                         'CBA Annuities heater (Billion euro)': 'Annuities heater',
                         'CBA Annuities insulation (Billion euro)': 'Annuities insulation',
                         'CBA Carbon Emission indirect (Billion euro)': 'Indirect emission',
                         'CBA Health cost (Billion euro)': 'Health cost',
                         'CBA COFP (Billion euro)': 'COFP'
                         }

            df = pd.DataFrame({k: i.loc[variables.keys(), :].sum(axis=1) for k, i in result.items()}).round(3)
            df = df.rename(index=variables)
            cba_total = df.sum(axis=0).rename('Total')
            make_stackedbar_plot(df.T, 'Cost-benefits analysis (Billion euro)', ncol=3, ymin=None,
                                 format_y=lambda y, _: '{:.0f}'.format(y),
                                 hline=0, scatterplot=cba_total, colors=resources_data['colors'],
                                 save=os.path.join(folder_img, 'cost_benefit_analysis.png'),
                                 rotation=90, left=1.3)

            diff = (df.T - df[reference]).T
            if not diff.empty and diff.shape[1] > 1:
                cba_diff_total = diff.T.sum(axis=1).rename('Total')

                make_stackedbar_plot(diff.drop(reference, axis=1).T,
                                     'Cost-benefits analysis compare to Reference to {} (Billion euro)'.format(end),
                                     ncol=3, ymin=None,
                                     format_y=lambda y, _: '{:.0f}'.format(y),
                                     hline=0, scatterplot=cba_diff_total.drop(reference), colors=resources_data['colors'],
                                     save=os.path.join(folder_img, 'cost_benefit_analysis_comparison.png'), rotation=0,
                                     left=1.3)


                energy_poverty = pd.Series({k: i.loc['Energy poverty (Million)', end] for k, i in result.items()})
                df = pd.concat((consumption_saving, emission_saving, cba_total, cba_diff_total, pd.Series(colors),
                                subsidies_total, energy_poverty), axis=1,
                               keys=['Consumption saving (TWh)',
                                     'Emission saving (MtCO2)',
                                     'CBA (Billion euro)',
                                     'CBA diff (Billion euro)',
                                     'colors',
                                     'Subsidies (Billion euro)',
                                     'Energy poverty (Million)'
                                     ])
                df.dropna(inplace=True)

                make_scatter_plot(df, 'CBA diff (Billion euro)', 'Consumption saving (TWh)',
                                  'Cost benefit analysis to {} (Billion euro)'.format(end),
                                  'Consumption saving to {} (TWh)'.format(end),
                                  hlines=0,
                                  format_x=lambda y, _: '{:.1f}'.format(y), ymin=0,
                                  format_y=lambda x, _: '{:.0%}'.format(x),
                                  save=os.path.join(folder_img, 'cba_annualized_consumption.png'),
                                  col_colors='colors',
                                  col_size='Energy poverty (Million)'
                                  )

                make_scatter_plot(df, 'CBA diff (Billion euro)', 'Emission saving (MtCO2)',
                                  'Cost benefit analysis to {} (Billion euro)'.format(end),
                                  'Emission saving to {} (MtCO2)'.format(end),
                                  hlines=0,
                                  format_x=lambda y, _: '{:.1f}'.format(y), ymin=0,
                                  format_y=lambda x, _: '{:.0%}'.format(x),
                                  save=os.path.join(folder_img, 'cba_annualized_emission.png'),
                                  col_colors='colors',
                                  col_size='Subsidies (Billion euro)'
                                  )
    except KeyError:
        pass



    # graph distributive impact - cost for households
    if False:
        data = pd.concat(result).rename_axis(['Scenario', 'Variable'], axis=0).rename_axis('Years', axis=1).unstack('Scenario')
        levels = ['Housing type', 'Occupancy status', 'Income tenant']
        idx = list(product(*[resources_data['index'][i] for i in levels]))

        stock = ['Stock {} - {} - {}'.format(i[0], i[1], i[2]) for i in idx]
        stock = data.loc[stock, :].set_axis(idx, axis=0).rename_axis('Household', axis=0)

        cost = ['Annuities cumulated {} - {} - {} (euro)'.format(i[0], i[1], i[2]) for i in idx]
        cost = data.loc[cost, :].set_axis(idx, axis=0).rename_axis('Household', axis=0)
        cost_avg = cost / stock

        energy = ['Energy expenditures {} - {} - {} (euro)'.format(i[0], i[1], i[2]) for i in idx]
        energy = data.loc[energy, :].set_axis(idx, axis=0).rename_axis('Household', axis=0)
        energy_avg = energy / stock

        df = pd.concat((cost_avg, energy_avg), axis=0, keys=['Cost', 'Energy'], names=['Type'])
        df = df.stack('Scenario')

        years = [2018, 2030, 2050]
        df = df.loc[:, [i for i in years if i in df.columns]]

        groupby = 'Type'
        name = 'cost_households_owner'
        temp = df.xs(('Single-family', 'Owner-occupied', 'C1'), level='Household').copy()
        temp.dropna(how='all', inplace=True, axis=1)
        if not temp.empty:
            if len(temp.columns) > 1:
                make_clusterstackedbar_plot(temp, groupby, colors=resources_data['colors'],
                                            format_y=lambda y, _: '{:.0f}'.format(y),
                                            save=os.path.join(folder_img, '{}_{}.png'.format(name, groupby.lower())),
                                            rotation=90, year_ini=2018)
        temp = df.xs(('Single-family', 'Privately rented', 'C1'), level='Household').copy()
        name = 'cost_households_renter'
        temp.dropna(how='all', inplace=True, axis=1)
        if not temp.empty:
            if len(temp.columns) > 1:
                make_clusterstackedbar_plot(temp, groupby, colors=resources_data['colors'],
                                            format_y=lambda y, _: '{:.0f}'.format(y),
                                            save=os.path.join(folder_img, '{}_{}.png'.format(name, groupby.lower())),
                                            rotation=90, year_ini=2018)

    # graph distributive impact - Calculate cumulated annuities
    if False:
        levels = ['Housing type', 'Occupancy status', 'Income tenant']
        idx = list(product(*[resources_data['index'][i] for i in levels]))

        years = [2020, 2030, 2040, 2050, max(result[reference].columns)]
        years = list(set(years))
        years = [y for y in years if y in result[reference].columns]
        years.sort()

        for k in ['', ' std']:
            ratio = ['Ratio expenditure{} {} - {} - {} (%)'.format(k, i[0], i[1], i[2]) for i in idx]

            idx = pd.MultiIndex.from_tuples(idx, names=levels)
            dict_rslt = {k: i.loc[ratio, :].set_axis(idx, axis=0).dropna(axis=1, how='all') for k, i in result.items()}

            start = min(dict_rslt[reference].columns)
            ini = pd.DataFrame({k: i.loc[:, start] for k, i in dict_rslt.items() if start in i.columns})

            for year in years:
                df = pd.DataFrame({k: i.loc[:, year] for k, i in dict_rslt.items() if year in i.columns})

                data = select(df, {'Occupancy status': ['Owner-occupied', 'Privately rented']})
                data = format_table(data, name='Scenarios')
                data['Decision maker'] = data['Housing type'] + ' - ' + data['Occupancy status']

                make_relplot(data, x='Income tenant', y='Data', col='Decision maker', hue='Scenarios',
                             palette=colors,
                             save=os.path.join(folder_img, 'energy_income_ratio{}_{}.png'.format(k.replace(' ', '_'), year)),
                             title=None)
                #title = 'Energy expenditure{} on income ratio\n{}'.format(k, year)

                # > 0 positive means households are loosing money compare to ref
                if df.shape[1] > 1:
                    rate = ((df.T - df.loc[:, reference]) / df.loc[:, reference]).T
                    rate = rate.loc[:, [i for i in rate.columns if i != reference]]
                    rate = select(rate, {'Occupancy status': ['Owner-occupied', 'Privately rented']})
                    rate = format_table(rate, name='Scenarios')
                    rate['Decision maker'] = rate['Housing type'] + ' - ' + rate['Occupancy status']

                    make_relplot(rate, x='Income tenant', y='Data', col='Decision maker', hue='Scenarios',
                                 palette=colors,
                                 save=os.path.join(folder_img, 'energy_income_ratio_rate{}_{}.png'.format(k.replace(' ', '_'), year)),
                                 title=None)
                   # title = 'Energy expenditure{} on income ratio\n{} compare to Reference'.format(k, year)

                # compare to initial
                if False:
                    # > 0 positive means households are loosing money compare to ini
                    diff = (df - ini) / ini
                    diff = select(diff, {'Occupancy status': ['Owner-occupied', 'Privately rented']})
                    diff = format_table(diff, name='Scenarios')
                    diff['Decision maker'] = diff['Housing type'] + ' - ' + diff['Occupancy status']

                    make_relplot(diff, x='Income tenant', y='Data', col='Decision maker', hue='Scenarios',
                                 palette=colors,
                                 save=os.path.join(folder_img, 'energy_income_ratio_ini{}_{}.png'.format(k.replace(' ', '_'), year)),
                                 title=None)
                    #title = 'Energy expenditure{} on income ratio\n{} compare to {}'.format(k, year, start)

    # graph line plot 2D comparison
    if 'consumption_total_hist' in resources_data.keys():
        consumption_total_hist = resources_data['consumption_total_hist']
    else:
        consumption_total_hist = pd.DataFrame(resources_data['consumption_hist'])
        consumption_total_hist.index = consumption_total_hist.index.astype(int)
        if result[reference].loc['Consumption Heating (TWh)', :].iloc[0] == 0:
            consumption_total_hist.drop('Heating', axis=1, inplace=True)
        consumption_total_hist = consumption_total_hist.sum(axis=1).rename('Historic')

    variables = {'Consumption (TWh)': {'name': 'consumption.png',
                                       'format_y': lambda y, _: '{:,.0f} TWh'.format(y),
                                       'y_label': 'Final energy consumption for space heating (TWh)'
                                       },
                 'Consumption PE (TWh)': {'name': 'consumption_pe.png',
                                          'format_y': lambda y, _: '{:,.0f} TWh'.format(y),
                                          'y_label': 'Primary energy consumption for space heating (TWh)'
                                          },
                 'Consumption standard (TWh)': {'name': 'consumption_standard.png',
                                                'format_y': lambda y, _: '{:,.0f} TWh'.format(y)},
                 'Heating intensity (%)': {'name': 'heating_intensity.png',
                                           'format_y': lambda y, _: '{:,.0%}'.format(y)},
                 'Emission (MtCO2)': {'name': 'emission.png',
                                      'format_y': lambda y, _: '{:,.0f} MtCO2'.format(y),
                                      'y_label': 'GHG emission (MtCO2)'
                                      },
                 'Stock Heat pump (Million)': {'name': 'stock_heat_pump.png',
                                               'format_y': lambda y, _: '{:,.1f} M'.format(y)},
                 'Energy poverty (Million)': {'name': 'energy_poverty.png',
                                              'format_y': lambda y, _: '{:,.1f} M'.format(y)},
                 'Retrofit (Thousand households)': {'name': 'retrofit.png',
                                                    'format_y': lambda y, _: '{:,.0f} k'.format(y)},
                 'Renovation (Thousand households)': {'name': 'renovation.png',
                                                      'format_y': lambda y, _: '{:,.0f} k'.format(y)},
                 'Investment total (Thousand euro/household)': {'name': 'investment_households.png',
                                                                'format_y': lambda y, _: '{:,.0f}'.format(y)},
                 'Consumption saving insulation (TWh/year)': {'name': 'saving_insulation.png',
                                                              'format_y': lambda y, _: '{:,.1f}'.format(y)},
                 'Consumption saving heater (TWh/year)': {'name': 'saving_heater.png',
                                                          'format_y': lambda y, _: '{:,.1f}'.format(y)},
                 'Retrofit at least 1 EPC (Thousand households)':
                     {'name': 'retrofit_jump_comparison.png',
                      'format_y': lambda y, _: '{:,.0f}'.format(y),
                      'exogenous': resources_data['retrofit_comparison']},
                 'Investment total (Billion euro)': {'name': 'investment_total.png',
                                                     'format_y': lambda y, _: '{:,.0f} B€'.format(y)},
                 'Subsidies total (Billion euro)': {'name': 'subsidies_total.png',
                                                    'format_y': lambda y, _: '{:,.0f} B€'.format(y)},
                 'Energy expenditures (Billion euro)': {
                     'name': 'energy_expenditures.png',
                     'format_y': lambda y, _: '{:,.0f}'.format(y)},
                 'Health cost (Billion euro)': {'name': 'health_cost.png',
                                                'format_y': lambda y, _: '{:,.0f}'.format(y)},
                 'Replacement total (Thousand)': {'name': 'replacement_total.png',
                                                  'format_y': lambda y, _: '{:,.0f}'.format(y)},
                 'Replacement insulation (Thousand)': {'name': 'replacement_insulation.png',
                                                       'format_y': lambda y, _: '{:,.0f}'.format(y)},
                 'Switch Heat pump (Thousand households)': {'name': 'switch_heat_pump.png',
                                                         'format_y': lambda y, _: '{:,.0f}'.format(y)},
                 'Efficiency insulation (euro/kWh standard)': {'name': 'efficiency_insulation.png',
                                                               'format_y': lambda y, _: '{:,.2f}'.format(y)},
                 'Efficiency subsidies insulation (euro/kWh standard)': {'name': 'efficiency_subsidies_insulation.png',
                                                                         'format_y': lambda y, _: '{:,.3f}'.format(y)},
                 'Consumption standard saving insulation (%)': {'name': 'consumption_saving_insulation.png',
                                                                'format_y': lambda y, _: '{:,.1%}'.format(y)},
                 }

    # graph line subplot comparison
    for variable, infos in variables.items():
        temp = pd.DataFrame({scenario: output.loc[variable, :] for scenario, output in result.items()}).astype(float)
        temp = temp.interpolate(limit_area='inside')

        if infos.get('exogenous') is not None:
            temp = pd.concat((temp, infos['exogenous']), axis=1).astype(float)
            temp.sort_index(inplace=True)

        y_label = variable
        if infos.get('y_label') is not None:
            y_label = infos['y_label']

        scatter = infos.get('scatter')

        colors_temp = colors
        t = [i for i in temp.columns if i not in colors_temp.keys()]
        if t:
            colors_temp.update({i: colors_add[k] for k, i in enumerate(t)})
        if order_scenarios is not None:
            order_temp = [i for i in temp.columns if i not in order_scenarios] + [i for i in order_scenarios if i in temp.columns]
            temp = temp.loc[:, order_temp]
        make_plot(temp, y_label, save=os.path.join(folder_img, '{}'.format(infos['name'])), format_y=infos['format_y'],
                  scatter=scatter, colors=colors, loc='left', left=1.2)

    variables_output = {
        'Consumption {} (TWh)': [
            {'groupby': 'Energy',
             'format_y': lambda y, _: '{:,.0f}'.format(y),
             'n_columns': 2,
             'exogenous': resources_data['consumption_hist']}],
        'Stock {} (Million)': [{'groupby': 'Performance',
                                'format_y': lambda y, _: '{:,.0f}'.format(y)},
                               {'groupby': 'Heater',
                                'format_y': lambda y, _: '{:,.0f}'.format(y)}],
        'Investment {} (Billion euro)': [{
            'groupby': 'Insulation',
            'format_y': lambda y, _: '{:,.1f}'.format(y),
            'n_columns': 4
        }],
        'Replacement {} (Thousand households)': [{
            'groupby': 'Insulation',
            'format_y': lambda y, _: '{:,.1f}'.format(y),
            'n_columns': 4
        }],
        'Retrofit measures {} (Thousand households)': [{
            'groupby': 'Count',
            'format_y': lambda y, _: '{:,.0f}'.format(y),
            'n_columns': 5}],
        'Renovation {} (Thousand households)': [{
            'groupby': 'Decision maker',
            'format_y': lambda y, _: '{:,.0f}'.format(y),
            'n_columns': 2}],
        'Rate {} (%)': [{
            'groupby': 'Decision maker',
            'format_y': lambda y, _: '{:,.0%}'.format(y),
            'n_columns': 2}],
        'Rate Single-family - Owner-occupied {} (%)': [{
            'groupby': 'Income owner',
            'format_y': lambda y, _: '{:,.1%}'.format(y),
            'n_columns': 5},
            {
                'groupby': 'Performance',
                'format_y': lambda y, _: '{:,.1%}'.format(y),
                'n_columns': 5},
        ],
        'Retrofit': [{
            'variables': ['Switch decarbonize (Thousand households)',
                          'Insulation (Thousand households)',
                          'Insulation and switch decarbonize (Thousand households)'
                          ],
            'name': 'retrofit_decarbonize_options.png',
            'format_y': lambda y, _: '{:,.0f}'.format(y),
            'n_columns': 3
        }],
        'Financing': [{
            'variables': ['Subsidies total (Billion euro)',
                          'Debt total (Billion euro)',
                          'Saving total (Billion euro)'
                          ],
            'name': 'retrofit_financing.png',
            'format_y': lambda y, _: '{:,.0f}'.format(y),
            'n_columns': 3
        }],
        'Share subsidies {} (%)': [{
            'groupby': 'Income owner',
            'format_y': lambda y, _: '{:,.1%}'.format(y),
            'n_columns': 5
        }]
    }

    for var, infos in variables_output.items():
        for info in infos:
            details_graphs(result, var, info, folder_img, colors=colors, order_scenarios=order_scenarios)

    # TODO: uncertainty plot to work on
    if False:
        if reference in result.keys() and len(result.keys()) > 1 and config_sensitivity is not None:
            variables = {'Consumption (TWh)': ('consumption_hist_uncertainty.png', lambda y, _: '{:,.0f}'.format(y),
                                               resources_data['consumption_total_hist'],
                                               resources_data['consumption_total_objectives']),
                         'Emission (MtCO2)': ('emission_uncertainty.png', lambda y, _: '{:,.0f}'.format(y)),

                         }
            for variable, infos in variables.items():
                temp = pd.DataFrame({scenario: output.loc[variable, :] for scenario, output in result.items()})
                columns = temp.columns
                try:
                    temp = pd.concat((temp, infos[2]), axis=1)
                    temp.sort_index(inplace=True)
                except IndexError:
                    pass

                try:
                    scatter = infos[3]
                except IndexError:
                    scatter = None
                    pass

                make_uncertainty_plot(temp, variable, save=os.path.join(folder_img, '{}'.format(infos[0])), format_y=infos[1],
                                      scatter=scatter, columns=columns)


def plot_compare_scenarios_simple(result, folder, quintiles=None, reference='Reference'):
    """Plot the results of the scenarios.

    Parameters
    ----------
    result: dict
    folder: str
    quintiles: bool, default None

    Returns
    -------
    None
    """

    if quintiles:
        resources_data['index']['Income tenant'] = resources_data['quintiles']
        resources_data['index']['Income owner'] = resources_data['quintiles']

    folder_img = os.path.join(folder, 'img')
    if not os.path.isdir(folder_img):
        os.mkdir(folder_img)

    # ini
    end = sorted(result.get(reference).columns)[-1]

    emission_ini = result.get(reference).loc['Emission (MtCO2)', :].iloc[0]
    consumption_ini = result.get(reference).loc['Consumption (TWh)', :].iloc[0]

    emission_saving = pd.Series({k: i.loc['Emission saving (MtCO2/year)', :].sum() for k, i in result.items()}).round(3)
    emission_saving_percent = emission_saving / emission_ini
    emission_saving_percent.rename('Emission saving (%)')

    consumption_saving = pd.Series({k: i.loc['Consumption saving (TWh/year)', :].sum() for k, i in result.items()}).round(3)
    consumption_saving_percent = consumption_saving / consumption_ini
    consumption_saving_percent.rename('Consumption saving (%)')

    subsidies_total = pd.Series({k: i.loc['Subsidies total (Billion euro)', :].sum() for k, i in result.items()})
    subsidies_loan_total = pd.Series({k: i.loc['Subsidies loan total (Billion euro)', :].sum() for k, i in result.items()})
    subsidies_total += subsidies_loan_total

    investment_total = pd.Series({k: i.loc['Investment total WT (Billion euro)', :].sum() for k, i in result.items()})
    emission = pd.Series({k: i.loc['Emission (MtCO2)', :].sum() for k, i in result.items()})

    heat_pump = pd.Series({k: i.loc['Stock Heat pump (Million)', end] for k, i in result.items()})
    energy_poverty = pd.Series({k: i.loc['Energy poverty (Million)', end] for k, i in result.items()})
    stock_low_efficient = pd.Series({k: i.loc['Stock low-efficient (Million)', end] for k, i in result.items()})
    renovation = pd.Series({k: i.loc['Renovation (Thousand households)', :].sum() for k, i in result.items()})

    # graph ACB
    variables = {'CBA Consumption saving EE (Billion euro)': 'Saving EE',
                 'CBA Consumption saving prices (Billion euro)': 'Saving price',
                 'CBA Thermal comfort EE (Billion euro)': 'Comfort EE',
                 'CBA Emission direct (Billion euro)': 'Direct emission',
                 'CBA Thermal loss prices (Billion euro)': 'Comfort prices',
                 'CBA Annuities heater (Billion euro)': 'Annuities heater',
                 'CBA Annuities insulation (Billion euro)': 'Annuities insulation',
                 'CBA Carbon Emission indirect (Billion euro)': 'Indirect emission',
                 'CBA Health cost (Billion euro)': 'Health cost',
                 'CBA COFP (Billion euro)': 'COFP'
                 }
    df = pd.DataFrame({k: i.loc[variables.keys(), :].sum(axis=1) for k, i in result.items()}).round(3)
    df = df.rename(index=variables)
    diff = (df.T - df[reference]).T
    cba_diff_total = diff.sum(axis=0).rename('Total')

    df = pd.concat((consumption_saving_percent,
                    emission_saving_percent,
                    cba_diff_total,
                    investment_total,
                    subsidies_total,
                    energy_poverty,
                    heat_pump,
                    emission,
                    stock_low_efficient,
                    renovation),
                   keys=['Consumption saving (%)',
                         'Emission saving (%)',
                         'CBA diff (Billion euro per year)',
                         'Investment (Billion euro)',
                         'Subsidies (Billion euro)',
                         'Energy poverty (Million)',
                         'Heat pump (Million)',
                         'Cumulated emission (MtCO2)',
                         'Stock low efficient (Million)',
                         'Renovation (Thousand households)'
                         ],
                   axis=1)
    df.dropna(inplace=True)
    df.to_csv(os.path.join(folder_img, '..', 'result.csv'))

    make_scatter_plot(df, 'Emission saving (%)', 'CBA diff (Billion euro per year)',
                      'Emission saving (%)', 'Cost-benefit analysis to {} (Billion euro per year)'.format(end),
                      hlines=0,
                      format_x=lambda x, _: '{:.0%}'.format(x), xmin=0,
                      format_y=lambda y, _: '{:.1f}'.format(y),
                      annotate=False,
                      col_size='Energy poverty (Million)',
                      save=os.path.join(folder_img, 'cba_emission.png')
                      )
    make_scatter_plot(df, 'Consumption saving (%)', 'CBA diff (Billion euro per year)',
                      'Consumption saving (%)', 'Cost-benefit analysis to {} (Billion euro per year)'.format(end),
                      hlines=0,
                      format_x=lambda x, _: '{:.0%}'.format(x),
                      format_y=lambda y, _: '{:.1f}'.format(y),
                      annotate=False,
                      col_size='Energy poverty (Million)',
                      save=os.path.join(folder_img, 'cba_consumption.png')
                      )
    make_scatter_plot(df, 'Investment (Billion euro)', 'CBA diff (Billion euro per year)',
                      'Investment (Billion euro)', 'Cost-benefit analysis to {} (Billion euro per year)'.format(end),
                      hlines=0,
                      format_x=lambda x, _: '{:,.0f}'.format(x),
                      format_y=lambda y, _: '{:.1f}'.format(y),
                      annotate=False,
                      save=os.path.join(folder_img, 'cba_investment.png')
                      )
    make_scatter_plot(df, 'Subsidies (Billion euro)', 'CBA diff (Billion euro per year)',
                      'Subsidies (Billion euro)', 'Cost-benefit analysis to {} (Billion euro per year)'.format(end),
                      hlines=0,
                      format_x=lambda x, _: '{:,.0f}'.format(x),
                      format_y=lambda y, _: '{:.1f}'.format(y),
                      annotate=False,
                      save=os.path.join(folder_img, 'cba_subsidies.png')
                      )
    make_scatter_plot(df, 'Cumulated emission (MtCO2)', 'Investment (Billion euro)',
                      'Cumulated emission (MtCO2)', 'Investment to {} (Billion euro)'.format(end),
                      format_x=lambda x, _: '{:,.0f}'.format(x),
                      format_y=lambda y, _: '{:.0f}'.format(y),
                      annotate=False,
                      save=os.path.join(folder_img, 'investment_cumulated_emission.png')
                      )
    make_scatter_plot(df, 'Cumulated emission (MtCO2)', 'CBA diff (Billion euro per year)',
                      'Cumulated emission (MtCO2)', 'Cost-benefit analysis to {} (Billion euro per year)'.format(end),
                      hlines=0,
                      format_x=lambda x, _: '{:,.0f}'.format(x),
                      format_y=lambda y, _: '{:.1f}'.format(y),
                      annotate=False,
                      save=os.path.join(folder_img, 'cba_cumulated_emission.png')
                      )
    make_scatter_plot(df, 'Energy poverty (Million)', 'CBA diff (Billion euro per year)',
                      'Energy poverty (Million)', 'Cost-benefit analysis to {} (Billion euro per year)'.format(end),
                      hlines=0,
                      format_x=lambda x, _: '{:,.0f}'.format(x),
                      format_y=lambda y, _: '{:.1f}'.format(y),
                      annotate=False,
                      save=os.path.join(folder_img, 'cba_energy_poverty.png')
                      )
    make_scatter_plot(df, 'Energy poverty (Million)', 'CBA diff (Billion euro per year)',
                      'Energy poverty (Million)', 'Cost-benefit analysis to {} (Billion euro per year)'.format(end),
                      hlines=0,
                      format_x=lambda x, _: '{:,.0f}'.format(x),
                      format_y=lambda y, _: '{:.1f}'.format(y),
                      annotate=False,
                      save=os.path.join(folder_img, 'cba_energy_poverty.png')
                      )


def indicator_policies(result, folder, cba_inputs, discount_rate=0.032, years=30, policy_name=None,
                       reference='Reference', order_scenarios=None, figure=True):

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

    def cost_benefit_analysis(data, scenarios, policy_name=None, save=None, factor_cofp=0.2, embodied_emission=True,
                              cofp=True, order_scenarios=None):
        """Calculate socioeconomic NPV.

        Double difference is calculated with : scenario - reference
        NPV is presented as such for ZP+1: ZP+1 - ZP
        If the scenario requires more investment than the reference, then the difference of investments is
        positive, and it is taken into account in the NPV as a negative impact: - Investment total
        If the scenario results in less energy consumed then the reference, then energy savings is positive,
        and taken into account in the NPV as a positive account.

        NPV is presented as AP - AP-1 for AP-1, opposit signs to always represent the net effect of the policy

        Parameters
        ----------
        data: pd.DataFrame
            Dataframe with all scenarios.
        scenarios: list
            List of scenarios to calculate NPV.
        policy_name: str or None, default None
            Name of the policy.
        save: str or None, default None
            Path to save the result.
        factor_cofp: float, default 0.2
            Factor to convert COFP in euro.
        embodied_emission: bool, default True
            If True, embodied emission are taken into account.
        cofp: bool, default True
            If True, cofp are taken into account.

        Returns
        -------
        pd.DataFrame
        """
        npv = {}
        for s in scenarios:
            df = data.loc[:, s]
            temp = dict()
            temp.update({'Investment': df['Investment total WT (Billion euro)']})
            if embodied_emission:
                temp.update({'Embodied emission': df['Carbon footprint (Billion euro)']})
            if cofp:
                temp.update({'Opportunity cost': (df['Subsidies total (Billion euro)'] - df['VAT (Billion euro)'] +
                                      df['Simple difference Health expenditure (Billion euro)']
                                      ) * factor_cofp})

            temp.update({'Energy saving': sum(df['Energy expenditures {} (Billion euro)'.format(i)]
                                              for i in resources_data['index']['Energy'])})

            # temp.update({'Comfort EE': - df['Thermal comfort EE (Billion euro)']})
            # temp.update({'Comfort prices': + df['Thermal loss prices (Billion euro)']})
            temp.update({'Thermal comfort': - df['Thermal comfort EE (Billion euro)'] + df['Thermal loss prices (Billion euro)']})

            temp.update({'Emission saving': sum(df['Carbon value {} (Billion euro)'.format(i)]
                                                for i in resources_data['index']['Energy'])})

            temp.update({'Health cost': df['Health cost (Billion euro)']})
            temp.update({'Unobserved value': df['Hidden cost (Billion euro)']})

            if isinstance(policy_name, list):
                policy_name = '-'.join(policy_name)

            if 'AP' in s:
                temp = pd.Series(temp)
                title = policy_name + ' : AP - ({})'.format(s)
            elif 'ZP' in s:
                temp = - pd.Series(temp)
                title = policy_name + ' : ({}) - ZP'.format(s)
            else:
                temp = pd.Series(temp)
                title = '{}'.format(s)

            if save:
                if cofp:
                    waterfall_chart(- temp, title=title,
                                    save=os.path.join(save, 'npv_{}_cofp.png'.format(s.lower().replace(' ', '_'))),
                                    colors=resources_data['colors'])

                else:
                    waterfall_chart(- temp, title=title,
                                    save=os.path.join(save, 'npv_{}_no_cofp.png'.format(s.lower().replace(' ', '_'))),
                                    colors=resources_data['colors'])

            npv[title] = temp

        npv = - pd.DataFrame(npv)
        if save:
            npv.drop('Embodied emission', axis=0, inplace=True, errors='ignore')
            if order_scenarios is not None:
                npv = npv.loc[:, [i for i in order_scenarios if i in npv.columns]]
            npv_private = npv.loc[['Investment', 'Energy saving', 'Thermal comfort', 'Unobserved value'], :].sum()
            make_stackedbar_plot(npv.T, 'Cost-benefits analysis (Billion euro)', ncol=3, ymin=None,
                                 format_y=lambda y, _: '{:.0f} B€'.format(y),
                                 hline=0, colors=resources_data['colors'],
                                 scatterplot=npv.sum(),
                                 save=os.path.join(save, 'cost_benefit_analysis_counterfactual.png'.lower().replace(' ', '_')),
                                 rotation=0, left=1.2, fontxtick=12, scatterplot_bis=npv_private)

        npv.loc['NPV', :] = npv.sum()
        npv.columns = scenarios
        return npv

    folder_policies = os.path.join(folder, 'policies')
    if not os.path.isdir(folder_policies):
        os.mkdir(folder_policies)

    if 'Discount rate' in cba_inputs.keys():
        discount_rate = float(cba_inputs['Discount rate'])

    if 'Lifetime' in cba_inputs.keys():
        years = int(cba_inputs['Lifetime'])

    # Getting inputs needed
    energy_prices = get_pandas(cba_inputs['energy_prices'], lambda x: pd.read_csv(x, index_col=[0])) * 10 ** 9  # euro/kWh to euro/TWh
    carbon_value = get_series(cba_inputs['carbon_value'], header=None)  # euro/kWh to euro/TWh
    carbon_emission = get_pandas(cba_inputs['carbon_emission'], lambda x: pd.read_csv(x, index_col=[0])) * 10 ** 6

    # euro/tCO2 * tCO2/TWh  = euro/TWh
    carbon_emission_value = (carbon_value * carbon_emission.T).T  # euro/TWh
    carbon_emission_value.dropna(how='all', inplace=True)

    if policy_name is not None:
        if isinstance(policy_name, list):
            policy_name = [i.replace('_', ' ').capitalize() for i in policy_name]
        else:
            policy_name = policy_name.replace('_', ' ').capitalize()

    # Double difference = Scenario - Reference
    scenarios = [s for s in result.keys() if s != reference and s != 'ZP']
    comparison = {}
    for scenario in scenarios:
        temp_comparison = {}

        # selecting input
        data = result[scenario]
        ref = result[reference]
        if 'ZP' in scenario:
            ref = result['ZP']

        for var in ['Consumption standard (TWh)',
                    'Consumption (TWh)',
                    'Emission (MtCO2)',
                    'Health cost (Billion euro)',
                    ]:
            temp_comparison[var] = double_difference(ref.loc[var, :], data.loc[var, :], values=None,
                                                     discount_rate=discount_rate, years=years)

        # We cannot calculate directly double difference of carbon value because carbon price will increase.
        for energy in resources_data['index']['Energy']:
            var = 'Consumption {} (TWh)'.format(energy)
            temp_comparison[var] = double_difference(ref.loc[var, :], data.loc[var, :], values=None)

            temp_comparison['Carbon value {} (Billion euro)'.format(energy)] = double_difference(ref.loc[var, :],
                                                                                      data.loc[var, :],
                                                                                      values=carbon_emission_value[energy]) / (10 ** 9)

            temp_comparison['Energy expenditures {} (Billion euro)'.format(energy)] = double_difference(
                ref.loc[var, :],
                data.loc[var, :],
                values=energy_prices[energy]) / (10 ** 9)
            # On a des euros

            temp_comparison['Emission {} (tCO2)'.format(energy)] = double_difference(ref.loc[var, :],
                                                                          data.loc[var, :],
                                                                          values=carbon_emission[energy])

        for var in ['Thermal comfort EE (Billion euro)', 'Thermal loss prices (Billion euro)']:
            simple_diff = data.loc[var, :] - ref.loc[var, :]
            simple_diff.rename(None, inplace=True)
            _discount = pd.Series([1 / (1 + discount_rate) ** i for i in range(years)])
            _result = simple_diff * _discount.sum()
            _discount = pd.Series([1 / (1 + discount_rate) ** i for i in range(_result.shape[0])],
                                  index=_result.index)
            temp_comparison[var] = (_result * _discount).sum()

        # We use simple diff when effect do not last
        variable = ['Investment total WT (Billion euro)',
                    'Hidden cost heater (Billion euro)',
                    'Hidden cost insulation (Billion euro)',
                    'Subsidies total (Billion euro)',
                    'VAT (Billion euro)',
                    'Health expenditure (Billion euro)',
                    'Carbon value indirect (Billion euro)',
                    ]
        if policy_name is not None:
            if not isinstance(policy_name, list):
                policy_name = [policy_name]

            cost_var = ['{} (Billion euro)'.format(i) for i in policy_name if
                        ('{} (Billion euro)'.format(i) in data.index)]
            cost_name = '{} (Billion euro)'.format('-'.join(policy_name))
            variable += [cost_var]

        for var in variable:
            name = var

            if var == 'Health expenditure (Billion euro)':
                name = 'Simple difference ' + var
            elif var == 'Carbon value indirect (Billion euro)':
                name = 'Carbon footprint (Billion euro)'

            if isinstance(var, list):
                # For now, only the case for policy cost
                try:
                    diff = (data.loc[var, :].sum() - ref.loc[var, :].sum())
                    name = cost_name
                except KeyError:
                    # That means no direct cost are associated with the policy (regulation, etc...).
                    diff = 0
                    name = cost_name
            else:
                diff = data.loc[var, :] - ref.loc[var, :]

            if isinstance(diff, (float, int)):
                temp_comparison[name] = 0
            else:
                discount = pd.Series(
                    [1 / (1 + discount_rate) ** i for i in range(diff.shape[0])],
                    index=diff.index)
                temp_comparison[name] = (diff * discount.T).sum()

        comparison[scenario] = temp_comparison
    comparison = pd.DataFrame(comparison)
    comparison.loc['Hidden cost (Billion euro)', :] = comparison.loc['Hidden cost heater (Billion euro)', :] + comparison.loc['Hidden cost insulation (Billion euro)', :]
    comparison.sort_index(axis=1, inplace=True)
    comparison.round(2).to_csv(os.path.join(folder_policies, 'comparison.csv'))

    temp = ['AP-{}'.format(y) for y in range(2018, 2051)]
    temp += ['ZP+{}'.format(y) for y in range(2018, 2051)]
    efficiency_scenarios = list(set(comparison.columns).intersection(temp))
    # cost-efficiency
    indicator = dict()
    # We want efficiency only for concerned scenario policy (that is cut at t-1)
    # comp_efficiency = comparison.loc[:, efficiency_scenarios]

    indicator.update({'Investment total WT (Billion euro)': comparison.loc['Investment total WT (Billion euro)']})
    indicator.update({'Consumption (TWh)': comparison.loc['Consumption (TWh)']})
    indicator.update({'Consumption standard (TWh)': comparison.loc['Consumption standard (TWh)']})
    indicator.update({'Emission (MtCO2)': comparison.loc['Emission (MtCO2)']})
    indicator.update({'Investment / energy savings (euro/kWh)': - comparison.loc['Investment total WT (Billion euro)'] / comparison.loc['Consumption (TWh)']})
    indicator.update({'Investment / energy savings standard (euro/kWh)': - comparison.loc['Investment total WT (Billion euro)'] / comparison.loc['Consumption standard (TWh)']})
    indicator.update({'Investment / emission (euro/tCO2)': - comparison.loc['Investment total WT (Billion euro)'] / comparison.loc['Emission (MtCO2)'] * 10**3})

    if policy_name is not None:
        policy_cost = comparison.loc[cost_name]
        if not (policy_cost == 0).all():
            indicator.update({cost_name: policy_cost})
            indicator.update({'Cost effectiveness (euro/kWh)': - policy_cost / comparison.loc['Consumption (TWh)']})
            indicator.update({'Cost effectiveness standard (euro/kWh)': - policy_cost / comparison.loc[
                'Consumption standard (TWh)']})
            indicator.update({'Cost effectiveness carbon (euro/tCO2)': - policy_cost / comparison.loc[
                'Emission (MtCO2)'] * 10**3})
            indicator.update({'Leverage (%)': comparison.loc['Investment total WT (Billion euro)'] / policy_cost})

            indicator = pd.DataFrame(indicator).T

            # And impact on retrofit rate : difference in retrofit rate / cost of subvention
            for s in efficiency_scenarios:
                try:
                    year = int(s[-4:])
                except:
                    year = result[s].columns[-1]

                ref = reference
                if 'ZP' in s:
                    ref = 'ZP'

                # Only work for non-cumulative insulation policies that we want to assess together
                if len(policy_name) == 2:
                    p_name = '-'.join(policy_name)
                    for k in [s, ref]:
                        result[k].loc['{} (Thousand households)'.format(p_name), :] = sum([result[k].loc['{} (Thousand households)'.format(i), :] for i in policy_name])
                        result[k].loc['{} insulation (Thousand households)'.format(p_name), :] = sum([result[k].loc['{} insulation (Thousand households)'.format(i), :] for i in policy_name])
                        result[k].loc['Average cost {} (euro)'.format(p_name), :] = sum([(result[k].loc['Average cost {} (euro)'.format(i), :] * result[k].loc['{} (Thousand households)'.format(i), :]) for i in policy_name]) / result[k].loc['{} (Thousand households)'.format(p_name), :]
                        result[k].loc['Average cost {} insulation (euro)'.format(p_name), :] = sum([(result[k].loc['Average cost {} insulation (euro)'.format(i), :] * result[k].loc['{} insulation (Thousand households)'.format(i), :]) for i in policy_name]) / result[k].loc['{} insulation (Thousand households)'.format(p_name), :]
                        result[k].loc['Average energy std saved {} insulation (MWh)'.format(p_name), :] = sum([(result[k].loc['Average energy std saved {} insulation (MWh)'.format(i), :] * result[k].loc['{} insulation (Thousand households)'.format(i), :]) for i in policy_name]) / result[k].loc['{} insulation (Thousand households)'.format(p_name), :]

                if year in result[ref].columns and len(policy_name) <= 2:
                    p_name = '-'.join(policy_name)
                    if 'Average cost {} (euro)'.format(p_name) in result[ref].index:

                        if ref == 'ZP':
                            indicator.loc['Intensive margin (euro)', s] = result[s].loc['Average cost {} (euro)'.format(p_name), year] - result[ref].loc['Average cost {} (euro)'.format(p_name), year]
                            indicator.loc['Intensive margin (%)', s] = indicator.loc['Intensive margin (euro)', s] / result[ref].loc['Average cost {} (euro)'.format(p_name), year]

                        else:
                            indicator.loc['Intensive margin (euro)', s] = result[ref].loc[
                                                                              'Average cost {} (euro)'.format(
                                                                                  p_name), year] - result[s].loc[
                                                                              'Average cost {} (euro)'.format(
                                                                                  p_name), year]
                            indicator.loc['Intensive margin (%)', s] = indicator.loc['Intensive margin (euro)', s] / \
                                                                       result[s].loc['Average cost {} (euro)'.format(
                                                                           p_name), year]

                        if 'Average cost {} insulation (euro)'.format(p_name) in result[ref].index:

                            if ref == 'ZP':
                                indicator.loc['Intensive margin insulation (euro)', s] = result[s].loc['Average cost {} insulation (euro)'.format(p_name), year] - \
                                                                                         result[ref].loc['Average cost {} insulation (euro)'.format(p_name), year]
                                indicator.loc['Intensive margin insulation (%)', s] = indicator.loc['Intensive margin insulation (euro)', s] / \
                                                                           result[ref].loc['Average cost {} insulation (euro)'.format(p_name), year]
                            else:
                                indicator.loc['Intensive margin insulation (euro)', s] = result[ref].loc['Average cost {} insulation (euro)'.format(p_name), year] - \
                                                                                         result[s].loc['Average cost {} insulation (euro)'.format(p_name), year]
                                indicator.loc['Intensive margin insulation (%)', s] = indicator.loc['Intensive margin insulation (euro)', s] / \
                                                                           result[s].loc['Average cost {} insulation (euro)'.format(
                                                                               p_name), year]

                        if 'Average cost {} heater (euro)'.format(p_name) in result[ref].index:

                            if ref == 'ZP':
                                indicator.loc['Intensive margin heater (euro)', s] = result[s].loc['Average cost {} heater (euro)'.format(p_name), year] - \
                                                                              result[ref].loc['Average cost {} heater (euro)'.format(p_name), year]
                                indicator.loc['Intensive margin heater (%)', s] = indicator.loc['Intensive margin heater (euro)', s] / result[ref].loc['Average cost {} heater (euro)'.format(p_name), year]
                            else:
                                indicator.loc['Intensive margin heater (euro)', s] = result[ref].loc['Average cost {} heater (euro)'.format(p_name), year] - \
                                                                              result[s].loc['Average cost {} heater (euro)'.format(p_name), year]
                                indicator.loc['Intensive margin heater (%)', s] = indicator.loc['Intensive margin heater (euro)', s] / \
                                                                           result[s].loc['Average cost {} heater (euro)'.format(p_name), year]

                        if ref == 'ZP':
                            indicator.loc['Non additional investment (Thousand households)', s] = result[ref].loc['{} (Thousand households)'.format(p_name), year]
                            indicator.loc['Additional investment (Thousand households)', s] = result[s].loc['{} (Thousand households)'.format(p_name), year] - result[ref].loc['{} (Thousand households)'.format(p_name), year]
                        else:
                            indicator.loc['Non additional investment (Thousand households)', s] = result[s].loc['{} (Thousand households)'.format(p_name), year]
                            indicator.loc['Additional investment (Thousand households)', s] = result[ref].loc['{} (Thousand households)'.format(p_name), year] - result[s].loc['{} (Thousand households)'.format(p_name), year]

                        indicator.loc['Extensive margin (%)', s] = indicator.loc['Additional investment (Thousand households)', s] / (indicator.loc['Additional investment (Thousand households)', s] + indicator.loc['Non additional investment (Thousand households)', s])
                        indicator.loc['Freeriding ratio (%)', s] = indicator.loc['Non additional investment (Thousand households)', s] / (indicator.loc['Additional investment (Thousand households)', s] + indicator.loc['Non additional investment (Thousand households)', s])

                        if '{} insulation (Thousand households)'.format(p_name) in result[ref].index:

                            if ref == 'ZP':
                                indicator.loc['Non additional insulation (Thousand households)', s] = result[ref].loc['{} insulation (Thousand households)'.format(p_name), year]
                                indicator.loc['Additional insulation (Thousand households)', s] = result[s].loc['{} insulation (Thousand households)'.format(p_name), year] - result[ref].loc['{} insulation (Thousand households)'.format(p_name), year]
                            else:
                                indicator.loc['Non additional insulation (Thousand households)', s] = result[s].loc['{} insulation (Thousand households)'.format(p_name), year]
                                indicator.loc['Additional insulation (Thousand households)', s] = result[ref].loc['{} insulation (Thousand households)'.format(p_name), year] - result[s].loc['{} insulation (Thousand households)'.format(p_name), year]

                            indicator.loc['Freeriding ratio insulation (%)', s] = indicator.loc['Non additional insulation (Thousand households)', s] / (indicator.loc['Additional insulation (Thousand households)', s] + indicator.loc['Non additional insulation (Thousand households)', s])
                            indicator.loc['Extensive margin insulation (%)', s] = indicator.loc['Additional insulation (Thousand households)', s] / (indicator.loc['Additional insulation (Thousand households)', s] + indicator.loc['Non additional insulation (Thousand households)', s])

                        if '{} heater (Thousand households)'.format(p_name) in result[ref].index:

                            if ref == 'ZP':
                                indicator.loc['Non additional heater (Thousand households)', s] = result[ref].loc[
                                    '{} heater (Thousand households)'.format(p_name), year]
                                indicator.loc['Additional heater (Thousand households)', s] = result[s].loc[
                                    '{} heater (Thousand households)'.format(p_name), year] - \
                                    result[ref].loc['{} heater (Thousand households)'.format(p_name), year]
                            else:
                                indicator.loc['Non additional heater (Thousand households)', s] = result[s].loc[
                                    '{} heater (Thousand households)'.format(p_name), year]
                                indicator.loc['Additional heater (Thousand households)', s] = result[ref].loc[
                                    '{} heater (Thousand households)'.format(p_name), year] - \
                                    result[s].loc['{} heater (Thousand households)'.format(p_name), year]

                            indicator.loc['Freeriding ratio heater (%)', s] = indicator.loc['Non additional heater (Thousand households)', s] / (indicator.loc['Additional heater (Thousand households)', s] + indicator.loc['Non additional heater (Thousand households)', s])
                            indicator.loc['Extensive margin heater (%)', s] = indicator.loc['Additional heater (Thousand households)', s] / (indicator.loc['Additional heater (Thousand households)', s] + indicator.loc['Non additional heater (Thousand households)', s])

                    if 'Average energy std saved {} insulation (MWh)'.format(p_name) in result[ref].index:
                        if ref == 'ZP':
                            indicator.loc['Intensive margin insulation (MWh)', s] = result[s].loc['Average energy std saved {} insulation (MWh)'.format(p_name), year] - result[ref].loc['Average energy std saved {} insulation (MWh)'.format(p_name), year]
                            indicator.loc['Intensive margin insulation energy (%)', s] = indicator.loc['Intensive margin insulation (MWh)', s] / result[ref].loc['Average energy std saved {} insulation (MWh)'.format(p_name), year]
                        else:
                            indicator.loc['Intensive margin insulation (MWh)', s] = result[ref].loc['Average energy std saved {} insulation (MWh)'.format(p_name), year] - result[s].loc['Average energy std saved {} insulation (MWh)'.format(p_name), year]
                            indicator.loc['Intensive margin insulation energy (%)', s] = indicator.loc['Intensive margin insulation (MWh)', s] / result[s].loc['Average energy std saved {} insulation (MWh)'.format(p_name), year]

                indicator.loc['Consumption saving (TWh/year)', s] = result[s].loc['Consumption (TWh)', year] - \
                                                                    result[ref].loc['Consumption (TWh)', year]
                indicator.loc['Emission saving (MtCO2/year)', s] = result[s].loc['Emission (MtCO2)', year] - \
                                                                    result[ref].loc['Emission (MtCO2)', year]
                indicator.loc['Cost-benefits analysis (Billion euro/year)', s] = result[s].loc['Cost-benefits analysis (Billion euro)', year] - \
                                                                    result[ref].loc['Cost-benefits analysis (Billion euro)', year]
                if False:
                    indicator.loc['Balance Tenant private - C1 (euro/year.household)', s] = result[s].loc['Balance Tenant private - C1 (euro/year.household)', year] - \
                                                                        result[ref].loc['Balance Tenant private - C1 (euro/year.household)', year]
                    indicator.loc['Balance Owner-occupied - C1 (euro/year.household)', s] = result[s].loc[
                                                                                                'Balance Owner-occupied - C1 (euro/year.household)', year] - \
                                                                                            result[ref].loc[
                                                                                                'Balance Owner-occupied - C1 (euro/year.household)', year]

            indicator.sort_index(inplace=True)
            indicator.sort_index(axis=1, inplace=True)
        else:
            indicator = pd.DataFrame(indicator).T
    else:
        indicator = pd.DataFrame(indicator).T

    effectiveness_scenarios = [s for s in comparison.columns if s not in efficiency_scenarios]
    if effectiveness_scenarios:
        if figure is True:
            save = folder_policies
        else:
            save = None
        cba = cost_benefit_analysis(comparison, effectiveness_scenarios, policy_name=policy_name, save=save,
                                    order_scenarios=order_scenarios)
        if indicator is not None:
            if set(list(cba.index)).issubset(list(indicator.index)):
                indicator.loc[list(cba.index), s] = cba[s]
            else:
                indicator = pd.concat((indicator, cba), axis=0)
        else:
            indicator = cba

        for s in effectiveness_scenarios:
            if 'ZP' in s:
                ref = 'ZP'
            elif 'AP' in s:
                ref = 'Reference'
            else:
                ref = reference

            variables_end = {
                'Consumption (TWh)': 'Consumption saving (TWh)',
                'Consumption standard (TWh)': 'Consumption standard saving (TWh)',
                'Consumption standard (kWh/m2)': 'Consumption standard saving (kWh/m2)',
                'Emission (MtCO2)': 'Emission saving (MtCO2)',
                'Heating intensity (%)': 'Heating intensity diff (%)',
                'Energy poverty (Million)': 'Energy poverty diff (Million)'
            }

            variables_cumulated = {
                'Investment total (Billion euro)': 'Investment total diff (Billion euro)',
                'Subsidies total (Billion euro)': 'Subsidies total diff (Billion euro)',
                'Investment heater (Billion euro)': 'Investment heater diff (Billion euro)',
                'Subsidies heater (Billion euro)': 'Subsidies heater diff (Billion euro)',
                'Investment insulation (Billion euro)': 'Investment insulation diff (Billion euro)',
                'Subsidies insulation (Billion euro)': 'Subsidies insulation diff (Billion euro)',
                'Emission (MtCO2)': 'Cumulated emission saving (MtCO2)',
                'Consumption (TWh)': 'Cumulated energy saving (TWh)'
            }

            end = result[ref].columns[-1]
            for var, name in variables_end.items():
                if ref == 'ZP':
                    temp = result[s].loc[var, end] - result[ref].loc[var, end]
                    temp_percent = temp / result[ref].loc[var, end]
                else:
                    temp = result[ref].loc[var, end] - result[s].loc[var, end]
                    temp_percent = temp / result[s].loc[var, end]

                indicator.loc[name, s] = temp
                indicator.loc['{} (%)'.format(name.split(' (')[0]), s] = temp_percent

            for var, name in variables_cumulated.items():
                if ref == 'ZP':
                    temp = result[s].loc[var, :].sum() - result[ref].loc[var, :].sum()
                    temp_percent = temp / result[ref].loc[var, :].sum()
                else:
                    temp = result[ref].loc[var, :].sum() - result[s].loc[var, :].sum()
                    temp_percent = temp / result[s].loc[var, :].sum()

                indicator.loc[name, s] = temp
                indicator.loc['{} (%)'.format(name.split(' (')[0]), s] = temp_percent

    if indicator is not None:
        indicator.round(3).to_csv(os.path.join(folder_policies, 'indicator.csv'))

    return comparison, indicator


def compare_results(output, path):
    """Write comparison between measured and calculated output.

    Parameters
    ----------
    output: Series
    path: str
    """

    data_validation = resources_data['data_validation']
    df = pd.concat((data_validation, output.reindex(data_validation.index).rename('Calculated')), axis=1)
    df.round(1).to_csv(os.path.join(path, 'validation.csv'))


def make_summary(path, option=None):

    images = []
    # 1. reference - input
    if option == 'input':
        path_reference = os.path.join(path, 'Reference')

        temp = ['ini/stock.png', 'ini/thermal_insulation.png', 'energy_prices.png', 'cost_curve_insulation.png', 'policy_scenario.png']
        temp += ['calibration/result_policies_assessment.png']
        images = [os.path.join(path_reference, i) for i in temp]

        # 2. result - reference
        path_reference_result = os.path.join(path_reference, 'img')
        temp = ['stock_performance.png', 'consumption_heater_saving.png', 'emission.png',
                'investment.png', 'financing_households.png', 'policies_validation.png']
        images += [os.path.join(path_reference_result, i) for i in temp]

    path_compare = os.path.join(path, 'img')
    temp = ['consumption.png',
            'emission.png',
            'stock_performance.png',
            'stock_heater.png',
            'consumption_heater.png',
            'consumption_saving_decomposition.png',
            'emission_saving_decomposition.png',
            'renovation.png',
            'investment_total.png',
            'policy_scenario.png',
            'cost_benefit_analysis_counterfactual.png',
            'energy_poverty.png',
            'energy_income_ratio_std_2020.png',
            'energy_income_ratio_std_2050.png',
            'cba_consumption.png',
            'cba_emission.png',
            ]
    images += [os.path.join(path_compare, i) for i in temp]

    # 3. result - compare
    path_policies = os.path.join(path, 'policies')
    temp = ['cost_benefit_analysis_counterfactual.png']
    images += [os.path.join(path_policies, i) for i in temp]

    images = [i for i in images if os.path.isfile(i)]

    images = [Image.open(img) for img in images]
    new_images = []
    for png in images:
        png.load()
        background = Image.new('RGB', png.size, (255, 255, 255))
        background.paste(png, mask=png.split()[3])  # 3 is the alpha channel
        new_images.append(background)

    pdf_path = os.path.join(path, 'summary.pdf')

    new_images[0].save(
        pdf_path, 'PDF', resolution=100.0, save_all=True, append_images=new_images[1:]
    )

