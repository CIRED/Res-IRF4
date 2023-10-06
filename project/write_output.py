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
    make_uncertainty_plot, format_table, select, make_clusterstackedbar_plot
from project.utils import stack_catplot, make_relplot, make_stackedbar_plot, make_scatter_plot
from itertools import product
from PIL import Image


def plot_scenario(output, stock, buildings, detailed_graph=False):
    path = os.path.join(buildings.path, 'img')
    if not os.path.isdir(path):
        os.mkdir(path)

    if buildings.quintiles:
        resources_data['index']['Income tenant'] = resources_data['quintiles']
        resources_data['index']['Income owner'] = resources_data['quintiles']

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

    # subsidies
    non_subsidies = ['subsidies_cap', 'obligation']
    temp = ['{} (Billion euro)'.format(i.capitalize().replace('_', ' ')) for i in buildings.policies + ['reduced_vta'] if i not in non_subsidies]
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
            df = pd.concat((df, df_historic), axis=0, keys=['Model', 'Realized'], names=['Source']).reset_index('Source')
            df.reset_index(drop=True, inplace=True)

            yrs = [str(i) for i in subset.index if i in resources_data['policies_hist'].index and i != buildings.first_year]
            df = df[df['Years'].isin(yrs)]

            palette = {s: resources_data['colors'][s] for s in df['Policies'].unique()}
            stack_catplot(x='Years', y='Data', cat='Source', stack='Policies', data=df, palette=palette,
                          y_label='Policies amount (Billion euro)',
                          save=os.path.join(path, 'policies_validation.png'),
                          format_y=lambda y, _: '{:.0f}'.format(y))

            make_area_plot(subset, 'Policies cost (Billion euro)', save=os.path.join(path, 'policies.png'),
                           colors=resources_data['colors'], format_y=lambda y, _: '{:.0f}'.format(y),
                           loc='left', left=1.2)

        else:
            make_area_plot(subset, 'Policies cost (Billion euro)', save=os.path.join(path, 'policies.png'),
                           colors=resources_data['colors'], format_y=lambda y, _: '{:.0f}'.format(y),
                           scatter=historic, loc='left', left=1.2)

    # graph public finance
    subset = output.loc[['VTA (Billion euro)', 'Taxes expenditure (Billion euro)', 'Subsidies heater (Billion euro)',
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


def plot_compare_scenarios(result, folder, quintiles=None):
    """Grouped scenarios output.

    Parameters
    ----------
    result: dict
    folder: str
    quintiles: bool, default None


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

    def details_graphs(data, v, inf, folder_img, colors=None):

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

        make_grouped_subplots(dict_data, format_y=inf.get('format_y'), n_columns=n_columns, save=os.path.join(folder_img, n),
                              order=order, scatter=inf.get('scatter'), colors=colors)

    if quintiles:
        resources_data['index']['Income tenant'] = resources_data['quintiles']
        resources_data['index']['Income owner'] = resources_data['quintiles']

    colors = dict(zip(result.keys(), sns.color_palette(n_colors=len(result.keys()))))
    colors.update({'Historic': (0, 0, 0)})
    colors_add = sns.color_palette("husl", 10)

    folder_img = os.path.join(folder, 'img')
    if not os.path.isdir(folder_img):
        os.mkdir(folder_img)

    # ini
    emission_ini = result.get('Reference').loc['Emission (MtCO2)', :].iloc[0]
    consumption_ini = result.get('Reference').loc['Consumption (TWh)', :].iloc[0]
    start = result.get('Reference').columns[0]
    end = result.get('Reference').columns[-1]

    # graph comparison stock
    variables = [('Stock {} (Million)', 'Heater'),
                 ('Stock {} (Million)', 'Performance'),
                 ('Consumption {} (TWh)', 'Heater'),
                 ('Consumption {} (TWh)', 'Performance')]
    for v in variables:
        name = v[0].split(' {}')[0].lower()
        groupby = v[1]
        v = v[0]
        years = [start, 2030, 2050]
        years = [i for i in years if (i >= start) and (i <= end)]
        temp = grouped(result, [v.format(i) for i in resources_data['index'][groupby]])
        temp = {k: i.loc[years, :] for k, i in temp.items()}
        replace = {v.format(i): i for i in resources_data['index'][groupby]}
        temp = {replace[key]: item for key, item in temp.items()}
        temp = pd.concat(temp).rename_axis([groupby, 'Years'], axis=0).rename_axis('Scenario', axis=1)
        temp = temp.stack('Scenario').unstack('Years')
        if not temp.empty:
            if len(temp.columns) > 1:
                make_clusterstackedbar_plot(temp, groupby, colors=resources_data['colors'],
                                            format_y=lambda y, _: '{:.0f}'.format(y),
                                            save=os.path.join(folder_img, '{}_{}.png'.format(name, groupby.lower())),
                                            rotation=90, year_ini=start)

    # graph emission saving
    variables = {
        'Emission saving heater (MtCO2/year)': 'Switch heater',
        'Emission saving insulation (MtCO2/year)': 'Insulation',
        'Emission saving prices (MtCO2/year)': 'Prices',
        'Emission saving carbon content (MtCO2/year)': 'Carbon content',
    }
    colors_temp = {'Carbon content': 'forestgreen',
                   'Prices': 'grey',
                   'Switch heater': 'royalblue',
                   'Insulation': 'firebrick'
                   }
    df = pd.DataFrame({k: i.loc[variables.keys(), :].sum(axis=1) for k, i in result.items()}).round(3)
    df = df.rename(index=variables)
    df /= emission_ini
    make_stackedbar_plot(df.T, 'Emission saving to {} (% of initial emission {} - {:.0f} MtCO2)'.format(end, start, emission_ini), ncol=2, ymin=None, format_y=lambda y, _: '{:.0%}'.format(y),
                         colors=colors_temp, save=os.path.join(folder_img, 'emission_saving_decomposition.png'),
                         rotation=20, left=1.2)
    emission_saving = df.sum()

    # consumption saving
    variables = {
        'CBA Consumption saving EE (TWh)': 'Energy efficiency',
        'CBA Consumption saving prices (TWh)': 'Prices',
        'CBA Rebound EE (TWh)': 'Rebound'
    }
    colors_temp = {
        'Energy efficiency': 'firebrick',
        'Prices': 'grey',
        'Rebound': 'black'
    }
    df = pd.DataFrame({k: i.loc[variables.keys(), :].sum(axis=1) for k, i in result.items()}).round(3)
    df = df.rename(index=variables)
    df /= consumption_ini
    make_stackedbar_plot(df.T, 'Consumption saving to {} (% of initial emission {} - {:.0f} TWh)'.format(end, start, consumption_ini),
                         ncol=2,
                         ymin=None, format_y=lambda y, _: '{:.0%}'.format(y),
                         colors=colors_temp, save=os.path.join(folder_img, 'consumption_saving_decomposition.png'),
                         rotation=20,  left=1.2)
    consumption_saving = df.loc[['Energy efficiency', 'Prices']].sum()

    # graph running cost
    try:
        subsidies_total = pd.Series({k: i.loc['Subsidies total (Billion euro)', :].sum() for k, i in result.items()})
        variables = {'Cost energy (Billion euro)': 'Energy expenditure',
                     'Loss thermal comfort (Billion euro)': 'Comfort',
                     'Cost emission (Billion euro)': 'Direct emission',
                     'Cost heath (Billion euro)': 'Health cost',
                     'Cost heater (Billion euro)': 'Annuities heater',
                     'Cost insulation (Billion euro)': 'Annuities insulation',
                     'COFP (Billion euro)': 'COFP'
                     }

        df = pd.DataFrame({k: i.loc[variables.keys(), :].sum(axis=1) for k, i in result.items()}).round(3)
        df = df.rename(index=variables)
        cost_total = df.sum(axis=0).rename('Total')
        make_stackedbar_plot(df.T, 'Running cost to {} (Billion euro)'.format(end), ncol=3, ymin=None,
                             format_y=lambda y, _: '{:.0f}'.format(y),
                             hline=0, scatterplot=cost_total, colors=resources_data['colors'],
                             save=os.path.join(folder_img, 'running_cost.png'),
                             rotation=0, left=1.3)

        # colors
        diff = (df.T - df['Reference']).T
        if not diff.empty and diff.shape[1] > 1:
            cost_diff_total = diff.T.sum(axis=1).rename('Total')
            make_stackedbar_plot(diff.drop('Reference', axis=1).T,
                                 'Running cost compare to Reference to {} (Billion euro)'.format(end), ncol=3, ymin=None,
                                 format_y=lambda y, _: '{:.0f}'.format(y),
                                 hline=0, scatterplot=cost_diff_total, colors=resources_data['colors'],
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

        # graph CBA
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
                             rotation=0, left=1.3)

        diff = (df.T - df['Reference']).T
        if not diff.empty and diff.shape[1] > 1:
            cba_diff_total = diff.T.sum(axis=1).rename('Total')

            make_stackedbar_plot(diff.drop('Reference', axis=1).T,
                                 'Cost-benefits analysis compare to Reference to {} (Billion euro)'.format(end),
                                 ncol=3, ymin=None,
                                 format_y=lambda y, _: '{:.0f}'.format(y),
                                 hline=0, scatterplot=cba_diff_total, colors=resources_data['colors'],
                                 save=os.path.join(folder_img, 'cost_benefit_analysis_comparison.png'), rotation=0,
                                 left=1.3)

            df = pd.concat((consumption_saving, emission_saving, cba_total, cba_diff_total, pd.Series(colors), subsidies_total), axis=1,
                           keys=['Consumption saving (TWh)',
                                 'Emission saving (MtCO2)',
                                 'CBA (Billion euro)',
                                 'CBA diff (Billion euro)',
                                 'colors',
                                 'Subsidies (Billion euro)'
                                 ])
            df.dropna(inplace=True)

            make_scatter_plot(df, 'Consumption saving (TWh)', 'CBA diff (Billion euro)',
                              'Consumption saving to {} (TWh)'.format(end),
                              'Cost benefit analysis to {} (Billion euro)'.format(end),
                              hlines=0,
                              format_x=lambda x, _: '{:.0%}'.format(x), xmin=0,
                              format_y=lambda y, _: '{:.1f}'.format(y),
                              save=os.path.join(folder_img, 'cba_consumption.png'),
                              col_colors='colors',
                              col_size='Subsidies (Billion euro)'
                              )

            make_scatter_plot(df, 'Emission saving (MtCO2)', 'CBA diff (Billion euro)',
                              'Emission saving to {} (MtCO2)'.format(end),
                              'Cost benefit analysis to {} (Billion euro)'.format(end),
                              hlines=0,
                              format_x=lambda x, _: '{:.0%}'.format(x), xmin=0,
                              format_y=lambda y, _: '{:.1f}'.format(y),
                              save=os.path.join(folder_img, 'cba_emission.png'),
                              col_colors='colors',
                              col_size='Subsidies (Billion euro)'
                              )
    except KeyError:
        pass

    # graph cost


    # graph distributive impact
    levels = ['Housing type', 'Occupancy status', 'Income tenant']
    idx = list(product(*[resources_data['index'][i] for i in levels]))

    years = [2020, 2030, 2040, 2050, max(result['Reference'].columns)]
    years = list(set(years))
    years = [y for y in years if y in result['Reference'].columns]

    ratio = ['Ratio expenditure {} - {} - {} (%)'.format(i[0], i[1], i[2]) for i in idx]

    idx = pd.MultiIndex.from_tuples(idx, names=levels)
    dict_rslt = {k: i.loc[ratio, :].set_axis(idx, axis=0).dropna(axis=1, how='all') for k, i in result.items()}

    start = min(dict_rslt['Reference'].columns)
    ini = pd.DataFrame({k: i.loc[:, start] for k, i in dict_rslt.items() if start in i.columns})

    for year in years:
        df = pd.DataFrame({k: i.loc[:, year] for k, i in dict_rslt.items() if year in i.columns})

        data = select(df, {'Occupancy status': ['Owner-occupied', 'Privately rented']})
        data = format_table(data, name='Scenarios')
        data['Decision maker'] = data['Housing type'] + ' - ' + data['Occupancy status']

        make_relplot(data, x='Income tenant', y='Data', col='Decision maker', hue='Scenarios',
                     palette=colors, save=os.path.join(folder_img, 'energy_income_ratio_{}.png'.format(year)),
                     title='Energy expenditure on income ratio in {}'.format(year))

        # > 0 positive means households are loosing money compare to ref
        if df.shape[1] > 1:
            rate = ((df.T - df.loc[:, 'Reference']) / df.loc[:, 'Reference']).T
            rate = rate.loc[:, [i for i in rate.columns if i != 'Reference']]
            rate = select(rate, {'Occupancy status': ['Owner-occupied', 'Privately rented']})
            rate = format_table(rate, name='Scenarios')
            rate['Decision maker'] = rate['Housing type'] + ' - ' + rate['Occupancy status']

            make_relplot(rate, x='Income tenant', y='Data', col='Decision maker', hue='Scenarios',
                         palette=colors, save=os.path.join(folder_img, 'energy_income_ratio_rate_{}.png'.format(year)),
                         title='Energy expenditure on income ratio in {} compare to Reference'.format(year))

        # > 0 positive means households are loosing money compare to ini
        diff = (df - ini) / ini
        diff = select(diff, {'Occupancy status': ['Owner-occupied', 'Privately rented']})
        diff = format_table(diff, name='Scenarios')
        diff['Decision maker'] = diff['Housing type'] + ' - ' + diff['Occupancy status']

        make_relplot(diff, x='Income tenant', y='Data', col='Decision maker', hue='Scenarios',
                     palette=colors, save=os.path.join(folder_img, 'energy_income_ratio_ini_{}.png'.format(year)),
                     title='Energy expenditure on income ratio in {} compare to Base year'.format(year))

    # graph line plot 2D comparison
    consumption_total_hist = pd.DataFrame(resources_data['consumption_hist'])
    consumption_total_hist.index = consumption_total_hist.index.astype(int)
    if result['Reference'].loc['Consumption Heating (TWh)', :].iloc[0] == 0:
        consumption_total_hist.drop('Heating', axis=1, inplace=True)
    consumption_total_hist = consumption_total_hist.sum(axis=1).rename('Historic')

    variables = {'Consumption (TWh)': {'name': 'consumption_hist.png',
                                       'format_y': lambda y, _: '{:,.0f}'.format(y),
                                       'exogenous': consumption_total_hist
                                       },
                 'Consumption standard (TWh)': {'name': 'consumption_standard.png',
                                                'format_y': lambda y, _: '{:,.0f}'.format(y)},
                 'Heating intensity (%)': {'name': 'heating_intensity.png',
                                           'format_y': lambda y, _: '{:,.0%}'.format(y)},
                 'Emission (MtCO2)': {'name': 'emission.png',
                                      'format_y': lambda y, _: '{:,.0f}'.format(y)
                                      },
                 'Stock Heat pump (Million)': {'name': 'stock_heat_pump.png',
                                               'format_y': lambda y, _: '{:,.1f}'.format(y)},
                 'Energy poverty (Million)': {'name': 'energy_poverty.png',
                                              'format_y': lambda y, _: '{:,.1f}'.format(y)},
                 'Retrofit (Thousand households)': {'name': 'retrofit.png',
                                                    'format_y': lambda y, _: '{:,.0f}'.format(y)},
                 'Renovation (Thousand households)': {'name': 'renovation.png',
                                                      'format_y': lambda y, _: '{:,.0f}'.format(y)},
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
                                                     'format_y': lambda y, _: '{:,.0f}'.format(y)},
                 'Subsidies total (Billion euro)': {'name': 'subsidies_total.png',
                                                    'format_y': lambda y, _: '{:,.0f}'.format(y)},
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

        scatter = infos.get('scatter')

        colors_temp = colors
        t = [i for i in temp.columns if i not in colors_temp.keys()]
        if t:
            colors_temp.update({i: colors_add[k] for k, i in enumerate(t)})
        
        make_plot(temp, variable, save=os.path.join(folder_img, '{}'.format(infos['name'])), format_y=infos['format_y'],
                  scatter=scatter, colors=colors)

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
            details_graphs(result, var, info, folder_img, colors=colors)

    # TODO: uncertainty plot to work on
    if False:
        if 'Reference' in result.keys() and len(result.keys()) > 1 and config_sensitivity is not None:
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


def plot_compare_scenarios_simple(result, folder, quintiles=None):
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
    end = sorted(result.get('Reference').columns)[-1]

    emission_ini = result.get('Reference').loc['Emission (MtCO2)', :].iloc[0]
    consumption_ini = result.get('Reference').loc['Consumption (TWh)', :].iloc[0]

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
    diff = (df.T - df['Reference']).T
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


def indicator_policies(result, folder, cba_inputs, discount_rate=0.032, years=30, policy_name=None):

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
                              cofp=True, details_health=False):
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
        details_health: bool, default False
            If True, details of health are taken into account.


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
                temp.update({'Embodied emission additional': df['Carbon footprint (Billion euro)']})
            if cofp:
                temp.update({'COFP': (df['Subsidies total (Billion euro)'] - df['VTA (Billion euro)'] +
                                      df['Simple difference Health expenditure (Billion euro)']
                                      ) * factor_cofp})

            temp.update({'Energy saving': sum(df['Energy expenditures {} (Billion euro)'.format(i)]
                                              for i in resources_data['index']['Energy'])})

            temp.update({'Comfort EE': - df['Thermal comfort EE (Billion euro)']})

            temp.update({'Comfort prices': + df['Thermal loss prices (Billion euro)']})

            temp.update({'Emission saving': sum(df['Carbon value {} (Billion euro)'.format(i)]
                                                for i in resources_data['index']['Energy'])})

            if details_health:
                temp.update({'Well-being benefit': df['Loss of well-being (Billion euro)']})
                temp.update({'Health savings': df['Health expenditure (Billion euro)']})
                temp.update({'Mortality reduction benefit': df['Social cost of mortality (Billion euro)']})
            else:
                temp.update({'Health cost': df['Health cost (Billion euro)']})

            if 'AP' in s:
                temp = pd.Series(temp)
                title = policy_name + ' : AP - ({})'.format(s)
            elif 'ZP' in s:
                temp = - pd.Series(temp)
                title = policy_name + ' : ({})- ZP'.format(s)
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
            make_stackedbar_plot(npv.T, 'Cost-benefits analysis (Billion euro)', ncol=3, ymin=None,
                                 format_y=lambda y, _: '{:.0f}'.format(y),
                                 hline=0, colors=resources_data['colors'],
                                 scatterplot=npv.sum(),
                                 save=os.path.join(save, 'cost_benefit_analysis_counterfactual.png'.lower().replace(' ', '_')),
                                 rotation=0, left=1.3)
        npv.loc['NPV', :] = npv.sum()
        return npv

    folder_policies = os.path.join(folder, 'policies')
    if not os.path.isdir(folder_policies):
        os.mkdir(folder_policies)

    if 'Discount rate' in cba_inputs.keys():
        discount_rate = float(cba_inputs['Discount rate'])

    if 'Lifetime' in cba_inputs.keys():
        years = int(cba_inputs['Lifetime'])

    # Getting inputs needed
    energy_prices = pd.read_csv(cba_inputs['energy_prices'], index_col=[0]) * 10 ** 9  # euro/kWh to euro/TWh
    carbon_value = pd.read_csv(cba_inputs['carbon_value'], index_col=[0], header=None).squeeze()  # euro/tCO2
    carbon_emission = pd.read_csv(cba_inputs['carbon_emission'], index_col=[0]) * 10 ** 6  # unit: kgCO2/kWh to tCO2/TWh
    # euro/tCO2 * tCO2/TWh  = euro/TWh
    carbon_emission_value = (carbon_value * carbon_emission.T).T  # euro/TWh
    carbon_emission_value.dropna(how='all', inplace=True)

    if policy_name is not None:
        policy_name = policy_name.replace('_', ' ').capitalize()

    # Calculating simple and double differences for needed variables, and storing them in agg
    # Double difference = Scenario - Reference
    scenarios = [s for s in result.keys() if s != 'Reference' and s != 'ZP']
    comparison = {}
    for scenario in scenarios:
        data = result[scenario]
        ref = result['Reference']
        if 'ZP' in scenario:
            ref = result['ZP']

        rslt = {}
        for var in ['Consumption standard (TWh)', 'Consumption (TWh)', 'Energy poverty (Million)',
                    'Health expenditure (Billion euro)', 'Social cost of mortality (Billion euro)',
                    'Loss of well-being (Billion euro)', 'Health cost (Billion euro)', 'Emission (MtCO2)']:
            rslt[var] = double_difference(ref.loc[var, :], data.loc[var, :],
                                          values=None)

        for energy in resources_data['index']['Energy']:
            var = 'Consumption {} (TWh)'.format(energy)
            rslt[var] = double_difference(ref.loc[var, :], data.loc[var, :],
                                          values=None)
            rslt['Energy expenditures {} (Billion euro)'.format(energy)] = double_difference(
                ref.loc[var, :],
                data.loc[var, :],
                values=energy_prices[energy]) / (10 ** 9)
            # On a des euros

            rslt['Emission {} (tCO2)'.format(energy)] = double_difference(ref.loc[var, :],
                                                                          data.loc[var, :],
                                                                          values=carbon_emission[energy])

            rslt['Carbon value {} (Billion euro)'.format(energy)] = double_difference(ref.loc[var, :],
                                                                                      data.loc[var, :],
                                                                                      values=carbon_emission_value[energy]) / (
                                                                            10 ** 9)

        for var in ['Thermal comfort EE (Billion euro)', 'Thermal loss prices (Billion euro)', ]:
            simple_diff = data.loc[var, :] - ref.loc[var, :]
            simple_diff.rename(None, inplace=True)
            _discount = pd.Series([1 / (1 + discount_rate) ** i for i in range(years)])
            _result = simple_diff * _discount.sum()
            _discount = pd.Series([1 / (1 + discount_rate) ** i for i in range(_result.shape[0])],
                                  index=_result.index)
            rslt[var] = (_result * _discount).sum()

        # Simple diff = scenario - ref
        for var in ['Subsidies total (Billion euro)', 'VTA (Billion euro)', 'Investment total WT (Billion euro)',
                    'Health expenditure (Billion euro)']:
            discount = pd.Series(
                [1 / (1 + discount_rate) ** i for i in range(ref.loc[var, :].shape[0])],
                index=ref.loc[var, :].index)
            if var == 'Health expenditure (Billion euro)':
                rslt['Simple difference ' + var] = ((data.loc[var, :] - ref.loc[var, :]) * discount.T).sum()
            else:
                rslt[var] = ((data.loc[var, :] - ref.loc[var, :]) * discount.T).sum()

        var = 'Carbon footprint (MtCO2)'
        discount = pd.Series([1 / (1 + discount_rate) ** i for i in range(ref.loc[var, :].shape[0])],
                             index=ref.loc[var, :].index)
        rslt['Carbon footprint (Billion euro)'] = ((data.loc[var,
                                                    :] - ref.loc[var, :]) * discount.T * carbon_value).sum() / 10**3

        var = '{} (Billion euro)'.format(policy_name)

        if var in data.index and var in ref.index:
            discount = pd.Series([1 / (1 + discount_rate) ** i for i in range(ref.loc[var, :].shape[0])],
                                 index=ref.loc[var, :].index)
            rslt[var] = (((data.loc[var, :]).fillna(0) - ref.loc[var, :].fillna(0)) * discount.T).sum()
            # We had NaN for year t with AP-t scnarios, so replaced these with 0... is it ok?
        elif var in ref.index:
            discount = pd.Series([1 / (1 + discount_rate) ** i for i in range(ref.loc[var, :].shape[0])],
                                 index=ref.loc[var, :].index)
            rslt[var] = (- ref.loc[var, :] * discount.T).sum()
        elif var in data.index:
            discount = pd.Series([1 / (1 + discount_rate) ** i for i in range(data.loc[var, :].shape[0])],
                                 index=data.loc[var, :].index)
            rslt[var] = (data.loc[var, :] * discount.T).sum()
        else:
            rslt[var] = 0
        comparison[scenario] = rslt
    comparison = pd.DataFrame(comparison)

    indicator = None
    temp = ['AP-{}'.format(y) for y in range(2018, 2051)]
    temp += [i for i in comparison.columns if i[:3] == 'ZP+']
    efficiency_scenarios = list(set(comparison.columns).intersection(temp))
    # cost-efficiency
    if policy_name is not None:
        # cost-efficiency: AP and AP-t scenarios
        indicator = dict()
        if efficiency_scenarios:
            # We want efficiency only for concerned scenario policy (that is cut at t-1)
            comp_efficiency = comparison.loc[:, efficiency_scenarios]

            policy_cost = comp_efficiency.loc['{} (Billion euro)'.format(policy_name)]
            indicator.update({'{} (Billion euro)'.format(policy_name): policy_cost})
            indicator.update({'Investment total WT (Billion euro)': comp_efficiency.loc['Investment total WT (Billion euro)']})
            indicator.update({'Consumption (TWh)': comp_efficiency.loc['Consumption (TWh)']})
            indicator.update({'Consumption standard (TWh)': comp_efficiency.loc['Consumption standard (TWh)']})
            indicator.update({'Emission (MtCO2)': comp_efficiency.loc['Emission (MtCO2)']})
            indicator.update({'Cost effectiveness (euro/kWh)': - policy_cost / comp_efficiency.loc['Consumption (TWh)']})
            indicator.update({'Cost effectiveness standard (euro/kWh)': - policy_cost / comp_efficiency.loc[
                'Consumption standard (TWh)']})
            indicator.update({'Cost effectiveness carbon (euro/tCO2)': - policy_cost / comp_efficiency.loc[
                'Emission (MtCO2)'] * 10**3})
            indicator.update({'Leverage (%)': comp_efficiency.loc['Investment total WT (Billion euro)'] / policy_cost})
            indicator.update({'Investment / energy savings (euro/kWh)': comp_efficiency.loc['Investment total WT (Billion euro)'] / comp_efficiency.loc['Consumption (TWh)']})
            indicator.update({'Investment / energy savings standard (euro/kWh)': comp_efficiency.loc['Investment total WT (Billion euro)'] / comp_efficiency.loc['Consumption standard (TWh)']})
            indicator.update({'Investment / emission (euro/tCO2)': comp_efficiency.loc['Investment total WT (Billion euro)'] / comp_efficiency.loc['Emission (MtCO2)'] * 10**3})

            indicator = pd.DataFrame(indicator).T

            # And impact on retrofit rate : difference in retrofit rate / cost of subvention
            for s in efficiency_scenarios:
                try:
                    year = int(s[-4:])
                except:
                    year = result[s].columns[-1]
                if year in result['Reference'].columns:
                    """
                    indicator.loc['Investment insulation reference (Thousand euro/household)', s] = result['Reference'].loc['Investment insulation (Thousand euro/household)', year]
                    indicator.loc['Investment insulation (Thousand euro/household)', s] = result[s].loc['Investment insulation (Thousand euro/household)', year]
                    indicator.loc['Intensive margin (Thousand euro/household)', s] = indicator.loc['Investment insulation reference (Thousand euro/household)', s] - indicator.loc['Investment insulation (Thousand euro/household)', s]
                    indicator.loc['Intensive margin (%)', s] = indicator.loc['Intensive margin (Thousand euro/household)', s] / result['Reference'].loc['Investment insulation (Thousand euro/household)', year]
                    """

                    indicator.loc['Non additional renovation (Thousand households)', s] = result[s].loc['Renovation (Thousand households)', year]
                    indicator.loc['Additional renovation (Thousand households)', s] = result['Reference'].loc['Renovation (Thousand households)', year] - result[s].loc['Renovation (Thousand households)', year]
                    indicator.loc['Freeriding renovation ratio (%)', s] = indicator.loc['Non additional renovation (Thousand households)', s] / (indicator.loc['Additional renovation (Thousand households)', s] + indicator.loc['Non additional renovation (Thousand households)', s])
                    indicator.loc['Extensive margin (%)', s] = indicator.loc['Additional renovation (Thousand households)', s] / indicator.loc['Non additional renovation (Thousand households)', s]

                indicator.loc['Consumption saving (TWh/year)', s] = result[s].loc['Consumption (TWh)', year] - \
                                                                    result['Reference'].loc['Consumption (TWh)', year]
                indicator.loc['Emission saving (MtCO2/year)', s] = result[s].loc['Emission (MtCO2)', year] - \
                                                                    result['Reference'].loc['Emission (MtCO2)', year]
                indicator.loc['Cost-benefits analysis (Billion euro/year)', s] = result[s].loc['Cost-benefits analysis (Billion euro)', year] - \
                                                                    result['Reference'].loc['Cost-benefits analysis (Billion euro)', year]
                indicator.loc['Balance Tenant private - C1 (euro/year.household)', s] = result[s].loc['Balance Tenant private - C1 (euro/year.household)', year] - \
                                                                    result['Reference'].loc['Balance Tenant private - C1 (euro/year.household)', year]
                indicator.loc['Balance Owner-occupied - C1 (euro/year.household)', s] = result[s].loc[
                                                                                            'Balance Owner-occupied - C1 (euro/year.household)', year] - \
                                                                                        result['Reference'].loc[
                                                                                            'Balance Owner-occupied - C1 (euro/year.household)', year]



        else:
            indicator = pd.DataFrame(indicator).T
        indicator.sort_index(axis=1, inplace=True)

    # Effectiveness all scenarios except AP- and ZP-
    effectiveness_scenarios = [s for s in comparison.columns if s not in efficiency_scenarios]
    if effectiveness_scenarios:
        cba = cost_benefit_analysis(comparison, effectiveness_scenarios, policy_name=policy_name, save=folder_policies)
        if indicator is not None:
            if set(list(cba.index)).issubset(list(indicator.index)):
                indicator.loc[list(cba.index), s] = cba[s]
            else:
                indicator = pd.concat((indicator, cba), axis=0)
        else:
            indicator = cba
        # Percentage of objectives accomplished

        # Objectives in param (resources_data), we need to make this cleaner but for now:

        comparison_results_energy = pd.DataFrame([result[s].loc['Consumption (TWh)'] for s in effectiveness_scenarios],
                                                 index=effectiveness_scenarios).T
        comparison_results_emissions = pd.DataFrame([result[s].loc['Emission (MtCO2)'] for s in effectiveness_scenarios],
                                                    index=effectiveness_scenarios).T

        # Selecting years with corresponding objectives and calculating the % of objective accomplished
        if False:
            for y in resources_data['consumption_total_objectives'].index:
                if y in comparison_results_energy.index:
                    indicator.loc['Consumption reduction {} (TWh) '.format(y), :] = (comparison_results_energy.iloc[0] -
                                                                                     comparison_results_energy.loc[y]).T

                    indicator.loc['Consumption reduction Obj {} (TWh)'.format(y), :] = (comparison_results_energy.iloc[0] -
                                                                                        resources_data['consumption_total_objectives'].loc[y]).T

                    indicator.loc['Percentage of {} consumption objective (%)'.format(y), :] = (comparison_results_energy.iloc[0] -
                                                                                                comparison_results_energy.loc[y]).T / (
                                                                                                comparison_results_energy.iloc[0] -
                                                                                                resources_data['consumption_total_objectives'].loc[y]).T

            for y in resources_data['emissions_total_objectives'].index:
                if y in comparison_results_emissions.index:
                    indicator.loc['Emission reduction {} (MtCO2) '.format(y), :] = (comparison_results_emissions.iloc[0] -
                                                                                    comparison_results_emissions.loc[y]).T

                    indicator.loc['Emission reduction Obj {} (MtCO2)'.format(y), :] = (comparison_results_emissions.iloc[0] -
                                                                                       resources_data['emissions_total_objectives'] .loc[y]).T

                    indicator.loc['Percentage of {} emission objective (%)'.format(y), :] = (comparison_results_emissions.iloc[0] -
                                                                                             comparison_results_emissions.loc[y]).T / (
                                                                                             comparison_results_emissions.iloc[0] -
                                                                                             resources_data['emissions_total_objectives'] .loc[y]).T
            # low_eff_var = 'Stock low-efficient (Million)'
            # Objective is zero in 2030 - introduce it in params to make it resilient
            comparison_results_low_eff = pd.DataFrame([result[s].loc['Stock low-efficient (Million)']
                                                       for s in effectiveness_scenarios], index=effectiveness_scenarios).T
            for y in resources_data['low_eff_objectives'].index:
                if y in comparison_results_low_eff.index:
                    indicator.loc['Low-efficient stock reduction {} (Million) '.format(y), :] = (comparison_results_low_eff.iloc[0] -
                                                                                                 comparison_results_low_eff.loc[y]).T

                    indicator.loc['Low-efficient stock reduction objective {} (Million) '.format(y), :] = (comparison_results_low_eff.iloc[0] -
                                                                                                           resources_data['low_eff_objectives'].loc[y]).T

                    indicator.loc['Percentage of {} low-efficient objective (%) '.format(y), :] = (comparison_results_low_eff.iloc[0] -
                                                                                                   comparison_results_low_eff.loc[y]).T / (
                                                                                                   comparison_results_low_eff.iloc[0] -
                                                                                                   resources_data['low_eff_objectives'].loc[y]).T
        # Energy poverty
        # No objective so simply showing the reduction between first and last year
        energy_poverty = pd.DataFrame([result[s].loc['Energy poverty (Million)'] for s in effectiveness_scenarios],
                                      index=effectiveness_scenarios).T
        indicator.loc['Energy poverty (Thousand)'] = (energy_poverty.iloc[0] - energy_poverty.iloc[-1]) * 10**3

    comparison.round(2).to_csv(os.path.join(folder_policies, 'comparison.csv'))
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


def make_summary(path, option='input'):

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

    # 3. result - compare
    path_policies = os.path.join(path, 'policies')
    temp = ['cost_benefit_analysis_counterfactual.png']
    images += [os.path.join(path_policies, i) for i in temp]

    path_compare = os.path.join(path, 'img')
    temp = ['cost_benefit_analysis_comparison.png',
            'cba_consumption.png', 'cba_emission.png',
            'running_cost_comparison.png', 'running_cost_consumption.png', 'running_cost_emission.png',
            'consumption_saving_decomposition.png', 'emission_saving_decomposition.png',
            'energy_income_ratio_rate_2030.png',
            'renovation.png', 'stock_heat_pump.png', 'retrofit_measures_count.png', 'consumption_hist.png',
            'consumption_energy.png', 'emission.png', 'investment_total.png', 'subsidies_total.png',
            'efficiency_insulation.png', 'energy_poverty.png', 'retrofit_financing.png',
            'retrofit_decarbonize_options.png',
            ]
    images += [os.path.join(path_compare, i) for i in temp]

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

