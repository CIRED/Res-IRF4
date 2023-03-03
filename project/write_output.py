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

from project.input.resources import resources_data
from project.utils import make_plot, make_grouped_subplots, make_area_plot, waterfall_chart, \
    assessment_scenarios, format_ax, format_legend, save_fig, make_uncertainty_plot
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
    make_area_plot(df, 'Energy consumption (TWh)', colors=resources_data['colors'],
                   format_y=lambda y, _: '{:.0f}'.format(y),
                   save=os.path.join(path, 'consumption_heater.png'),
                   total=False, loc='left', left=1.2) # scatter=resources_data['consumption_total_objectives']

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

    # graph subsidies
    try:
        non_subsidies = ['subsidies_cap', 'obligation']
        subsidies = output.loc[['{} (Billion euro)'.format(i.capitalize().replace('_', ' ')) for i in buildings.policies if i not in non_subsidies], :]
        taxes_expenditures = output.loc[['{} (Billion euro)'.format(i.capitalize().replace('_', ' ').replace('Cee', 'Cee tax')) for i in buildings.taxes_list], :]

        subset = pd.concat((subsidies, -taxes_expenditures.loc[['{} (Billion euro)'.format(i) for i in ['Cee tax', 'Carbon tax']],:]), axis=0)
        subset.fillna(0, inplace=True)
        subset = subset.loc[:, (subset != 0).any(axis=0)].T
        subset.columns = [c.split(' (Billion euro)')[0].capitalize().replace('_', ' ') for c in subset.columns]
        if not subset.empty:
            scatter = resources_data['public_policies_2019']
            if scatter is not None and list(scatter.index) == ['Cee', 'Cite', 'Mpr', 'Reduced tax', 'Zero interest loan', 'Mpr serenite']:
                fig, ax = plt.subplots(1, 2, figsize=(12.8, 9.6), gridspec_kw={'width_ratios': [1, 5]}, sharey=True)
                subset.index = subset.index.astype(int)
                subset.plot.area(ax=ax[1], stacked=True, color=resources_data['colors'])
                scatter.T.plot.bar(ax=ax[0], stacked=True, color=resources_data['colors'], legend=False, width=1.5, rot=0)
                ax[0] = format_ax(ax[0], y_label='Billion euro', xinteger=True, format_y=lambda y, _: '{:.0f}'.format(y), ymin=None)
                ax[1] = format_ax(ax[1], xinteger=True, format_y=lambda y, _: '{:.0f}'.format(y), ymin=None)
                subset.sum(axis=1).rename('Total').plot(ax=ax[1], color='black')
                format_legend(ax[1], loc='left', left=1.2)
                ax[0].set_title('Realized')
                ax[1].set_title('Model results')
                save_fig(fig, save=os.path.join(path, 'policies_validation.png'))

                make_area_plot(subset, 'Policies cost (Billion euro)', save=os.path.join(path, 'policies.png'),
                               colors=resources_data['colors'], format_y=lambda y, _: '{:.0f}'.format(y),
                               loc='left', left=1.2)

            else:
                make_area_plot(subset, 'Policies cost (Billion euro)', save=os.path.join(path, 'policies.png'),
                               colors=resources_data['colors'], format_y=lambda y, _: '{:.0f}'.format(y),
                               scatter=resources_data['public_policies_2019'], loc='left', left=1.2)
    except KeyError:
        print('Policies graphic impossible because lack of subsidy color')

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

    subset = output.loc[['Balance Tenant private - {} (euro/year.household)'.format(i) for i in resources_data['index']['Income tenant']], :].T
    subset.dropna(how='any', inplace=True)
    subset.columns = resources_data['index']['Income tenant']
    if not subset.empty:
        make_plot(subset, 'Balance Tenant private (euro per year)',
                  save=os.path.join(path, 'balance_tenant.png'),
                  format_y=lambda y, _: '{:.0f}'.format(y),
                  colors=resources_data['colors'], ymin=None)

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

    images_ini = ['energy_prices.png', 'stock_performance.png', 'cost_curve_insulation.png']
    images_ini = [os.path.join(buildings.path, i) for i in images_ini]

    images_to_save = ['stock_performance.png', 'consumption_heater.png', 'emission.png']
    images_to_save = [os.path.join(path, i) for i in images_to_save]
    images = [Image.open(img) for img in images_to_save + images_ini]
    new_images = []
    for png in images:
        png.load()
        background = Image.new('RGB', png.size, (255, 255, 255))
        background.paste(png, mask=png.split()[3])  # 3 is the alpha channel
        new_images.append(background)

    pdf_path = os.path.join(buildings.path, 'summary_pdf.pdf')

    new_images[0].save(
        pdf_path, 'PDF', resolution=100.0, save_all=True, append_images=new_images[1:]
    )


def grouped_output(result, folder, config_runs=None, config_sensitivity=None, quintiles=None):
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
    result: dict
    folder: str
    config_runs: dict or None, default None
    config_sensitivity: dict or None, default None
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

    def details_graphs(data, v, inf, folder_img):
        n = (v.split(' {}')[0] + '_' + inf[0] + '.png').replace(' ', '_').lower()
        temp = grouped(data, [v.format(i) for i in resources_data['index'][inf[0]]])
        replace = {v.format(i): i for i in resources_data['index'][inf[0]]}
        temp = {replace[key]: item.astype(float).interpolate(limit_area='inside') for key, item in temp.items()}

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
                              order=resources_data['index'][inf[0]])

    if quintiles:
        resources_data['index']['Income tenant'] = resources_data['quintiles']
        resources_data['index']['Income owner'] = resources_data['quintiles']

    folder_img = os.path.join(folder, 'img')
    if not os.path.isdir(folder_img):
        os.mkdir(folder_img)

    variables = {'Consumption (TWh)': ('consumption_hist.png', lambda y, _: '{:,.0f}'.format(y),
                                       resources_data['consumption_total_hist'],
                                       resources_data['consumption_total_objectives']),
                 'Consumption standard (TWh)': ('consumption_standard.png', lambda y, _: '{:,.0f}'.format(y)),
                 'Heating intensity (%)': ('heating_intensity.png', lambda y, _: '{:,.0%}'.format(y)),
                 'Emission (MtCO2)': ('emission.png', lambda y, _: '{:,.0f}'.format(y), None,
                                       resources_data['emissions_total_objectives']),
                 'Stock Heat pump (Million)': ('stock_heat_pump.png', lambda y, _: '{:,.0f}'.format(y)),
                 'Energy poverty (Million)': ('energy_poverty.png', lambda y, _: '{:,.1f}'.format(y)),
                 'Retrofit (Thousand households)': ('retrofit.png', lambda y, _: '{:,.0f}'.format(y)),
                 'Renovation (Thousand households)': ('renovation.png', lambda y, _: '{:,.0f}'.format(y)),
                 'Investment total (Thousand euro/household)': ('investment_households.png', lambda y, _: '{:,.0f}'.format(y)),
                 'Consumption saving insulation (TWh/year)': ('saving_insulation.png', lambda y, _: '{:,.1f}'.format(y)),
                 'Consumption saving heater (TWh/year)': ('saving_heater.png', lambda y, _: '{:,.1f}'.format(y)),
                 'Retrofit at least 1 EPC (Thousand households)': (
                     'retrofit_jump_comparison.png', lambda y, _: '{:,.0f}'.format(y),
                     resources_data['retrofit_comparison']),
                 'Investment total (Billion euro)': ('investment_total.png', lambda y, _: '{:,.0f}'.format(y)),
                 'Subsidies total (Billion euro)': ('subsidies_total.png', lambda y, _: '{:,.0f}'.format(y)),
                 'Energy expenditures (Billion euro)': (
                 'energy_expenditures.png', lambda y, _: '{:,.0f}'.format(y)),
                 'Health cost (Billion euro)': ('health_cost.png', lambda y, _: '{:,.0f}'.format(y)),
                 'Replacement total (Thousand)': ('replacement_total.png', lambda y, _: '{:,.0f}'.format(y)),
                 'Replacement insulation (Thousand)': ('replacement_insulation.png', lambda y, _: '{:,.0f}'.format(y)),
                 'Switch heater (Thousand households)': ('switch_heater.png', lambda y, _: '{:,.0f}'.format(y))
                 }

    for variable, infos in variables.items():
        temp = pd.DataFrame({scenario: output.loc[variable, :] for scenario, output in result.items()}).astype(float)
        temp = temp.interpolate(limit_area='inside')

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

        make_plot(temp, variable, save=os.path.join(folder_img, '{}'.format(infos[0])), format_y=infos[1], scatter=scatter)

    variables_output = {
        'Consumption {} (TWh)': [
            ('Energy', lambda y, _: '{:,.0f}'.format(y), 2, resources_data['consumption_hist'])],
        'Stock {} (Million)': [('Performance', lambda y, _: '{:,.0f}'.format(y)),
                               ('Heater', lambda y, _: '{:,.0f}'.format(y))],
        'Investment {} (Billion euro)': [('Insulation', lambda y, _: '{:,.0f}'.format(y), 2)],
        'Retrofit measures {} (Thousand households)': [('Count', lambda y, _: '{:,.0f}'.format(y), 2)],
        'Renovation {} (Thousand households)': [('Decision maker', lambda y, _: '{:,.0f}'.format(y), 2)],
    }
    #         'Share subsidies {} (%)': [('Income owner', lambda y, _: '{:,.1%}'.format(y))],

    for var, infos in variables_output.items():
        for info in infos:
            details_graphs(result, var, info, folder_img)

    if 'Reference' in result.keys() and len(result.keys()) > 1 and config_runs is not None:
        indicator_policies(result, folder, config_runs)

    # TODO: uncertainty plot to work on
    if False:
        if 'Reference' in result.keys() and len(result.keys()) > 1 and config_sensitivity is not None:
            variables = {'Consumption (TWh)': ('consumption_hist_uncertainty.png', lambda y, _: '{:,.0f}'.format(y),
                                               resources_data['consumption_total_hist'],
                                               resources_data['consumption_total_objectives']),
                         'Emission (MtCO2)': ('emission_uncertainty.png', lambda y, _: '{:,.0f}'.format(y)),
                         'Retrofit at least 1 EPC (Thousand households)': (
                             'retrofit_jump_comparison_uncertainty.png', lambda y, _: '{:,.0f}'.format(y),
                             resources_data['retrofit_comparison'])
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


def indicator_policies(result, folder, config, discount_rate=0.045, years=30):

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

        for energy in resources_data['index']['Energy']:
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
        for var in ['Subsidies total (Billion euro)', 'VTA (Billion euro)', 'Investment total HT (Billion euro)',
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

        if var in result[s].index and var in ref.index:
            discount = pd.Series([1 / (1 + discount_rate) ** i for i in range(ref.loc[var, :].shape[0])],
                                 index=ref.loc[var, :].index)
            rslt[var] = (((result[s].loc[var, :]).fillna(0) - ref.loc[var, :].fillna(0)) * discount.T).sum()
            # We had NaN for year t with AP-t scnarios, so replaced these with 0... is it ok?
        elif var in ref.index:
            discount = pd.Series([1 / (1 + discount_rate) ** i for i in range(ref.loc[var, :].shape[0])],
                                 index=ref.loc[var, :].index)
            rslt[var] = (- ref.loc[var, :] * discount.T).sum()
        elif var in result[s].index:
            discount = pd.Series([1 / (1 + discount_rate) ** i for i in range(result[s].loc[var, :].shape[0])],
                                 index=result[s].loc[var, :].index)
            rslt[var] = (result[s].loc[var, :] * discount.T).sum()
        else:
            rslt[var] = 0
        comparison[s] = rslt

    comparison = pd.DataFrame(comparison)

    # Efficiency: AP and AP-t scenarios
    efficiency_scenarios = list(set(comparison.columns).intersection(['AP-{}'.format(y) for y in range(2018, 2051)]))
    indicator = dict()
    if efficiency_scenarios:
        # We want efficiency only for concerned scenario policy (that is cut at t-1)
        comp_efficiency = comparison.loc[:, efficiency_scenarios]

        policy_cost = comp_efficiency.loc['{} (Billion euro)'.format(policy_name)]
        indicator.update({'{} (Billion euro)'.format(policy_name): policy_cost})
        indicator.update({'Investment total HT (Billion euro)': comp_efficiency.loc['Investment total HT (Billion euro)']})

        indicator.update({'Consumption (TWh)': comp_efficiency.loc['Consumption (TWh)']})
        indicator.update({'Consumption standard (TWh)': comp_efficiency.loc['Consumption standard (TWh)']})
        indicator.update({'Emission (MtCO2)': comp_efficiency.loc['Emission (MtCO2)']})
        indicator.update({'Cost effectiveness (euro/kWh)': - policy_cost / comp_efficiency.loc['Consumption (TWh)']})
        indicator.update({'Cost effectiveness standard (euro/kWh)': - policy_cost / comp_efficiency.loc[
            'Consumption standard (TWh)']})
        indicator.update({'Cost effectiveness carbon (euro/tCO2)': - policy_cost / comp_efficiency.loc[
            'Emission (MtCO2)'] * 10**3})
        indicator.update({'Leverage (%)': comp_efficiency.loc['Investment total HT (Billion euro)'] / policy_cost})
        indicator.update({'Investment / energy savings (euro/kWh)': comp_efficiency.loc['Investment total HT (Billion euro)'] / comp_efficiency.loc['Consumption (TWh)']})
        indicator.update({'Investment / energy savings standard (euro/kWh)': comp_efficiency.loc['Investment total HT (Billion euro)'] / comp_efficiency.loc['Consumption standard (TWh)']})
        indicator.update({'Investment / emission (euro/tCO2)': comp_efficiency.loc['Investment total HT (Billion euro)'] / comp_efficiency.loc['Emission (MtCO2)'] * 10**3})

        indicator = pd.DataFrame(indicator).T

        # Retrofit ratio = freerider ratio
        # And impact on retrofit rate : difference in retrofit rate / cost of subvention
        for s in efficiency_scenarios:
            year = int(s[-4:])
            if year in result['Reference'].columns:

                indicator.loc['Intensive margin diff (Thousand euro / households)', s] = result['Reference'].loc['Investment insulation / households (Thousand euro)', year] - result[s].loc['Investment insulation / households (Thousand euro)', year]
                indicator.loc['Intensive margin diff (%)', s] = indicator.loc['Intensive margin diff (Thousand euro / households)', s] / result['Reference'].loc['Investment insulation / households (Thousand euro)', year]
                indicator.loc['Freeriding renovation (Thousand households)', s] = result[s].loc['Renovation (Thousand households)', year]
                indicator.loc['Non-freeriding renovation (Thousand households)', s] = result['Reference'].loc['Renovation (Thousand households)', year] - (
                    result[s].loc['Renovation (Thousand households)', year])
                indicator.loc['Freeriding investment diff (Billion euro)', s] = indicator.loc['Freeriding renovation (Thousand households)', s] * 10**3 * indicator.loc['Intensive margin diff (Thousand euro / households)', s] / 10**6
                indicator.loc['Non-freeriding investment diff (Billion euro)', s] = indicator.loc['Non-freeriding renovation (Thousand households)', s] * 10**3 * result[s].loc['Investment insulation / households (Thousand euro)', year] / 10**6
                indicator.loc['Freeriding renovation ratio (%)', s] = result[s].loc['Renovation (Thousand households)', year] / (
                    result['Reference'].loc['Renovation (Thousand households)', year])
                decile = ['D{}'.format(i) for i in range(1, 11)]
                df = result[s].loc[['Renovation {} (Thousand households)'.format(d) for d in decile], year] / \
                     result['Reference'].loc[['Renovation {} (Thousand households)'.format(d) for d in decile], year]
                df.index = ['Freeriding renovation ratio {} (%)'.format(d) for d in decile]
                df.name = s
                df = pd.DataFrame(df)
                # This part to be improved
                if set(list(df.index)).issubset(list(indicator.index)):
                    indicator.loc[list(df.index), s] = df[s]
                else:
                    indicator = pd.concat((indicator, df), axis=0)

    else:
        indicator = pd.DataFrame(indicator).T

    indicator.sort_index(axis=1, inplace=True)

    def socioeconomic_npv(data, scenarios, pol_name, save=None, factor_cofp=0.2, embodied_emission=True, cofp=True):
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
            temp.update({'Investment': df['Investment total HT (Billion euro)']})
            if embodied_emission:
                temp.update({'Embodied emission additional': df['Carbon footprint (Billion euro)']})
            if cofp:
                temp.update({'Cofp': (df['Subsidies total (Billion euro)'] - df['VTA (Billion euro)'] +
                                      df['Simple difference Health expenditure (Billion euro)']
                                      ) * factor_cofp})

            temp.update({'Energy saving': sum(df['Energy expenditures {} (Billion euro)'.format(i)]
                                              for i in resources_data['index']['Energy'])})
            temp.update({'Emission saving': sum(df['Carbon value {} (Billion euro)'.format(i)]
                                                for i in resources_data['index']['Energy'])})
            temp.update({'Well-being benefit': df['Loss of well-being (Billion euro)']})
            temp.update({'Health savings': df['Health expenditure (Billion euro)']})
            temp.update({'Mortality reduction benefit': df['Social cost of mortality (Billion euro)']})
            if 'AP' in s:
                temp = pd.Series(temp)
                title = pol_name + ' : AP - ({})'.format(s)
            else:
                temp = - pd.Series(temp)
                title = pol_name + ' : ({})- ZP'.format(s)

            if save:
                if cofp:
                    waterfall_chart(temp, title=title,
                                save=os.path.join(save, 'npv_{}_cofp.png'.format(s.lower().replace(' ', '_'))),
                                colors=resources_data['colors'])

                    waterfall_chart(temp.loc[temp.index != 'Cofp'], title=title,
                                save=os.path.join(save, 'npv_{}_no_cofp.png'.format(s.lower().replace(' ', '_'))),
                                colors=resources_data['colors'])
                else:
                    waterfall_chart(temp, title=title,
                                save=os.path.join(save, 'npv_{}_no_cofp.png'.format(s.lower().replace(' ', '_'))),
                                colors=resources_data['colors'])

            npv[title] = temp

        npv = pd.DataFrame(npv)
        if save:
            if cofp:
                assessment_scenarios(npv.T, save=os.path.join(save, 'npv_cofp.png'.lower().replace(' ', '_')), colors=resources_data['colors'])
                assessment_scenarios(npv.loc[npv.index != 'Cofp'].T, save=os.path.join(save, 'npv_no_cofp.png'.lower().replace(' ', '_')),
                                     colors=resources_data['colors'])
            else:
                assessment_scenarios(npv.T, save=os.path.join(save, 'npv.png'.lower().replace(' ', '_')),
                                     colors=resources_data['colors'])


        npv.loc['NPV', :] = npv.sum()
        return npv

    # Effectiveness : AP/AP-1 and ZP/ ZP+1 scenarios
    effectiveness_scenarios = [s for s in comparison.columns if s not in efficiency_scenarios]
    if effectiveness_scenarios:
        se_npv = socioeconomic_npv(comparison, effectiveness_scenarios, policy_name, save=folder_policies)
        if indicator is not None:
            if set(list(se_npv.index)).issubset(list(indicator.index)):
                indicator.loc[list(se_npv.index), s] = se_npv[s]
            else:
                indicator = pd.concat((indicator, se_npv), axis=0)
        else:
            indicator = se_npv
        # Percentage of objectives accomplished

        # Objectives in param (resources_data), we need to make this cleaner but for now:

        """"* result['Reference'].loc['Sizing factor (%)'].iloc[0]"""
        # TODO: have sizing factor in generic input
        comparison_results_energy = pd.DataFrame([result[s].loc['Consumption (TWh)'] for s in effectiveness_scenarios],
                                                 index=effectiveness_scenarios).T
        comparison_results_emissions = pd.DataFrame([result[s].loc['Emission (MtCO2)'] for s in effectiveness_scenarios],
                                                    index=effectiveness_scenarios).T

        # Selecting years with corresponding objectives and calculating the % of objective accomplished
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

        for y in resources_data['emissions_total_objectives'] .index:
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

