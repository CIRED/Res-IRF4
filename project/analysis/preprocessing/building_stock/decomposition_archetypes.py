import pandas as pd
import os

# same than notebook

for number in [1, 3, 5]:
    print(number)
    stock_buildings = pd.read_csv('output/building_stock_sdes2018_aggregated.csv')
    stock_buildings = stock_buildings.rename(columns={'Heating energy': 'Energy', 'Energy performance': 'Performance'}).set_index(['Housing type', 'Performance', 'Energy', 'Occupancy status', 'Income owner', 'Income tenant'])
    print(stock_buildings.sum())

    archetypes = pd.read_csv(os.path.join('archetypes', 'ademe_dpe_buildingmodel', 'archetypes_{}.csv'.format(number)))
    archetypes = archetypes.rename(columns={'DPE': 'Performance'}).set_index(['Housing type', 'Performance', 'Energy'])
    archetypes = archetypes.drop(['Efficiency', 'clusters'], axis=1)

    columns = ['Housing type', 'Performance', 'Energy']
    archetypes_columns = ['Wall', 'Floor', 'Roof', 'Windows', 'Heating system']
    #%%
    idx_stock = pd.MultiIndex.from_frame(stock_buildings.reset_index()[columns]).unique().sort_values()
    idx_archetypes = archetypes.index.unique().sort_values()
    #%%
    idx_missing = idx_stock.drop(idx_stock.intersection(idx_archetypes))

    missing = {}
    for i in idx_missing:
        if i[1] == 'A':
            missing.update({i: [archetypes['Wall'].min(), archetypes['Floor'].min(), archetypes['Roof'].min(), archetypes['Windows'].min(), '{}-Standard boiler'.format(i[2]), 1]})
        if i[1] == 'G':
            missing.update({i: [archetypes['Wall'].max(), archetypes['Floor'].max(), archetypes['Roof'].max(), archetypes['Windows'].max(), '{}-Standard boiler'.format(i[2]), 1]})
    missing = pd.DataFrame(missing).T.set_axis(['Wall', 'Floor', 'Roof', 'Windows', 'Heating system', 'Weight'], axis=1)
    missing = missing[archetypes.columns]

    #%%
    archetypes_completed = pd.concat((missing, archetypes), axis=0).sort_index()

    ## Merging dataset
    #%%
    new_stock = pd.DataFrame()
    for idx, group_stock in stock_buildings.groupby(columns):
        if archetypes_completed.loc[idx, :].shape[0] > 1 and isinstance(archetypes_completed.loc[idx, :], pd.DataFrame):
            for _, row in archetypes_completed.loc[idx, :].iterrows():
                stock = row['Weight'] * group_stock
                stock = pd.concat((stock, pd.concat([row[archetypes_columns]] * group_stock.shape[0], keys=group_stock.index, axis=1).T), axis=1).set_index(archetypes_columns, append=True)
                new_stock = pd.concat((new_stock, stock), axis=0)
        else:
            row = archetypes_completed.loc[idx, :].squeeze()
            stock = row['Weight'] * group_stock
            stock = pd.concat(
                (stock, pd.concat([row[archetypes_columns]] * group_stock.shape[0], keys=group_stock.index, axis=1).T),
                axis=1).set_index(archetypes_columns, append=True)
            new_stock = pd.concat((new_stock, stock), axis=0)

    new_stock = new_stock.droplevel('Performance')
    new_stock = new_stock.droplevel('Energy')
    new_stock = new_stock.groupby(new_stock.index.names).sum()

    count = 1
    for i in new_stock.index.names:
        count *= new_stock.index.get_level_values(i).unique().shape[0]
    print(count)
    # print(new_stock[new_stock.index.duplicated()])

    new_stock.to_csv('output/buildingstock_sdes2018_{}.csv'.format(number))
