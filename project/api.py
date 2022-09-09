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


def grouped_inputs():
    """Grouped all inputs in the same DataFrame.

    Returns
    -------
    DataFrame
    """
    data = DataFrame(columns=['variables', 'index', 'value'])
    metadata = DataFrame(columns=['variables', 'type', 'name', 'index', 'columns'])
    for key, item in inputs.items():
        i = True

        if isinstance(item, dict):
            metadata = concat((metadata.T, Series({'variables': key, 'type': type(item).__name__})), axis=1).T
            i = False
            item = Series(item)

        if isinstance(item, (float, int)):
            data = concat((data.T, Series({'variables': key, 'value': item})), axis=1).T
            metadata = concat((metadata.T, Series({'variables': key, 'type': type(item).__name__})), axis=1).T

        if isinstance(item, DataFrame):
            metadata = concat((metadata.T, Series({'variables': key, 'type': type(item).__name__,
                                                   'index': item.index.names.copy(),
                                                   'columns': item.columns.names.copy()})), axis=1).T
            i = False
            item = item.stack(item.columns.names)

        if isinstance(item, Series):
            if i:
                metadata = concat((metadata.T, Series({'variables': key, 'type': type(item).__name__, 'name': item.name,
                                                       'index': item.index.names.copy()})), axis=1).T

            if isinstance(item.index, MultiIndex):
                item.index = item.index.to_flat_index()

            item.index = item.index.rename('index')
            df = concat([item.rename('value').reset_index()], keys=[key], names=['variables']).reset_index('variables')
            data = concat((data, df), axis=0)

    data = data.astype({'variables': 'string', 'value': 'float64'})
    data.reset_index(drop=True, inplace=True)
    return data


def degrouped_inputs(data, metadata):
    """Parse aggregate data pandas and return dict fill with several inputs.

    Parameters
    ----------
    data: DataFrame
        Model data input.
    metadata: DataFrame
        Additional information to find out how to parse data.

    Returns
    -------
    dict
    """

    def parse_index(n, index_values):
        if len(n) == 1:
            idx = Index(index_values, name=n[0])
        else:
            idx = MultiIndex.from_tuples(index_values)
            idx.names = n
        return idx

    parsed_input = dict()
    for variables, df in data.groupby('variables'):
        meta = metadata[metadata['variables'] == variables]
        if meta['type'].iloc[0] == 'int':
            parsed_input.update({variables: int(df['value'].iloc[0])})
        elif meta['type'].iloc[0] == 'float':
            parsed_input.update({variables: float(df['value'].iloc[0])})
        elif meta['type'].iloc[0] == 'Series':
            idx = parse_index(meta['index'].iloc[0], df['index'].values)
            parsed_input.update({variables: Series(df['value'].values, name=str(meta['name'].iloc[0]), index=idx)})
        elif meta['type'].iloc[0] == 'DataFrame':
            idx = parse_index(meta['index'].iloc[0] + meta['columns'].iloc[0], df['index'].values)
            parsed_input.update({variables: Series(df['value'].values, name=str(meta['name'].iloc[0]), index=idx).unstack(
                meta['columns'].iloc[0])})

        elif meta['type'].iloc[0] == 'dict':
            parsed_input.update({variables: Series(df['value'].values, index=df['index'].values).to_dict()})

    return parsed_input

print('break')
print('break')
