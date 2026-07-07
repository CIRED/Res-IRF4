"""Quick smoke tests for Res-IRF using dummy and mini building stocks.

These tests verify the configuration loading, stock reading, input parsing,
initialization, and calibration pipeline without running the full dynamic
simulation (which requires a comprehensive building stock for heater calibration).

Two stocks are used:
- buildingstock_dummy.csv (~17k rows): the existing dummy with full heterogeneity
- buildingstock_mini.csv (16 rows): a tiny stock for fast tests that still covers
  key dimensions (2 housing types, 3 occupancy statuses, multiple income deciles,
  various thermal performances, and 6 heating systems)
"""

import os
import pytest
import pandas as pd

from project.model import prepare_config, config2inputs, initialize
from project.read_input import read_stock
from project.utils import get_json


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CONFIG_PATH = os.path.join('project', 'config', 'test_quick.json')

HEATING_SYSTEM_MAP = {
    "Oil fuel-Standard boiler": "Oil fuel-Performance boiler",
    "Natural gas-Standard boiler": "Natural gas-Performance boiler",
    "Wood fuel-Standard boiler": "Wood fuel-Performance boiler",
    "Natural gas-Collective boiler": "Natural gas-Performance boiler",
    "Oil fuel-Collective boiler": "Oil fuel-Performance boiler",
    "Electricity-Heat pump air": "Electricity-Heat pump water",
    "Electricity-Performance boiler": "Electricity-Direct electric",
}


def _make_config(stock_path):
    """Build a prepared config dict for a given stock file."""
    raw = get_json(CONFIG_PATH)
    raw['Reference']['building_stock'] = stock_path
    return prepare_config(raw['Reference'])


@pytest.fixture(scope="module")
def config_mini():
    return _make_config('project/input/stock/buildingstock_mini.csv')


@pytest.fixture(scope="module")
def config_dummy():
    return _make_config('project/input/stock/buildingstock_dummy.csv')


@pytest.fixture(scope="module")
def pipeline_mini(config_mini):
    """Run config2inputs + initialize for the mini stock."""
    config, inputs, stock, year, p_h, p_i, taxes = config2inputs(config_mini)
    inputs_dynamics = initialize(inputs, stock, year, taxes, path=None, config=config)
    return config, inputs, stock, year, p_h, p_i, taxes, inputs_dynamics


@pytest.fixture(scope="module")
def pipeline_dummy(config_dummy):
    """Run config2inputs + initialize for the dummy stock."""
    config, inputs, stock, year, p_h, p_i, taxes = config2inputs(config_dummy)
    inputs_dynamics = initialize(inputs, stock, year, taxes, path=None, config=config)
    return config, inputs, stock, year, p_h, p_i, taxes, inputs_dynamics


# ---------------------------------------------------------------------------
# 1. Configuration loading
# ---------------------------------------------------------------------------

class TestConfig:

    def test_prepare_config_returns_dict(self, config_mini):
        assert isinstance(config_mini, dict)

    def test_config_has_required_keys(self, config_mini):
        required = ['start', 'end', 'building_stock', 'simple', 'policies',
                     'energy', 'technical', 'macro', 'renovation', 'switch_heater']
        for key in required:
            assert key in config_mini, f"Missing config key: {key}"

    def test_config_years(self, config_mini):
        assert config_mini['start'] < config_mini['end']

    def test_heating_system_map_applied(self, config_mini):
        hs = config_mini['simple']['heating_system']
        assert 'Electricity-Performance boiler' in hs
        assert hs['Electricity-Performance boiler'] == 'Electricity-Direct electric'


# ---------------------------------------------------------------------------
# 2. Stock reading
# ---------------------------------------------------------------------------

class TestReadStock:

    def test_mini_stock_is_series(self, config_mini):
        stock = read_stock(config_mini)
        assert isinstance(stock, pd.Series)

    def test_mini_stock_positive(self, config_mini):
        stock = read_stock(config_mini)
        assert (stock > 0).all()

    def test_mini_stock_index_levels(self, config_mini):
        stock = read_stock(config_mini)
        expected_levels = {'Existing', 'Occupancy status', 'Income owner',
                           'Income tenant', 'Housing type', 'Heating system',
                           'Wall', 'Floor', 'Roof', 'Windows'}
        assert set(stock.index.names) == expected_levels

    def test_mini_stock_has_both_housing_types(self, config_mini):
        stock = read_stock(config_mini)
        housing = stock.index.get_level_values('Housing type').unique()
        assert 'Single-family' in housing
        assert 'Multi-family' in housing

    def test_dummy_stock_larger_than_mini(self, config_mini, config_dummy):
        stock_mini = read_stock(config_mini)
        stock_dummy = read_stock(config_dummy)
        assert stock_dummy.shape[0] > stock_mini.shape[0]

    def test_heat_pump_split(self, config_mini):
        """read_stock splits 'Electricity-Heat pump' into water and air variants."""
        stock = read_stock(config_mini)
        hs = stock.index.get_level_values('Heating system').unique()
        assert 'Electricity-Heat pump' not in hs
        assert 'Electricity-Heat pump water' in hs


# ---------------------------------------------------------------------------
# 3. config2inputs pipeline
# ---------------------------------------------------------------------------

class TestConfig2Inputs:

    def test_returns_correct_tuple_length(self, pipeline_mini):
        # config, inputs, stock, year, p_h, p_i, taxes, inputs_dynamics
        assert len(pipeline_mini) == 8

    def test_stock_is_series(self, pipeline_mini):
        stock = pipeline_mini[2]
        assert isinstance(stock, pd.Series)

    def test_year_matches_config_start(self, pipeline_mini):
        config, _, _, year, *_ = pipeline_mini
        assert year == config['start']

    def test_policies_are_lists(self, pipeline_mini):
        _, _, _, _, p_h, p_i, taxes, _ = pipeline_mini
        assert isinstance(p_h, list)
        assert isinstance(p_i, list)
        assert isinstance(taxes, list)

    def test_has_heater_and_insulation_policies(self, pipeline_mini):
        _, _, _, _, p_h, p_i, _, _ = pipeline_mini
        assert len(p_h) > 0, "Expected at least one heater policy"
        assert len(p_i) > 0, "Expected at least one insulation policy"

    def test_quintiles_reduces_income_levels(self, pipeline_mini):
        stock = pipeline_mini[2]
        incomes = stock.index.get_level_values('Income owner').unique()
        # quintiles means 5 levels (C1-C5) instead of 10 deciles (D1-D10)
        assert len(incomes) == 5
        assert all(i.startswith('C') for i in incomes)


# ---------------------------------------------------------------------------
# 4. Initialization
# ---------------------------------------------------------------------------

class TestInitialize:

    def test_buildings_object_created(self, pipeline_mini):
        inputs_dynamics = pipeline_mini[-1]
        buildings = inputs_dynamics['buildings']
        assert hasattr(buildings, 'stock')
        assert hasattr(buildings, 'first_year')

    def test_buildings_stock_matches(self, pipeline_mini):
        stock = pipeline_mini[2]
        buildings = inputs_dynamics = pipeline_mini[-1]['buildings']
        assert buildings.stock.shape[0] == stock.shape[0]

    def test_energy_prices_is_dataframe(self, pipeline_mini):
        inputs_dynamics = pipeline_mini[-1]
        assert isinstance(inputs_dynamics['energy_prices'], pd.DataFrame)

    def test_energy_prices_cover_simulation_period(self, pipeline_mini):
        config = pipeline_mini[0]
        prices = pipeline_mini[-1]['energy_prices']
        assert config['start'] in prices.index
        # end-1 because the model iterates up to end-1
        assert config['end'] - 1 in prices.index

    def test_cost_heater_not_empty(self, pipeline_mini):
        inputs_dynamics = pipeline_mini[-1]
        assert inputs_dynamics['cost_heater'].shape[0] > 0

    def test_cost_insulation_not_empty(self, pipeline_mini):
        inputs_dynamics = pipeline_mini[-1]
        assert inputs_dynamics['cost_insulation'].shape[0] > 0


# ---------------------------------------------------------------------------
# 5. Calibration (consumption)
# ---------------------------------------------------------------------------

class TestCalibration:

    def test_calibration_consumption_runs(self, pipeline_mini):
        inputs_dynamics = pipeline_mini[-1]
        buildings = inputs_dynamics['buildings']
        buildings.calibration_consumption(
            inputs_dynamics['energy_prices'].loc[buildings.first_year, :],
            inputs_dynamics['consumption_ini'],
            inputs_dynamics['health_cost_income'],
            inputs_dynamics['health_cost_dpe'],
        )
        assert buildings.coefficient_global is not None
        assert buildings.coefficient_global > 0

    def test_calibration_on_dummy_stock(self, pipeline_dummy):
        inputs_dynamics = pipeline_dummy[-1]
        buildings = inputs_dynamics['buildings']
        buildings.calibration_consumption(
            inputs_dynamics['energy_prices'].loc[buildings.first_year, :],
            inputs_dynamics['consumption_ini'],
            inputs_dynamics['health_cost_income'],
            inputs_dynamics['health_cost_dpe'],
        )
        assert buildings.coefficient_global > 0


# ---------------------------------------------------------------------------
# 6. Cross-stock consistency
# ---------------------------------------------------------------------------

class TestCrossStock:

    def test_same_heating_systems(self, pipeline_mini, pipeline_dummy):
        """Both stocks should yield the same set of heating systems after processing."""
        hs_mini = set(pipeline_mini[2].index.get_level_values('Heating system').unique())
        hs_dummy = set(pipeline_dummy[2].index.get_level_values('Heating system').unique())
        assert hs_mini == hs_dummy

    def test_same_index_level_names(self, pipeline_mini, pipeline_dummy):
        assert pipeline_mini[2].index.names == pipeline_dummy[2].index.names
