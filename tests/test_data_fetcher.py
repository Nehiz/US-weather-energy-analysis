"""
Unit tests for data_fetcher module
"""
import pytest
import pandas as pd
from unittest.mock import Mock, patch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_fetcher import WeatherEnergyFetcher


@pytest.fixture
def sample_config():
    return {
        'cities': [
            {
                'name': 'Test City',
                'state': 'Test State',
                'noaa_station': 'GHCND:TEST123',
                'eia_region': 'TEST'
            }
        ],
        'api': {
            'noaa_base_url': 'https://test.noaa.gov',
            'eia_base_url': 'https://test.eia.gov',
            'request_delay': 0.1,
            'max_retries': 3
        }
    }


def test_fetcher_initialization_without_tokens():
    """Test that fetcher raises error without API tokens"""
    config = {'api': {}}
    with pytest.raises(ValueError):
        WeatherEnergyFetcher(config)


@patch.dict(os.environ, {'NOAA_TOKEN': 'test_token', 'EIA_API_KEY': 'test_key'})
def test_fetcher_initialization_success(sample_config):
    """Test successful fetcher initialization"""
    fetcher = WeatherEnergyFetcher(sample_config)
    assert fetcher.noaa_token == 'test_token'
    assert fetcher.eia_key == 'test_key'


@patch('data_fetcher.requests.get')
@patch.dict(os.environ, {'NOAA_TOKEN': 'test_token', 'EIA_API_KEY': 'test_key'})
def test_fetch_weather_data_timeout(mock_get, sample_config):
    """Test weather data fetch handles timeout"""
    mock_get.side_effect = Exception("Timeout")
    fetcher = WeatherEnergyFetcher(sample_config)
    result = fetcher.fetch_weather_data('GHCND:TEST', '2024-01-01', '2024-01-07')
    assert result is None
