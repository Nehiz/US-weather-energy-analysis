"""
Data Fetcher Module for Weather-Energy Analysis Pipeline
Handles API calls to NOAA (weather) and EIA (energy) data sources
"""

import requests
import pandas as pd
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class WeatherEnergyFetcher:
    """
    Fetches weather data from NOAA and energy data from EIA APIs
    Combines data for multiple cities with error handling and rate limiting
    """
    
    def __init__(self, config: Dict):
        """
        Initialize fetcher with configuration
        
        Args:
            config: Dictionary containing API URLs, cities, and settings
        """
        self.config = config
        
        # Load API credentials from environment variables
        self.noaa_token = os.getenv('NOAA_TOKEN')
        self.eia_key = os.getenv('EIA_API_KEY')
        
        # Validate that API keys exist
        if not self.noaa_token:
            raise ValueError("NOAA_TOKEN not found in .env file")
        if not self.eia_key:
            raise ValueError("EIA_API_KEY not found in .env file")
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info("WeatherEnergyFetcher initialized successfully")
    
    def fetch_weather_data(self, station_id: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch weather data from NOAA Climate Data Online API
        
        Args:
            station_id: NOAA station identifier (e.g., 'GHCND:USW00094728')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with columns: date, TMAX, TMIN (temperatures in Fahrenheit)
            None if request fails
        """
        url = f"{self.config['api']['noaa_base_url']}/data"
        
        headers = {'token': self.noaa_token}
        params = {
            'datasetid': 'GHCND',  # Global Historical Climatology Network Daily
            'stationid': station_id,
            'startdate': start_date,
            'enddate': end_date,
            'datatypeid': 'TMAX,TMIN',  # Maximum and minimum temperature
            'limit': 1000,  # Max records per request
            'units': 'standard'  # Standard units
        }
        
        try:
            self.logger.info(f"Fetching weather data for station {station_id}")
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Check if we got results
            if 'results' not in data or len(data['results']) == 0:
                self.logger.warning(f"No weather data found for station {station_id}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data['results'])
            
            # IMPORTANT: NOAA returns temperature in tenths of degrees Celsius
            # Convert to Fahrenheit: (Celsius * 9/5) + 32
            df['value_fahrenheit'] = (df['value'] / 10 * 9/5) + 32
            df['date'] = pd.to_datetime(df['date'])
            
            # Pivot to get TMAX and TMIN as separate columns
            df_pivot = df.pivot_table(
                index='date', 
                columns='datatype', 
                values='value_fahrenheit'
            ).reset_index()
            
            # Clean up column names
            df_pivot.columns.name = None
            
            self.logger.info(f"Successfully fetched {len(df_pivot)} weather records")
            return df_pivot
            
        except requests.exceptions.Timeout:
            self.logger.error(f"Timeout fetching weather data for {station_id}")
            return None
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to fetch weather data for {station_id}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error fetching weather data: {e}")
            return None
    
    def fetch_energy_data(self, region_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch energy consumption data from EIA API
        
        Args:
            region_code: EIA region code (e.g., 'NYIS' for New York)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with columns: date, energy_consumption (in megawatthours)
            None if request fails
        """
        url = self.config['api']['eia_base_url']
        
        params = {
            'api_key': self.eia_key,
            'frequency': 'daily',
            'data[0]': 'value',  # Get demand values
            'facets[respondent][]': region_code,
            'start': start_date,
            'end': end_date,
            'sort[0][column]': 'period',
            'sort[0][direction]': 'asc',
            'offset': 0,
            'length': 5000
        }
        
        try:
            self.logger.info(f"Fetching energy data for region {region_code}")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Check if we got results
            if 'response' not in data or 'data' not in data['response']:
                self.logger.warning(f"No energy data found for region {region_code}")
                return None
            
            if len(data['response']['data']) == 0:
                self.logger.warning(f"Empty energy dataset for region {region_code}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data['response']['data'])
            
            # Clean and rename columns
            df['date'] = pd.to_datetime(df['period'])
            df = df.rename(columns={'value': 'energy_consumption'})
            
            # Convert energy_consumption to numeric, handling any non-numeric values
            df['energy_consumption'] = pd.to_numeric(df['energy_consumption'], errors='coerce')
            
            # Select only needed columns
            df = df[['date', 'energy_consumption']]
            
            # Remove duplicates and sort
            df = df.drop_duplicates(subset=['date']).sort_values('date')
            
            self.logger.info(f"Successfully fetched {len(df)} energy records")
            return df
            
        except requests.exceptions.Timeout:
            self.logger.error(f"Timeout fetching energy data for {region_code}")
            return None
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to fetch energy data for {region_code}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error fetching energy data: {e}")
            return None
    
    def fetch_city_data(self, city_config: Dict, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch combined weather and energy data for a single city
        
        Args:
            city_config: Dictionary with city name, station ID, region code
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with weather and energy data merged by date
            None if either API fails
        """
        city_name = city_config['name']
        self.logger.info(f"Fetching data for {city_name}")
        
        # Fetch weather data
        weather_df = self.fetch_weather_data(
            city_config['noaa_station'], 
            start_date, 
            end_date
        )
        
        # Respect API rate limits - wait between requests
        time.sleep(self.config['api']['request_delay'])
        
        # Fetch energy data
        energy_df = self.fetch_energy_data(
            city_config['eia_region'], 
            start_date, 
            end_date
        )
        
        # Check if both fetches succeeded
        if weather_df is None:
            self.logger.error(f"Failed to fetch weather data for {city_name}")
            return None
        
        if energy_df is None:
            self.logger.error(f"Failed to fetch energy data for {city_name}")
            return None
        
        # Merge weather and energy data on date
        combined_df = pd.merge(
            weather_df, 
            energy_df, 
            on='date', 
            how='inner'  # Only keep dates with both weather and energy data
        )
        
        # Add city information
        combined_df['city'] = city_name
        combined_df['state'] = city_config['state']
        
        self.logger.info(f"Successfully combined data for {city_name}: {len(combined_df)} records")
        return combined_df
    
    def fetch_all_cities_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch data for all configured cities
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Combined DataFrame with all cities' data
            
        Raises:
            ValueError: If no data could be fetched for any city
        """
        all_data = []
        failed_cities = []
        
        self.logger.info(f"Starting data fetch for {len(self.config['cities'])} cities")
        
        for city_config in self.config['cities']:
            try:
                city_data = self.fetch_city_data(city_config, start_date, end_date)
                
                if city_data is not None:
                    all_data.append(city_data)
                    self.logger.info(f"✓ {city_config['name']}: {len(city_data)} records")
                else:
                    failed_cities.append(city_config['name'])
                    self.logger.warning(f"✗ {city_config['name']}: No data retrieved")
                
                # Rate limiting between cities
                time.sleep(self.config['api']['request_delay'])
                
            except Exception as e:
                failed_cities.append(city_config['name'])
                self.logger.error(f"Error processing {city_config['name']}: {e}")
        
        # Check if we got any data
        if not all_data:
            error_msg = f"No data could be fetched for any city. Failed cities: {failed_cities}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Combine all city data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        self.logger.info(f"Data fetch complete: {len(combined_data)} total records from {len(all_data)} cities")
        if failed_cities:
            self.logger.warning(f"Failed to fetch data for: {', '.join(failed_cities)}")
        
        return combined_data


# Test function - can be run directly
if __name__ == "__main__":
    import yaml
    
    # Setup basic logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    with open('../config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create fetcher instance
    fetcher = WeatherEnergyFetcher(config)
    
    # Test with last 7 days
    end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    print(f"\nTesting data fetch from {start_date} to {end_date}")
    print("=" * 60)
    
    # Fetch data for all cities
    data = fetcher.fetch_all_cities_data(start_date, end_date)
    
    print(f"\n✅ Successfully fetched {len(data)} records")
    print(f"Cities: {data['city'].unique()}")
    print(f"\nSample data:")
    print(data.head())