"""
Data Processor Module for Weather-Energy Analysis Pipeline
Handles data cleaning, validation, quality checks, and feature engineering
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Tuple


class DataProcessor:
    """
    Processes raw weather-energy data with cleaning, validation, and quality checks
    Creates derived features for analysis
    """
    
    def __init__(self):
        """Initialize the data processor with logging"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("DataProcessor initialized")
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate the combined weather-energy dataset
        
        Steps:
        1. Remove duplicates
        2. Handle missing values
        3. Validate temperature ranges
        4. Validate energy consumption
        5. Create derived features
        
        Args:
            df: Raw combined DataFrame with weather and energy data
            
        Returns:
            Cleaned DataFrame with derived features
        """
        self.logger.info(f"Starting data cleaning. Input: {len(df)} records")
        
        # Make a copy to avoid modifying original
        cleaned_df = df.copy()
        
        # 1. Remove duplicate records (same date and city)
        initial_count = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates(subset=['date', 'city'])
        duplicates_removed = initial_count - len(cleaned_df)
        if duplicates_removed > 0:
            self.logger.info(f"Removed {duplicates_removed} duplicate records")
        
        # 2. Handle missing values
        missing_before = cleaned_df.isnull().sum().sum()
        if missing_before > 0:
            self.logger.warning(f"Found {missing_before} missing values")
        
        # Drop rows with missing critical values
        cleaned_df = cleaned_df.dropna(subset=['TMAX', 'TMIN', 'energy_consumption'])
        missing_removed = initial_count - len(cleaned_df) - duplicates_removed
        if missing_removed > 0:
            self.logger.info(f"Removed {missing_removed} records with missing values")
        
        # 3. Validate temperature ranges (-50°F to 130°F is reasonable)
        temp_mask = (
            (cleaned_df['TMAX'] >= -50) & (cleaned_df['TMAX'] <= 130) &
            (cleaned_df['TMIN'] >= -50) & (cleaned_df['TMIN'] <= 130) &
            (cleaned_df['TMAX'] >= cleaned_df['TMIN'])  # Max should be >= Min
        )
        
        invalid_temps = (~temp_mask).sum()
        if invalid_temps > 0:
            self.logger.warning(f"Found {invalid_temps} records with invalid temperatures")
            cleaned_df = cleaned_df[temp_mask]
        
        # 4. Validate energy consumption (must be positive)
        energy_mask = cleaned_df['energy_consumption'] > 0
        invalid_energy = (~energy_mask).sum()
        if invalid_energy > 0:
            self.logger.warning(f"Found {invalid_energy} records with invalid energy values")
            cleaned_df = cleaned_df[energy_mask]
        
        # 5. Create derived features
        self.logger.info("Creating derived features")
        
        # Average temperature
        cleaned_df['avg_temp'] = (cleaned_df['TMAX'] + cleaned_df['TMIN']) / 2
        
        # Temperature range (daily variation)
        cleaned_df['temp_range'] = cleaned_df['TMAX'] - cleaned_df['TMIN']
        
        # Day of week (Monday=0, Sunday=6)
        cleaned_df['day_of_week'] = cleaned_df['date'].dt.day_name()
        cleaned_df['is_weekend'] = cleaned_df['date'].dt.dayofweek >= 5
        
        # Month and season for seasonal analysis
        cleaned_df['month'] = cleaned_df['date'].dt.month
        cleaned_df['season'] = cleaned_df['month'].apply(self._get_season)
        
        # Temperature categories for heatmap analysis
        cleaned_df['temp_category'] = pd.cut(
            cleaned_df['avg_temp'],
            bins=[-np.inf, 50, 60, 70, 80, 90, np.inf],
            labels=['<50°F', '50-60°F', '60-70°F', '70-80°F', '80-90°F', '>90°F']
        )
        
        # Calculate percentage change in energy consumption (day-over-day by city)
        cleaned_df = cleaned_df.sort_values(['city', 'date'])
        cleaned_df['energy_pct_change'] = cleaned_df.groupby('city')['energy_consumption'].pct_change() * 100
        
        # Reset index
        cleaned_df = cleaned_df.reset_index(drop=True)
        
        self.logger.info(f"Data cleaning complete. Output: {len(cleaned_df)} records")
        self.logger.info(f"Total records removed: {initial_count - len(cleaned_df)}")
        
        return cleaned_df
    
    def _get_season(self, month: int) -> str:
        """
        Convert month number to season name
        
        Args:
            month: Month number (1-12)
            
        Returns:
            Season name
        """
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    def generate_quality_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive data quality metrics
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            Dictionary with quality metrics
        """
        self.logger.info("Generating data quality report")
        
        report = {
            'total_records': len(df),
            'date_range': {
                'start': df['date'].min().strftime('%Y-%m-%d'),
                'end': df['date'].max().strftime('%Y-%m-%d'),
                'days': (df['date'].max() - df['date'].min()).days
            },
            'missing_values': self._check_missing_values(df),
            'outliers': self._detect_outliers(df),
            'data_freshness': self._check_freshness(df),
            'city_coverage': self._check_city_coverage(df),
            'temperature_stats': self._get_temperature_stats(df),
            'energy_stats': self._get_energy_stats(df)
        }
        
        self.logger.info("Quality report generated successfully")
        return report
    
    def _check_missing_values(self, df: pd.DataFrame) -> Dict:
        """Check for missing values in the dataset"""
        missing = df.isnull().sum()
        missing_dict = missing[missing > 0].to_dict()
        
        return {
            'total_missing': missing.sum(),
            'columns_with_missing': missing_dict,
            'completeness_pct': ((1 - missing.sum() / (len(df) * len(df.columns))) * 100)
        }
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict:
        """
        Detect temperature and energy outliers using multiple methods
        """
        outliers = {
            'temperature_outliers': 0,
            'energy_outliers': 0,
            'outlier_details': []
        }
        
        # Temperature outliers (outside reasonable range)
        temp_outliers_mask = (
            (df['TMAX'] > 130) | (df['TMAX'] < -50) |
            (df['TMIN'] > 130) | (df['TMIN'] < -50)
        )
        outliers['temperature_outliers'] = temp_outliers_mask.sum()
        
        # Energy outliers using IQR method (per city to account for size differences)
        energy_outliers = []
        for city in df['city'].unique():
            city_data = df[df['city'] == city]['energy_consumption']
            Q1 = city_data.quantile(0.25)
            Q3 = city_data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            city_outliers = ((city_data < lower_bound) | (city_data > upper_bound)).sum()
            if city_outliers > 0:
                energy_outliers.append({
                    'city': city,
                    'count': city_outliers,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                })
        
        outliers['energy_outliers'] = sum(item['count'] for item in energy_outliers)
        outliers['outlier_details'] = energy_outliers
        
        return outliers
    
    def _check_freshness(self, df: pd.DataFrame) -> Dict:
        """
        Check how recent the data is
        """
        if df.empty:
            return {
                'status': 'no_data',
                'latest_date': None,
                'days_old': None,
                'is_fresh': False
            }
        
        latest_date = df['date'].max()
        days_old = (pd.Timestamp.now() - latest_date).days
        
        # Define freshness criteria
        if days_old <= 1:
            status = 'fresh'
            is_fresh = True
        elif days_old <= 3:
            status = 'recent'
            is_fresh = True
        elif days_old <= 7:
            status = 'acceptable'
            is_fresh = False
        else:
            status = 'stale'
            is_fresh = False
        
        return {
            'status': status,
            'latest_date': latest_date.strftime('%Y-%m-%d'),
            'days_old': days_old,
            'is_fresh': is_fresh
        }
    
    def _check_city_coverage(self, df: pd.DataFrame) -> Dict:
        """
        Check data coverage for each city
        """
        coverage = {}
        
        for city in df['city'].unique():
            city_data = df[df['city'] == city]
            coverage[city] = {
                'record_count': len(city_data),
                'date_range': {
                    'start': city_data['date'].min().strftime('%Y-%m-%d'),
                    'end': city_data['date'].max().strftime('%Y-%m-%d')
                },
                'avg_temp': round(city_data['avg_temp'].mean(), 2),
                'avg_energy': round(city_data['energy_consumption'].mean(), 2)
            }
        
        return coverage
    
    def _get_temperature_stats(self, df: pd.DataFrame) -> Dict:
        """
        Calculate temperature statistics
        """
        return {
            'avg_temp_overall': round(df['avg_temp'].mean(), 2),
            'max_temp_recorded': round(df['TMAX'].max(), 2),
            'min_temp_recorded': round(df['TMIN'].min(), 2),
            'avg_daily_range': round(df['temp_range'].mean(), 2),
            'temp_std_dev': round(df['avg_temp'].std(), 2)
        }
    
    def _get_energy_stats(self, df: pd.DataFrame) -> Dict:
        """
        Calculate energy consumption statistics
        """
        return {
            'avg_consumption': round(df['energy_consumption'].mean(), 2),
            'max_consumption': round(df['energy_consumption'].max(), 2),
            'min_consumption': round(df['energy_consumption'].min(), 2),
            'total_consumption': round(df['energy_consumption'].sum(), 2),
            'std_dev': round(df['energy_consumption'].std(), 2)
        }
    
    def print_quality_report(self, report: Dict):
        """
        Print formatted quality report to console
        
        Args:
            report: Quality report dictionary
        """
        print("\n" + "="*70)
        print("DATA QUALITY REPORT")
        print("="*70)
        
        print(f"\n📊 OVERVIEW:")
        print(f"  Total Records: {report['total_records']}")
        print(f"  Date Range: {report['date_range']['start']} to {report['date_range']['end']}")
        print(f"  Days of Data: {report['date_range']['days']}")
        
        print(f"\n✓ DATA FRESHNESS:")
        fresh = report['data_freshness']
        print(f"  Status: {fresh['status'].upper()}")
        print(f"  Latest Data: {fresh['latest_date']}")
        print(f"  Age: {fresh['days_old']} days old")
        
        print(f"\n📍 CITY COVERAGE:")
        for city, stats in report['city_coverage'].items():
            print(f"  {city}: {stats['record_count']} records")
            print(f"    Avg Temp: {stats['avg_temp']}°F | Avg Energy: {stats['avg_energy']:,.0f} MWh")
        
        print(f"\n⚠️  OUTLIERS:")
        print(f"  Temperature Outliers: {report['outliers']['temperature_outliers']}")
        print(f"  Energy Outliers: {report['outliers']['energy_outliers']}")
        
        print(f"\n🌡️  TEMPERATURE STATS:")
        temp_stats = report['temperature_stats']
        print(f"  Average: {temp_stats['avg_temp_overall']}°F")
        print(f"  Range: {temp_stats['min_temp_recorded']}°F to {temp_stats['max_temp_recorded']}°F")
        print(f"  Avg Daily Variation: {temp_stats['avg_daily_range']}°F")
        
        print(f"\n⚡ ENERGY STATS:")
        energy_stats = report['energy_stats']
        print(f"  Average Consumption: {energy_stats['avg_consumption']:,.0f} MWh")
        print(f"  Range: {energy_stats['min_consumption']:,.0f} to {energy_stats['max_consumption']:,.0f} MWh")
        print(f"  Total Consumption: {energy_stats['total_consumption']:,.0f} MWh")
        
        print("\n" + "="*70 + "\n")


# Test function
if __name__ == "__main__":
    import yaml
    from data_fetcher import WeatherEnergyFetcher
    from datetime import timedelta
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Testing Data Processor Module")
    print("="*70)
    
    # Load configuration
    with open('../config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Fetch some test data
    fetcher = WeatherEnergyFetcher(config)
    processor = DataProcessor()
    
    # Get last 14 days of data
    end_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=17)).strftime('%Y-%m-%d')
    
    print(f"\nFetching test data from {start_date} to {end_date}...")
    raw_data = fetcher.fetch_all_cities_data(start_date, end_date)
    
    print(f"\n✓ Fetched {len(raw_data)} raw records")
    
    # Clean the data
    print("\nCleaning data...")
    cleaned_data = processor.clean_data(raw_data)
    
    print(f"✓ Cleaned data: {len(cleaned_data)} records")
    
    # Generate quality report
    print("\nGenerating quality report...")
    quality_report = processor.generate_quality_report(cleaned_data)
    
    # Print the report
    processor.print_quality_report(quality_report)
    
    # Show sample of cleaned data with derived features
    print("Sample of cleaned data with derived features:")
    print(cleaned_data[['date', 'city', 'avg_temp', 'temp_category', 'day_of_week', 
                        'is_weekend', 'energy_consumption']].head(10))