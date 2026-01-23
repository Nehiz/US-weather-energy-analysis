"""
Pipeline Orchestrator Module for Weather-Energy Analysis
Main controller that coordinates data fetching, processing, and storage
"""

import yaml
import logging
import pandas as pd
from datetime import datetime, timedelta
import os
from pathlib import Path
from typing import Dict, Tuple

from data_fetcher import WeatherEnergyFetcher
from data_processor import DataProcessor


class EnergyAnalysisPipeline:
    """
    Main pipeline orchestrator that coordinates the entire workflow
    Handles daily updates and historical data backfill
    """
    
    def __init__(self, config_path: str = '../config/config.yaml'):
        """
        Initialize pipeline with configuration
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._ensure_directories()
        
        # Initialize components
        self.fetcher = WeatherEnergyFetcher(self.config)
        self.processor = DataProcessor()
        
        self.logger.info("EnergyAnalysisPipeline initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict:
        """
        Load configuration from YAML file
        
        Args:
            config_path: Path to config file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
    
    def _setup_logging(self):
        """Setup logging configuration with both file and console handlers"""
        
        # Create logs directory if it doesn't exist
        log_dir = Path('../logs')
        log_dir.mkdir(exist_ok=True)
        
        # Configure logging
        log_level = getattr(logging, self.config['logging']['level'])
        log_format = self.config['logging']['format']
        
        # Clear any existing handlers
        logging.getLogger().handlers = []
        
        # Setup logging with both file and console output
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler('../logs/pipeline.log'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            '../data/raw',
            '../data/processed',
            '../logs'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def run_daily_update(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Run daily data update - fetches yesterday's data and appends to existing dataset
        
        Returns:
            Tuple of (cleaned_data, quality_report)
        """
        self.logger.info("="*70)
        self.logger.info("STARTING DAILY PIPELINE UPDATE")
        self.logger.info("="*70)
        
        # Get yesterday's date (most recent complete data available)
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        start_date = end_date  # Just fetch one day for daily update
        
        self.logger.info(f"Fetching data for: {start_date}")
        
        try:
            # Fetch new data
            self.logger.info("Step 1: Fetching new data from APIs...")
            new_data = self.fetcher.fetch_all_cities_data(start_date, end_date)
            self.logger.info(f"✓ Fetched {len(new_data)} new records")
            
            # Load existing data if it exists
            processed_file = '../data/processed/combined_data.csv'
            
            if os.path.exists(processed_file):
                self.logger.info("Step 2: Loading existing data...")
                existing_data = pd.read_csv(processed_file, parse_dates=['date'])
                self.logger.info(f"✓ Loaded {len(existing_data)} existing records")
                
                # Remove any duplicate dates before appending (in case of re-runs)
                existing_data = existing_data[existing_data['date'] < pd.to_datetime(end_date)]
                
                # Combine old and new data
                combined_data = pd.concat([existing_data, new_data], ignore_index=True)
                self.logger.info(f"✓ Combined into {len(combined_data)} total records")
            else:
                self.logger.info("Step 2: No existing data found, starting fresh")
                combined_data = new_data
            
            # Clean and process all data
            self.logger.info("Step 3: Cleaning and processing data...")
            cleaned_data = self.processor.clean_data(combined_data)
            self.logger.info(f"✓ Cleaned data: {len(cleaned_data)} records")
            
            # Save processed data
            self.logger.info("Step 4: Saving processed data...")
            cleaned_data.to_csv(processed_file, index=False)
            self.logger.info(f"✓ Saved to: {processed_file}")
            
            # Also save raw new data for backup
            raw_file = f"../data/raw/daily_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            new_data.to_csv(raw_file, index=False)
            self.logger.info(f"✓ Raw data backup: {raw_file}")
            
            # Generate quality report
            self.logger.info("Step 5: Generating quality report...")
            quality_report = self.processor.generate_quality_report(cleaned_data)
            
            # Print summary
            self.logger.info("="*70)
            self.logger.info("DAILY UPDATE COMPLETED SUCCESSFULLY")
            self.logger.info(f"Total records in database: {len(cleaned_data)}")
            self.logger.info(f"Date range: {quality_report['date_range']['start']} to {quality_report['date_range']['end']}")
            self.logger.info(f"Cities covered: {len(quality_report['city_coverage'])}")
            self.logger.info("="*70)
            
            return cleaned_data, quality_report
            
        except Exception as e:
            self.logger.error(f"Daily update failed: {e}")
            self.logger.exception("Full error traceback:")
            raise
    
    def run_historical_backfill(self, days: int = 90) -> Tuple[pd.DataFrame, Dict]:
        """
        Fetch historical data for initial setup or backfill
        
        Args:
            days: Number of days of historical data to fetch (default: 90)
            
        Returns:
            Tuple of (cleaned_data, quality_report)
        """
        self.logger.info("="*70)
        self.logger.info(f"STARTING HISTORICAL BACKFILL - {days} DAYS")
        self.logger.info("="*70)
        
        # Calculate date range (avoid very recent dates that might not have data yet)
        end_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days + 3)).strftime('%Y-%m-%d')
        
        self.logger.info(f"Date range: {start_date} to {end_date}")
        self.logger.info(f"This will fetch approximately {days} days of data")
        
        try:
            # Fetch historical data
            self.logger.info("Step 1: Fetching historical data from APIs...")
            self.logger.info("This may take several minutes due to API rate limiting...")
            
            historical_data = self.fetcher.fetch_all_cities_data(start_date, end_date)
            self.logger.info(f"✓ Fetched {len(historical_data)} historical records")
            
            # Clean and process
            self.logger.info("Step 2: Cleaning and processing data...")
            cleaned_data = self.processor.clean_data(historical_data)
            self.logger.info(f"✓ Cleaned data: {len(cleaned_data)} records")
            
            # Save raw historical data
            self.logger.info("Step 3: Saving data...")
            raw_file = '../data/raw/historical_raw.csv'
            historical_data.to_csv(raw_file, index=False)
            self.logger.info(f"✓ Raw data saved: {raw_file}")
            
            # Save processed data
            processed_file = '../data/processed/combined_data.csv'
            cleaned_data.to_csv(processed_file, index=False)
            self.logger.info(f"✓ Processed data saved: {processed_file}")
            
            # Generate quality report
            self.logger.info("Step 4: Generating quality report...")
            quality_report = self.processor.generate_quality_report(cleaned_data)
            
            # Print detailed summary
            self.logger.info("="*70)
            self.logger.info("HISTORICAL BACKFILL COMPLETED SUCCESSFULLY")
            self.logger.info(f"Total records collected: {len(cleaned_data)}")
            self.logger.info(f"Date range: {quality_report['date_range']['start']} to {quality_report['date_range']['end']}")
            self.logger.info(f"Actual days of data: {quality_report['date_range']['days']}")
            
            self.logger.info("\nCity Coverage:")
            for city, stats in quality_report['city_coverage'].items():
                self.logger.info(f"  {city}: {stats['record_count']} records")
            
            self.logger.info(f"\nData Freshness: {quality_report['data_freshness']['status']}")
            self.logger.info(f"Temperature Outliers: {quality_report['outliers']['temperature_outliers']}")
            self.logger.info(f"Energy Outliers: {quality_report['outliers']['energy_outliers']}")
            self.logger.info("="*70)
            
            # Print the full quality report
            self.processor.print_quality_report(quality_report)
            
            return cleaned_data, quality_report
            
        except Exception as e:
            self.logger.error(f"Historical backfill failed: {e}")
            self.logger.exception("Full error traceback:")
            raise
    
    def get_data_summary(self) -> Dict:
        """
        Get summary of current data in the system
        
        Returns:
            Dictionary with data summary statistics
        """
        processed_file = '../data/processed/combined_data.csv'
        
        if not os.path.exists(processed_file):
            return {
                'status': 'no_data',
                'message': 'No processed data found. Run historical backfill first.'
            }
        
        try:
            data = pd.read_csv(processed_file, parse_dates=['date'])
            
            summary = {
                'status': 'data_exists',
                'total_records': len(data),
                'cities': data['city'].unique().tolist(),
                'date_range': {
                    'start': data['date'].min().strftime('%Y-%m-%d'),
                    'end': data['date'].max().strftime('%Y-%m-%d'),
                    'days': (data['date'].max() - data['date'].min()).days
                },
                'last_updated': datetime.fromtimestamp(
                    os.path.getmtime(processed_file)
                ).strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return summary
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error reading data: {e}'
            }


# Test and demonstration functions
def test_pipeline():
    """Test the pipeline with a small data fetch"""
    print("\n" + "="*70)
    print("TESTING PIPELINE MODULE")
    print("="*70 + "\n")
    
    # Initialize pipeline
    pipeline = EnergyAnalysisPipeline()
    
    # Check current data status
    print("Checking current data status...")
    summary = pipeline.get_data_summary()
    
    if summary['status'] == 'no_data':
        print("No existing data found.")
        print("\nRunning a small 14-day historical backfill for testing...")
        data, report = pipeline.run_historical_backfill(days=14)
    else:
        print(f"Existing data found:")
        print(f"  Records: {summary['total_records']}")
        print(f"  Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        print(f"  Last Updated: {summary['last_updated']}")
        print("\nData already exists. Backfill completed previously.")


if __name__ == "__main__":
    test_pipeline()