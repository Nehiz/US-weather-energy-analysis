"""
Data quality monitoring and alerting module
"""
import pandas as pd
import logging
from typing import Dict, List
from datetime import datetime, timedelta


class DataQualityMonitor:
    """Monitor data quality and send alerts for anomalies"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.alerts = []
    
    def check_data_completeness(self, df: pd.DataFrame, expected_days: int) -> Dict:
        """Check if data is complete for all cities"""
        results = {
            'status': 'PASS',
            'missing_dates': [],
            'missing_cities': [],
            'completeness_score': 0.0
        }
        
        # Check date range
        date_range = pd.date_range(
            start=df['date'].min(),
            end=df['date'].max(),
            freq='D'
        )
        
        actual_dates = set(df['date'].dt.date)
        expected_dates = set(date_range.date)
        missing_dates = expected_dates - actual_dates
        
        if missing_dates:
            results['status'] = 'WARNING'
            results['missing_dates'] = sorted(missing_dates)
            self.logger.warning(f"Missing {len(missing_dates)} dates in data")
        
        # Calculate completeness score
        expected_records = expected_days * df['city'].nunique()
        actual_records = len(df)
        results['completeness_score'] = (actual_records / expected_records) * 100
        
        return results
    
    def check_data_freshness(self, df: pd.DataFrame, max_age_days: int = 7) -> Dict:
        """Check if data is fresh enough"""
        latest_date = df['date'].max()
        age_days = (datetime.now() - latest_date).days
        
        status = 'PASS' if age_days <= max_age_days else 'FAIL'
        
        if status == 'FAIL':
            alert = f"Data is {age_days} days old (threshold: {max_age_days} days)"
            self.alerts.append(alert)
            self.logger.error(alert)
        
        return {
            'status': status,
            'latest_date': latest_date.strftime('%Y-%m-%d'),
            'age_days': age_days,
            'threshold_days': max_age_days
        }
    
    def check_anomalies(self, df: pd.DataFrame) -> Dict:
        """Detect anomalies in energy consumption"""
        results = {
            'status': 'PASS',
            'anomalies': []
        }
        
        # Check for extreme values using IQR method
        Q1 = df['energy_consumption'].quantile(0.25)
        Q3 = df['energy_consumption'].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        anomalies = df[
            (df['energy_consumption'] < lower_bound) |
            (df['energy_consumption'] > upper_bound)
        ]
        
        if len(anomalies) > 0:
            results['status'] = 'WARNING'
            results['anomalies'] = anomalies[['date', 'city', 'energy_consumption']].to_dict('records')
            self.logger.warning(f"Detected {len(anomalies)} anomalies")
        
        return results
    
    def generate_health_report(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive data health report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_records': len(df),
            'date_range': {
                'start': df['date'].min().strftime('%Y-%m-%d'),
                'end': df['date'].max().strftime('%Y-%m-%d')
            },
            'completeness': self.check_data_completeness(df, expected_days=30),
            'freshness': self.check_data_freshness(df, max_age_days=7),
            'anomalies': self.check_anomalies(df),
            'alerts': self.alerts
        }
        
        # Overall health score
        scores = []
        if report['completeness']['status'] == 'PASS':
            scores.append(report['completeness']['completeness_score'])
        if report['freshness']['status'] == 'PASS':
            scores.append(100)
        if report['anomalies']['status'] == 'PASS':
            scores.append(100)
        
        report['health_score'] = sum(scores) / len(scores) if scores else 0
        
        return report


if __name__ == "__main__":
    # Test the monitor
    import sys
    sys.path.append('..')
    
    logging.basicConfig(level=logging.INFO)
    
    # Load data
    df = pd.read_csv('../data/processed/combined_data.csv', parse_dates=['date'])
    
    # Run monitoring
    monitor = DataQualityMonitor()
    report = monitor.generate_health_report(df)
    
    print("\n" + "="*60)
    print("DATA HEALTH REPORT")
    print("="*60)
    print(f"Health Score: {report['health_score']:.1f}%")
    print(f"Total Records: {report['total_records']}")
    print(f"Date Range: {report['date_range']['start']} to {report['date_range']['end']}")
    print(f"\nCompleteness: {report['completeness']['status']}")
    print(f"Freshness: {report['freshness']['status']}")
    print(f"Anomalies: {report['anomalies']['status']}")
    
    if report['alerts']:
        print("\n⚠️ ALERTS:")
        for alert in report['alerts']:
            print(f"  - {alert}")
