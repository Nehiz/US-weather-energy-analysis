"""
Analysis Module for Weather-Energy Analysis Pipeline
Provides statistical analysis and correlation calculations
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import logging


class EnergyAnalysis:
    """
    Performs statistical analysis on weather-energy data
    Calculates correlations, regression statistics, and usage patterns
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize analysis with dataset
        
        Args:
            data: Cleaned DataFrame with weather and energy data
        """
        self.data = data.copy()
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"EnergyAnalysis initialized with {len(data)} records")
    
    def calculate_correlations(self) -> pd.DataFrame:
        """
        Calculate temperature-energy correlations for each city
        
        Returns:
            DataFrame with correlation statistics by city
        """
        self.logger.info("Calculating correlations by city")
        
        correlations = []
        
        for city in self.data['city'].unique():
            city_data = self.data[self.data['city'] == city].copy()
            
            # Calculate correlations
            temp_energy_corr = city_data['avg_temp'].corr(city_data['energy_consumption'])
            tmax_energy_corr = city_data['TMAX'].corr(city_data['energy_consumption'])
            tmin_energy_corr = city_data['TMIN'].corr(city_data['energy_consumption'])
            
            # Calculate correlation with temperature range
            temp_range_corr = city_data['temp_range'].corr(city_data['energy_consumption'])
            
            correlations.append({
                'city': city,
                'avg_temp_correlation': round(temp_energy_corr, 4),
                'tmax_correlation': round(tmax_energy_corr, 4),
                'tmin_correlation': round(tmin_energy_corr, 4),
                'temp_range_correlation': round(temp_range_corr, 4),
                'sample_size': len(city_data),
                'avg_temperature': round(city_data['avg_temp'].mean(), 2),
                'avg_energy': round(city_data['energy_consumption'].mean(), 2)
            })
        
        corr_df = pd.DataFrame(correlations)
        self.logger.info(f"Correlations calculated for {len(corr_df)} cities")
        
        return corr_df
    
    def calculate_overall_correlation(self) -> Dict:
        """
        Calculate overall correlation across all cities combined
        
        Returns:
            Dictionary with overall correlation statistics
        """
        self.logger.info("Calculating overall correlation statistics")
        
        # Overall correlation (all cities combined)
        overall_corr = self.data['avg_temp'].corr(self.data['energy_consumption'])
        
        # Correlation by season
        season_corr = {}
        for season in self.data['season'].unique():
            season_data = self.data[self.data['season'] == season]
            if len(season_data) > 1:
                season_corr[season] = round(
                    season_data['avg_temp'].corr(season_data['energy_consumption']), 4
                )
        
        # Weekend vs weekday correlation
        weekend_data = self.data[self.data['is_weekend'] == True]
        weekday_data = self.data[self.data['is_weekend'] == False]
        
        weekend_corr = None
        weekday_corr = None
        
        if len(weekend_data) > 1:
            weekend_corr = round(
                weekend_data['avg_temp'].corr(weekend_data['energy_consumption']), 4
            )
        
        if len(weekday_data) > 1:
            weekday_corr = round(
                weekday_data['avg_temp'].corr(weekday_data['energy_consumption']), 4
            )
        
        return {
            'overall_correlation': round(overall_corr, 4),
            'season_correlations': season_corr,
            'weekend_correlation': weekend_corr,
            'weekday_correlation': weekday_corr,
            'total_observations': len(self.data)
        }
    
    def calculate_regression_stats(self, x_col: str = 'avg_temp', 
                                   y_col: str = 'energy_consumption') -> Dict:
        """
        Calculate linear regression statistics
        
        Args:
            x_col: Independent variable column name
            y_col: Dependent variable column name
            
        Returns:
            Dictionary with regression statistics
        """
        self.logger.info(f"Calculating regression: {y_col} ~ {x_col}")
        
        # Get data and remove NaN values
        x = self.data[x_col]
        y = self.data[y_col]
        
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[valid_mask]
        y_clean = y[valid_mask]
        
        # Calculate linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
        
        # Calculate predicted values for residual analysis
        y_predicted = slope * x_clean + intercept
        residuals = y_clean - y_predicted
        
        # Calculate additional statistics
        mean_squared_error = np.mean(residuals ** 2)
        root_mean_squared_error = np.sqrt(mean_squared_error)
        
        regression_stats = {
            'slope': round(slope, 2),
            'intercept': round(intercept, 2),
            'r_squared': round(r_value ** 2, 4),
            'correlation': round(r_value, 4),
            'p_value': round(p_value, 6),
            'std_error': round(std_err, 2),
            'rmse': round(root_mean_squared_error, 2),
            'equation': f'y = {slope:.2f}x + {intercept:.2f}',
            'interpretation': self._interpret_correlation(r_value),
            'sample_size': len(x_clean)
        }
        
        self.logger.info(f"Regression R² = {regression_stats['r_squared']}")
        
        return regression_stats
    
    def _interpret_correlation(self, r_value: float) -> str:
        """
        Interpret correlation coefficient strength
        
        Args:
            r_value: Correlation coefficient
            
        Returns:
            Text interpretation
        """
        abs_r = abs(r_value)
        
        if abs_r >= 0.9:
            strength = "Very Strong"
        elif abs_r >= 0.7:
            strength = "Strong"
        elif abs_r >= 0.5:
            strength = "Moderate"
        elif abs_r >= 0.3:
            strength = "Weak"
        else:
            strength = "Very Weak"
        
        direction = "Positive" if r_value > 0 else "Negative"
        
        return f"{strength} {direction}"
    
    def get_usage_patterns(self) -> pd.DataFrame:
        """
        Calculate average energy usage patterns by temperature category and day of week
        
        Returns:
            Pivot table with usage patterns
        """
        self.logger.info("Calculating usage patterns for heatmap")
        
        # Group by temperature category and day of week
        pattern_data = self.data.groupby(
            ['temp_category', 'day_of_week']
        )['energy_consumption'].mean().reset_index()
        
        # Create pivot table
        pattern_pivot = pattern_data.pivot(
            index='temp_category', 
            columns='day_of_week', 
            values='energy_consumption'
        )
        
        # Reorder columns (days of week)
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        existing_days = [day for day in day_order if day in pattern_pivot.columns]
        pattern_pivot = pattern_pivot[existing_days]
        
        # Reorder rows (temperature categories)
        temp_order = ['<50°F', '50-60°F', '60-70°F', '70-80°F', '80-90°F', '>90°F']
        existing_temps = [temp for temp in temp_order if temp in pattern_pivot.index]
        pattern_pivot = pattern_pivot.reindex(existing_temps)
        
        self.logger.info("Usage patterns calculated")
        
        return pattern_pivot
    
    def get_city_usage_patterns(self, city: str) -> pd.DataFrame:
        """
        Calculate usage patterns for a specific city
        
        Args:
            city: City name
            
        Returns:
            Pivot table with usage patterns for the city
        """
        city_data = self.data[self.data['city'] == city].copy()
        
        if city_data.empty:
            self.logger.warning(f"No data found for city: {city}")
            return pd.DataFrame()
        
        # Group by temperature category and day of week
        pattern_data = city_data.groupby(
            ['temp_category', 'day_of_week']
        )['energy_consumption'].mean().reset_index()
        
        # Create pivot table
        pattern_pivot = pattern_data.pivot(
            index='temp_category', 
            columns='day_of_week', 
            values='energy_consumption'
        )
        
        # Reorder columns and rows
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        existing_days = [day for day in day_order if day in pattern_pivot.columns]
        pattern_pivot = pattern_pivot[existing_days]
        
        temp_order = ['<50°F', '50-60°F', '60-70°F', '70-80°F', '80-90°F', '>90°F']
        existing_temps = [temp for temp in temp_order if temp in pattern_pivot.index]
        pattern_pivot = pattern_pivot.reindex(existing_temps)
        
        return pattern_pivot
    
    def get_weekend_vs_weekday_analysis(self) -> Dict:
        """
        Compare weekend vs weekday energy consumption patterns
        
        Returns:
            Dictionary with comparison statistics
        """
        self.logger.info("Analyzing weekend vs weekday patterns")
        
        weekend_data = self.data[self.data['is_weekend'] == True]
        weekday_data = self.data[self.data['is_weekend'] == False]
        
        comparison = {
            'weekend_avg': round(weekend_data['energy_consumption'].mean(), 2),
            'weekday_avg': round(weekday_data['energy_consumption'].mean(), 2),
            'weekend_median': round(weekend_data['energy_consumption'].median(), 2),
            'weekday_median': round(weekday_data['energy_consumption'].median(), 2),
            'weekend_std': round(weekend_data['energy_consumption'].std(), 2),
            'weekday_std': round(weekday_data['energy_consumption'].std(), 2),
            'weekend_count': len(weekend_data),
            'weekday_count': len(weekday_data)
        }
        
        # Calculate percentage difference
        if comparison['weekday_avg'] > 0:
            pct_diff = ((comparison['weekend_avg'] - comparison['weekday_avg']) / 
                       comparison['weekday_avg'] * 100)
            comparison['percentage_difference'] = round(pct_diff, 2)
        else:
            comparison['percentage_difference'] = None
        
        # Statistical test (t-test)
        if len(weekend_data) > 1 and len(weekday_data) > 1:
            t_stat, p_value = stats.ttest_ind(
                weekend_data['energy_consumption'],
                weekday_data['energy_consumption']
            )
            comparison['t_statistic'] = round(t_stat, 4)
            comparison['p_value'] = round(p_value, 6)
            comparison['significant_difference'] = p_value < 0.05
        
        return comparison
    
    def get_city_statistics(self) -> pd.DataFrame:
        """
        Get comprehensive statistics for each city
        
        Returns:
            DataFrame with city-level statistics
        """
        self.logger.info("Calculating city statistics")
        
        city_stats = []
        
        for city in self.data['city'].unique():
            city_data = self.data[self.data['city'] == city]
            
            stats_dict = {
                'city': city,
                'record_count': len(city_data),
                'avg_temp': round(city_data['avg_temp'].mean(), 2),
                'min_temp': round(city_data['avg_temp'].min(), 2),
                'max_temp': round(city_data['avg_temp'].max(), 2),
                'avg_energy': round(city_data['energy_consumption'].mean(), 2),
                'min_energy': round(city_data['energy_consumption'].min(), 2),
                'max_energy': round(city_data['energy_consumption'].max(), 2),
                'energy_std': round(city_data['energy_consumption'].std(), 2),
                'temp_energy_correlation': round(
                    city_data['avg_temp'].corr(city_data['energy_consumption']), 4
                )
            }
            
            city_stats.append(stats_dict)
        
        return pd.DataFrame(city_stats)
    
    def print_analysis_summary(self):
        """Print comprehensive analysis summary"""
        print("\n" + "="*70)
        print("COMPREHENSIVE ANALYSIS SUMMARY")
        print("="*70)
        
        # Overall correlation
        overall_stats = self.calculate_overall_correlation()
        print(f"\n📊 OVERALL CORRELATION:")
        print(f"  Temperature vs Energy: {overall_stats['overall_correlation']}")
        print(f"  Total Observations: {overall_stats['total_observations']}")
        
        if overall_stats['season_correlations']:
            print(f"\n  By Season:")
            for season, corr in overall_stats['season_correlations'].items():
                print(f"    {season}: {corr}")
        
        if overall_stats['weekend_correlation'] and overall_stats['weekday_correlation']:
            print(f"\n  Weekend: {overall_stats['weekend_correlation']}")
            print(f"  Weekday: {overall_stats['weekday_correlation']}")
        
        # Regression statistics
        regression = self.calculate_regression_stats()
        print(f"\n📈 REGRESSION ANALYSIS:")
        print(f"  Equation: {regression['equation']}")
        print(f"  R-squared: {regression['r_squared']}")
        print(f"  Interpretation: {regression['interpretation']}")
        print(f"  P-value: {regression['p_value']}")
        
        # City correlations
        city_corrs = self.calculate_correlations()
        print(f"\n🏙️  CORRELATION BY CITY:")
        for _, row in city_corrs.iterrows():
            print(f"  {row['city']}: {row['avg_temp_correlation']} (n={row['sample_size']})")
        
        # Weekend vs Weekday
        weekend_analysis = self.get_weekend_vs_weekday_analysis()
        print(f"\n📅 WEEKEND vs WEEKDAY:")
        print(f"  Weekend Average: {weekend_analysis['weekend_avg']:,.0f} MWh")
        print(f"  Weekday Average: {weekend_analysis['weekday_avg']:,.0f} MWh")
        if weekend_analysis.get('percentage_difference'):
            print(f"  Difference: {weekend_analysis['percentage_difference']:.1f}%")
        if weekend_analysis.get('significant_difference'):
            sig = "Yes" if weekend_analysis['significant_difference'] else "No"
            print(f"  Statistically Significant: {sig} (p={weekend_analysis['p_value']})")
        
        print("\n" + "="*70 + "\n")


# Test function
if __name__ == "__main__":
    import logging
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*70)
    print("TESTING ANALYSIS MODULE")
    print("="*70)
    
    # Load processed data
    try:
        data = pd.read_csv('../data/processed/combined_data.csv', parse_dates=['date'])
        print(f"\n✓ Loaded {len(data)} records from {data['city'].nunique()} cities")
        
        # Initialize analysis
        analysis = EnergyAnalysis(data)
        
        # Run comprehensive analysis
        analysis.print_analysis_summary()
        
        # Test specific analyses
        print("\nTesting specific analysis functions...")
        
        # City statistics
        city_stats = analysis.get_city_statistics()
        print(f"\n✓ City Statistics:")
        print(city_stats.to_string(index=False))
        
        # Usage patterns
        usage_patterns = analysis.get_usage_patterns()
        print(f"\n✓ Usage Patterns (heatmap data):")
        print(usage_patterns)
        
        print("\n" + "="*70)
        print("✅ ANALYSIS MODULE TEST COMPLETE")
        print("="*70 + "\n")
        
    except FileNotFoundError:
        print("\n❌ Error: No processed data found!")
        print("Please run the pipeline first to collect data.")
        print("Run: cd src && python pipeline.py")