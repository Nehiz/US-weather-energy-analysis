"""
Recommendation engine for actionable energy insights
Generates data-driven recommendations for energy optimization
"""
import pandas as pd
import numpy as np
from typing import List, Dict
import logging


class EnergyRecommendationEngine:
    """
    Generates actionable recommendations based on data analysis
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_city_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze which cities are most efficient"""
        city_stats = df.groupby('city').agg({
            'energy_consumption': ['mean', 'std', 'min', 'max'],
            'avg_temp': 'mean'
        }).round(2)
        
        city_stats.columns = ['_'.join(col).strip() for col in city_stats.columns.values]
        city_stats['efficiency_score'] = (
            city_stats['energy_consumption_mean'] / city_stats['avg_temp_mean']
        )
        city_stats = city_stats.sort_values('efficiency_score', ascending=True)
        
        return city_stats
    
    def identify_high_consumption_days(self, df: pd.DataFrame, threshold_percentile: int = 90) -> pd.DataFrame:
        """Identify days with unusually high energy consumption"""
        threshold = df['energy_consumption'].quantile(threshold_percentile / 100)
        
        high_days = df[df['energy_consumption'] >= threshold].copy()
        high_days['excess_consumption'] = high_days['energy_consumption'] - df['energy_consumption'].median()
        
        return high_days.sort_values('energy_consumption', ascending=False)
    
    def temperature_sensitivity_analysis(self, df: pd.DataFrame) -> Dict:
        """Analyze how each city responds to temperature changes"""
        results = {}
        
        for city in df['city'].unique():
            city_data = df[df['city'] == city]
            
            # Calculate correlation
            corr = city_data['avg_temp'].corr(city_data['energy_consumption'])
            
            # Find optimal temperature range (lowest consumption)
            temp_bins = pd.cut(city_data['avg_temp'], bins=5)
            avg_by_temp = city_data.groupby(temp_bins)['energy_consumption'].mean()
            optimal_temp_range = avg_by_temp.idxmin()
            
            # Identify extreme weather days
            cold_days = city_data[city_data['avg_temp'] < city_data['avg_temp'].quantile(0.25)]
            hot_days = city_data[city_data['avg_temp'] > city_data['avg_temp'].quantile(0.75)]
            
            results[city] = {
                'correlation': corr,
                'optimal_temp_range': str(optimal_temp_range),
                'avg_consumption': city_data['energy_consumption'].mean(),
                'cold_day_avg': cold_days['energy_consumption'].mean() if len(cold_days) > 0 else 0,
                'hot_day_avg': hot_days['energy_consumption'].mean() if len(hot_days) > 0 else 0,
                'temp_sensitivity': abs(corr)
            }
        
        return results
    
    def weekend_vs_weekday_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compare weekend vs weekday consumption patterns"""
        analysis = df.groupby(['city', 'is_weekend']).agg({
            'energy_consumption': ['mean', 'std', 'count']
        }).round(2)
        
        analysis.columns = ['_'.join(col).strip() for col in analysis.columns.values]
        
        # Calculate weekend savings
        weekend_analysis = []
        for city in df['city'].unique():
            weekday_avg = df[(df['city'] == city) & (~df['is_weekend'])]['energy_consumption'].mean()
            weekend_avg = df[(df['city'] == city) & (df['is_weekend'])]['energy_consumption'].mean()
            
            savings_pct = ((weekday_avg - weekend_avg) / weekday_avg * 100) if weekday_avg > 0 else 0
            
            weekend_analysis.append({
                'city': city,
                'weekday_avg': weekday_avg,
                'weekend_avg': weekend_avg,
                'savings_pct': savings_pct,
                'potential_annual_savings_mwh': (weekday_avg - weekend_avg) * 52 * 2  # 2 days per weekend
            })
        
        return pd.DataFrame(weekend_analysis).sort_values('savings_pct', ascending=False)
    
    def generate_recommendations(self, df: pd.DataFrame) -> List[Dict]:
        """
        Generate prioritized, actionable recommendations
        """
        recommendations = []
        
        # 1. Temperature-based recommendations
        temp_analysis = self.temperature_sensitivity_analysis(df)
        most_sensitive_city = max(temp_analysis.items(), key=lambda x: x[1]['temp_sensitivity'])
        
        recommendations.append({
            'priority': 'HIGH',
            'category': 'Temperature Management',
            'title': f'Optimize HVAC in {most_sensitive_city[0]}',
            'description': f'{most_sensitive_city[0]} shows high temperature sensitivity (correlation: {most_sensitive_city[1]["correlation"]:.2f}). '
                          f'Implementing smart thermostat controls could reduce consumption by 10-15%.',
            'estimated_savings_mwh': most_sensitive_city[1]['avg_consumption'] * 0.12 * 365,
            'estimated_cost_savings': most_sensitive_city[1]['avg_consumption'] * 0.12 * 365 * 50,  # $50/MWh
            'action_items': [
                'Install programmable thermostats in commercial buildings',
                'Set temperature bands: 68-72°F in winter, 74-78°F in summer',
                'Implement pre-cooling/pre-heating during off-peak hours'
            ]
        })
        
        # 2. Peak demand reduction
        high_days = self.identify_high_consumption_days(df, threshold_percentile=90)
        avg_excess = high_days['excess_consumption'].mean()
        
        recommendations.append({
            'priority': 'HIGH',
            'category': 'Peak Demand Reduction',
            'title': 'Implement Demand Response Program',
            'description': f'Top 10% consumption days use {avg_excess:,.0f} MWh more than median. '
                          f'Peak shaving strategies could reduce demand charges significantly.',
            'estimated_savings_mwh': avg_excess * 36,  # ~10% of days
            'estimated_cost_savings': avg_excess * 36 * 150,  # Higher cost for peak demand
            'action_items': [
                'Enroll in utility demand response programs',
                'Shift non-critical loads to off-peak hours',
                'Install battery storage for peak shaving',
                'Implement automated load curtailment systems'
            ]
        })
        
        # 3. Weekend efficiency patterns
        weekend_analysis = self.weekend_vs_weekday_analysis(df)
        if not weekend_analysis.empty:
            top_opportunity = weekend_analysis.iloc[0]
            
            if top_opportunity['savings_pct'] > 0:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'category': 'Operational Efficiency',
                    'title': f'Replicate {top_opportunity["city"]} Weekend Efficiency',
                    'description': f'{top_opportunity["city"]} achieves {top_opportunity["savings_pct"]:.1f}% lower consumption on weekends. '
                                  f'Apply similar strategies to weekdays where possible.',
                    'estimated_savings_mwh': top_opportunity['potential_annual_savings_mwh'] * 0.3,  # Conservative 30% transfer
                    'estimated_cost_savings': top_opportunity['potential_annual_savings_mwh'] * 0.3 * 50,
                    'action_items': [
                        'Audit weekend operations to identify efficiency drivers',
                        'Reduce lighting and HVAC in unoccupied areas during weekdays',
                        'Implement occupancy-based control systems',
                        'Schedule energy-intensive tasks during optimal hours'
                    ]
                })
        
        # 4. City benchmarking
        city_perf = self.analyze_city_performance(df)
        best_city = city_perf.index[0]
        worst_city = city_perf.index[-1]
        
        gap = city_perf.loc[worst_city, 'energy_consumption_mean'] - city_perf.loc[best_city, 'energy_consumption_mean']
        
        recommendations.append({
            'priority': 'MEDIUM',
            'category': 'Best Practices',
            'title': f'Benchmark {worst_city} Against {best_city}',
            'description': f'{best_city} uses {gap:,.0f} MWh/day less than {worst_city}. '
                          f'Cross-regional learning could improve efficiency.',
            'estimated_savings_mwh': gap * 0.4 * 365,  # 40% improvement potential
            'estimated_cost_savings': gap * 0.4 * 365 * 50,
            'action_items': [
                f'Conduct site visit to {best_city} facilities',
                'Document and replicate best practices',
                'Standardize energy management procedures across regions',
                'Share efficiency metrics across all locations'
            ]
        })
        
        # 5. Data quality and monitoring
        missing_data = df.isnull().sum().sum()
        recommendations.append({
            'priority': 'LOW',
            'category': 'Data & Monitoring',
            'title': 'Enhance Real-Time Monitoring',
            'description': f'Current data has {missing_data} missing values. Better monitoring enables proactive optimization.',
            'estimated_savings_mwh': df['energy_consumption'].sum() * 0.05,  # 5% from better visibility
            'estimated_cost_savings': df['energy_consumption'].sum() * 0.05 * 50,
            'action_items': [
                'Install smart meters in all facilities',
                'Deploy real-time energy dashboard',
                'Set up automated alerts for anomalies',
                'Conduct weekly energy reviews'
            ]
        })
        
        # Sort by priority
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        recommendations.sort(key=lambda x: (priority_order[x['priority']], -x['estimated_cost_savings']))
        
        return recommendations
    
    def generate_executive_summary(self, recommendations: List[Dict]) -> Dict:
        """Generate executive summary of all recommendations"""
        total_savings_mwh = sum(r['estimated_savings_mwh'] for r in recommendations)
        total_cost_savings = sum(r['estimated_cost_savings'] for r in recommendations)
        
        high_priority = len([r for r in recommendations if r['priority'] == 'HIGH'])
        
        return {
            'total_recommendations': len(recommendations),
            'high_priority_count': high_priority,
            'total_estimated_savings_mwh': total_savings_mwh,
            'total_estimated_cost_savings': total_cost_savings,
            'payback_period_months': 6,  # Estimated based on typical implementation costs
            'summary': f'Implementing these {len(recommendations)} recommendations could save '
                      f'{total_savings_mwh:,.0f} MWh annually (${total_cost_savings:,.0f}). '
                      f'Focus on {high_priority} high-priority items first for quick wins.'
        }


if __name__ == "__main__":
    import sys
    sys.path.append('..')
    
    logging.basicConfig(level=logging.INFO)
    
    # Load data
    df = pd.read_csv('../data/processed/combined_data.csv', parse_dates=['date'])
    
    # Generate recommendations
    engine = EnergyRecommendationEngine()
    recommendations = engine.generate_recommendations(df)
    
    print("\n" + "="*80)
    print("ACTIONABLE ENERGY RECOMMENDATIONS")
    print("="*80)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. [{rec['priority']}] {rec['title']}")
        print(f"   Category: {rec['category']}")
        print(f"   {rec['description']}")
        print(f"   💰 Estimated Annual Savings: ${rec['estimated_cost_savings']:,.0f}")
        print(f"   ⚡ Energy Reduction: {rec['estimated_savings_mwh']:,.0f} MWh/year")
        print(f"   Action Items:")
        for action in rec['action_items']:
            print(f"      • {action}")
    
    # Executive summary
    summary = engine.generate_executive_summary(recommendations)
    print("\n" + "="*80)
    print("EXECUTIVE SUMMARY")
    print("="*80)
    print(summary['summary'])
    print(f"Total Potential Annual Savings: ${summary['total_estimated_cost_savings']:,.0f}")
    print(f"Total Energy Reduction: {summary['total_estimated_savings_mwh']:,.0f} MWh/year")
    print(f"Estimated Payback Period: {summary['payback_period_months']} months")
