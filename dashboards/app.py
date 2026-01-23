"""
Streamlit Dashboard for Weather-Energy Analysis
Interactive visualizations showing temperature-energy relationships
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add src to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from analysis import EnergyAnalysis
from forecasting import EnergyForecaster
from recommendations import EnergyRecommendationEngine

# Page configuration
st.set_page_config(
    page_title="Weather-Energy Analysis Dashboard",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    /* Make metric text visible with dark colors */
    [data-testid="stMetricValue"] {
        color: #000000 !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #1a1a1a !important;
        font-weight: 600 !important;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load processed data with caching for performance"""
    # Try multiple possible paths
    possible_paths = [
        '../data/processed/combined_data.csv',
        'data/processed/combined_data.csv',
        os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'combined_data.csv')
    ]
    
    for path in possible_paths:
        try:
            if os.path.exists(path):
                data = pd.read_csv(path, parse_dates=['date'])
                return data
        except Exception:
            continue
    
    # If no path worked, show error
    st.error("❌ No data found! Please run the pipeline first.")
    st.info("Run: `cd src && python pipeline.py` to collect data")
    return pd.DataFrame()


def main():
    # Title and description
    st.title("🌡️ Weather-Energy Analysis Dashboard")
    st.markdown("""
    Real-time insights into temperature-energy consumption relationships across 5 US cities.
    This dashboard analyzes how weather patterns influence electrical grid demand.
    """)
    
    # Load data
    data = load_data()
    if data.empty:
        st.stop()
    
    # Sidebar filters
    st.sidebar.header("🎛️ Dashboard Controls")
    
    # Date range selector
    min_date = data['date'].min().date()
    max_date = data['date'].max().date()
    
    st.sidebar.subheader("Date Range")
    date_range = st.sidebar.date_input(
        "Select dates",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="date_range"
    )
    
    # City selector
    st.sidebar.subheader("Cities")
    all_cities = sorted(data['city'].unique())
    cities = st.sidebar.multiselect(
        "Select cities to analyze",
        options=all_cities,
        default=all_cities
    )
    
    if not cities:
        st.warning("⚠️ Please select at least one city")
        st.stop()
    
    # Filter data based on selections
    if len(date_range) == 2:
        filtered_data = data[
            (data['date'].dt.date >= date_range[0]) & 
            (data['date'].dt.date <= date_range[1]) &
            (data['city'].isin(cities))
        ]
    else:
        filtered_data = data[data['city'].isin(cities)]
    
    if filtered_data.empty:
        st.warning("⚠️ No data available for selected filters")
        st.stop()
    
    # Initialize analysis
    analysis = EnergyAnalysis(filtered_data)
    
    # Calculate key metrics
    regression_stats = analysis.calculate_regression_stats()
    city_correlations = analysis.calculate_correlations()
    
    # Top metrics row
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "📊 Total Records",
            f"{len(filtered_data):,}",
            delta=None
        )
    
    with col2:
        st.metric(
            "🏙️ Cities",
            f"{filtered_data['city'].nunique()}",
            delta=None
        )
    
    with col3:
        avg_corr = city_correlations['avg_temp_correlation'].mean()
        st.metric(
            "📈 Avg Correlation",
            f"{avg_corr:.3f}",
            delta=None,
            help="Average temperature-energy correlation across cities"
        )
    
    with col4:
        latest_date = filtered_data['date'].max().strftime('%b %d, %Y')
        st.metric(
            "📅 Latest Data",
            latest_date,
            delta=None
        )
    
    with col5:
        st.metric(
            "🎯 R² Score",
            f"{regression_stats['r_squared']:.3f}",
            delta=None,
            help="How well temperature explains energy variance"
        )
    
    st.markdown("---")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Exploratory Analysis",
        "🤖 ML Forecasting",
        "💡 Recommendations",
        "📖 About"
    ])
    
    with tab1:
        # Visualization 1: Geographic Overview
        st.header("📍 Geographic Overview")
        st.markdown("Interactive map showing current status for each city")
        
        # Prepare city summary data
    city_summary = filtered_data.groupby('city').agg({
        'avg_temp': 'last',
        'energy_consumption': ['last', lambda x: ((x.iloc[-1] - x.iloc[-2]) / x.iloc[-2] * 100) if len(x) > 1 else 0],
        'state': 'first'
    }).reset_index()
    
    city_summary.columns = ['city', 'current_temp', 'current_energy', 'energy_change_pct', 'state']
    
    # Add coordinates for cities
    city_coords = {
        'New York': {'lat': 40.7128, 'lon': -74.0060},
        'Chicago': {'lat': 41.8781, 'lon': -87.6298},
        'Houston': {'lat': 29.7604, 'lon': -95.3698},
        'Phoenix': {'lat': 33.4484, 'lon': -112.0740},
        'Seattle': {'lat': 47.6062, 'lon': -122.3321}
    }
    
    city_summary['lat'] = city_summary['city'].map(lambda x: city_coords.get(x, {}).get('lat', 0))
    city_summary['lon'] = city_summary['city'].map(lambda x: city_coords.get(x, {}).get('lon', 0))
    
    # Create map
    fig_map = px.scatter_mapbox(
        city_summary,
        lat='lat',
        lon='lon',
        size=abs(city_summary['current_energy']) / 100000,
        color='current_energy',
        hover_name='city',
        hover_data={
            'current_temp': ':.1f',
            'energy_change_pct': ':.1f',
            'lat': False,
            'lon': False
        },
        color_continuous_scale='RdYlGn_r',
        title=f"Current Status Across Cities (Last Updated: {latest_date})",
        zoom=3,
        height=500
    )
    
    fig_map.update_layout(mapbox_style="open-street-map")
    fig_map.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0})
    
    st.plotly_chart(fig_map, use_container_width=True)
    
    # Show city details table
    with st.expander("📊 View Detailed City Statistics"):
        city_stats = analysis.get_city_statistics()
        st.dataframe(
            city_stats.style.format({
                'avg_temp': '{:.2f}',
                'avg_energy': '{:,.0f}',
                'temp_energy_correlation': '{:.4f}'
            }),
            use_container_width=True
            )
        
        st.markdown("---")
        
        # Visualization 2: Time Series Analysis
        st.header("📈 Time Series Analysis")
        st.markdown("Temperature and energy consumption trends over time")
        
        # City selector for time series
        col1, col2 = st.columns([3, 1])
        with col1:
            ts_city_option = st.selectbox(
                "Select city for time series view",
                options=["All Cities Combined"] + sorted(cities),
                key="ts_city"
            )
    
    # Prepare time series data
    if ts_city_option == "All Cities Combined":
        ts_data = filtered_data.groupby('date').agg({
            'avg_temp': 'mean',
            'energy_consumption': 'mean'
        }).reset_index()
        title_suffix = "All Cities (Average)"
    else:
        ts_data = filtered_data[filtered_data['city'] == ts_city_option].copy()
        title_suffix = ts_city_option
    
    # Create dual-axis time series plot
    fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add temperature trace
    fig_ts.add_trace(
        go.Scatter(
            x=ts_data['date'],
            y=ts_data['avg_temp'],
            name="Temperature (°F)",
            line=dict(color='#FF6B6B', width=2),
            mode='lines+markers'
        ),
        secondary_y=False,
    )
    
    # Add energy trace
    fig_ts.add_trace(
        go.Scatter(
            x=ts_data['date'],
            y=ts_data['energy_consumption'],
            name="Energy (MWh)",
            line=dict(color='#4ECDC4', width=2, dash='dot'),
            mode='lines+markers'
        ),
        secondary_y=True,
    )
    
    # Add weekend shading
    for idx, row in ts_data.iterrows():
        date = pd.to_datetime(row['date'])
        if date.weekday() >= 5:  # Weekend
            fig_ts.add_vrect(
                x0=date,
                x1=date + timedelta(days=1),
                fillcolor="lightgray",
                opacity=0.2,
                layer="below",
                line_width=0,
            )
    
    # Update axes
    fig_ts.update_xaxes(title_text="Date")
    fig_ts.update_yaxes(title_text="Temperature (°F)", secondary_y=False)
    fig_ts.update_yaxes(title_text="Energy Consumption (MWh)", secondary_y=True)
    
    # Update layout
    fig_ts.update_layout(
        title_text=f"Temperature vs Energy Consumption - {title_suffix}",
        hovermode='x unified',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig_ts, use_container_width=True)
    
    # Show weekend analysis
    with st.expander("📅 Weekend vs Weekday Analysis"):
        weekend_analysis = analysis.get_weekend_vs_weekday_analysis()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Weekend Average",
                f"{weekend_analysis['weekend_avg']:,.0f} MWh",
                delta=f"{weekend_analysis.get('percentage_difference', 0):.1f}%"
            )
        
        with col2:
            st.metric(
                "Weekday Average",
                f"{weekend_analysis['weekday_avg']:,.0f} MWh"
            )
        
        with col3:
            if weekend_analysis.get('significant_difference') is not None:
                    sig_text = "Yes ✓" if weekend_analysis['significant_difference'] else "No ✗"
                    st.metric(
                        "Statistically Significant",
                        sig_text,
                        help=f"P-value: {weekend_analysis.get('p_value', 'N/A')}"
                    )
        
        st.markdown("---")
        
        # Visualization 3: Correlation Analysis
        st.header("🔗 Correlation Analysis")
        st.markdown("Scatter plot showing temperature-energy relationship with regression line")
        
        # Create scatter plot with trendline
        fig_scatter = px.scatter(
            filtered_data,
        x='avg_temp',
        y='energy_consumption',
        color='city',
        hover_data=['date', 'city', 'TMAX', 'TMIN'],
        trendline='ols',
        title=f"Temperature vs Energy Consumption (R² = {regression_stats['r_squared']:.4f})",
        labels={
            'avg_temp': 'Average Temperature (°F)',
            'energy_consumption': 'Energy Consumption (MWh)'
        },
        height=600
    )
    
    # Add annotation with regression equation
    fig_scatter.add_annotation(
        x=filtered_data['avg_temp'].quantile(0.95),
        y=filtered_data['energy_consumption'].quantile(0.95),
        text=f"<b>Regression Analysis</b><br>" +
             f"{regression_stats['equation']}<br>" +
             f"R² = {regression_stats['r_squared']:.4f}<br>" +
             f"Correlation = {regression_stats['correlation']:.4f}<br>" +
             f"P-value = {regression_stats['p_value']:.6f}<br>" +
             f"<i>{regression_stats['interpretation']}</i>",
        showarrow=False,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        borderpad=10,
        font=dict(size=11),
        align="left"
    )
    
    fig_scatter.update_layout(
        showlegend=True,
        legend=dict(
            title="City",
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Show correlation table
    with st.expander("📊 View Correlation Statistics by City"):
        st.dataframe(
            city_correlations.style.format({
                'avg_temp_correlation': '{:.4f}',
                'tmax_correlation': '{:.4f}',
                'tmin_correlation': '{:.4f}',
                'avg_temperature': '{:.2f}',
                    'avg_energy': '{:,.0f}'
                }),
                use_container_width=True
            )
        
        st.markdown("---")
        
        # Visualization 4: Usage Patterns Heatmap
        st.header("🔥 Usage Patterns Heatmap")
        st.markdown("Average energy usage by temperature range and day of week")
        
        # City selector for heatmap
        col1, col2 = st.columns([3, 1])
        with col1:
            heatmap_city = st.selectbox(
                "Select city for usage patterns",
            options=["All Cities Combined"] + sorted(cities),
            key="heatmap_city"
        )
    
    # Get usage patterns
    if heatmap_city == "All Cities Combined":
        heatmap_data = analysis.get_usage_patterns()
        title_suffix = "All Cities"
    else:
        heatmap_data = analysis.get_city_usage_patterns(heatmap_city)
        title_suffix = heatmap_city
    
    # Create heatmap
    if not heatmap_data.empty:
        fig_heatmap = px.imshow(
            heatmap_data,
            text_auto='.0f',
            aspect="auto",
            color_continuous_scale='RdYlBu_r',
            title=f"Average Energy Usage Patterns - {title_suffix}",
            labels=dict(x="Day of Week", y="Temperature Range", color="Energy (MWh)"),
            height=400
        )
        
        fig_heatmap.update_xaxes(side="bottom")
        fig_heatmap.update_layout(
            xaxis_title="Day of Week",
            yaxis_title="Temperature Range"
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("Not enough data to generate heatmap for selected filters")
    
    # Key insights
    with st.expander("💡 Key Insights from Usage Patterns"):
        st.markdown(f"""
        **Temperature-Day Patterns:**
        - The heatmap shows how energy consumption varies by temperature range and day of week
            - Darker colors indicate higher energy consumption
            - Weekend patterns may differ from weekday patterns due to reduced commercial activity
            
            **Current Data Characteristics:**
            - Temperature Range: {filtered_data['avg_temp'].min():.1f}°F to {filtered_data['avg_temp'].max():.1f}°F
            - Most data falls in the colder temperature ranges (winter season)
            - This affects the distribution across temperature categories
            """)
        
        st.markdown("---")
        
        # Footer with summary statistics
        st.header("📊 Summary Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Temperature Stats")
            temp_stats = {
                'Average Temperature': f"{filtered_data['avg_temp'].mean():.2f}°F",
                'Min Temperature': f"{filtered_data['avg_temp'].min():.2f}°F",
            'Max Temperature': f"{filtered_data['avg_temp'].max():.2f}°F",
            'Temperature Range': f"{filtered_data['temp_range'].mean():.2f}°F"
        }
        for key, value in temp_stats.items():
                st.metric(key, value)
        
        with col2:
            st.subheader("Energy Stats")
            energy_stats = {
                'Average Consumption': f"{filtered_data['energy_consumption'].mean():,.0f} MWh",
                'Min Consumption': f"{filtered_data['energy_consumption'].min():,.0f} MWh",
                'Max Consumption': f"{filtered_data['energy_consumption'].max():,.0f} MWh",
                'Total Consumption': f"{filtered_data['energy_consumption'].sum():,.0f} MWh"
            }
            for key, value in energy_stats.items():
                st.metric(key, value)
    
    # ML Forecasting Tab
    with tab2:
        display_ml_forecasting(filtered_data)
    
    # Recommendations Tab
    with tab3:
        display_recommendations(filtered_data)
    
    # About Tab
    with tab4:
        st.header("📖 About This Dashboard")
        
        st.markdown("""
        ### Overview
        This dashboard provides comprehensive analysis of weather-energy relationships across 5 major US cities,
        combining exploratory data analysis, machine learning forecasting, and actionable business recommendations.
        
        ### Data Sources
        - **NOAA Climate Data Online**: Daily temperature readings from weather stations
        - **EIA (Energy Information Administration)**: Regional electricity consumption data
        - **Cities Analyzed**: New York, Chicago, Houston, Phoenix, Seattle
        
        ### Features
        
        #### 📊 Exploratory Analysis
        - Geographic visualization of current energy status
        - Time series trends for temperature and energy consumption
        - Correlation analysis with statistical significance
        - Usage pattern heatmaps by temperature and day of week
        
        #### 🤖 ML Forecasting
        - **Gradient Boosting Regressor** for energy demand prediction
        - Feature importance analysis showing key drivers
        - Interactive forecast tool for scenario planning
        - Business impact analysis with estimated cost savings
        - Model performance metrics (R², MAE, MAPE)
        
        #### 💡 Recommendations
        - Data-driven, actionable recommendations prioritized by impact
        - Temperature sensitivity analysis by city
        - Weekend vs weekday efficiency patterns
        - City benchmarking to identify best practices
        - Cost-benefit analysis for each recommendation
        
        ### Technical Stack
        - **Python 3.14**: Core programming language
        - **Streamlit**: Interactive web dashboard framework
        - **Plotly**: Advanced data visualization
        - **Scikit-learn**: Machine learning models
        - **Pandas**: Data manipulation and analysis
        - **NumPy & SciPy**: Scientific computing
        
        ### Key Insights
        - **Negative correlation** between temperature and energy consumption in most cities
        - Heating demand drives winter energy usage patterns
        - Weekend consumption typically lower than weekdays
        - Predictive models achieve ~18% R² with potential for improvement with more data
        - Estimated annual savings potential: Varies by city (see Recommendations tab)
        
        ### Use Cases
        1. **Energy Providers**: Demand forecasting and capacity planning
        2. **Building Managers**: HVAC optimization strategies
        3. **Policy Makers**: Grid reliability and efficiency initiatives
        4. **Researchers**: Understanding urban energy dynamics
        
        ### Future Enhancements
        - Integration with real-time weather APIs
        - Hourly granularity for intraday patterns
        - Renewable energy integration analysis
        - Cost optimization algorithms
        - Automated alert system for anomalies
        
        ### Project Repository
        [View on GitHub](https://github.com/Nehiz/US-weather-energy-analysis)
        
        ### Author
        Built as part of a data engineering and data science portfolio project.
        
        ---
        *Last Updated: {datetime.now().strftime('%B %d, %Y')}*
        """)
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.info("""
    This dashboard analyzes weather-energy relationships using:
    - **NOAA Climate Data**: Daily temperature readings
    - **EIA Energy Data**: Electricity consumption by region
    - **Cities**: New York, Chicago, Houston, Phoenix, Seattle
    
    **Built with:** Python, Streamlit, Plotly, Scikit-learn
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# =============================================================================
# ML FORECASTING SECTION
# =============================================================================
def display_ml_forecasting(df):
    """Display ML forecasting and predictions"""
    st.header("🤖 Machine Learning Forecasting")
    
    with st.spinner("Training forecasting model..."):
        forecaster = EnergyForecaster()
        metrics = forecaster.train_model(df)
    
    # Model Performance Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model R² Score", f"{metrics['test_r2']:.3f}", 
                 help="How well the model explains variance (1.0 = perfect)")
    with col2:
        st.metric("Test MAE", f"{metrics['test_mae']:,.0f} MWh",
                 help="Average prediction error")
    with col3:
        st.metric("Test MAPE", f"{metrics['test_mape']:.1f}%",
                 help="Mean Absolute Percentage Error")
    with col4:
        st.metric("CV Score", f"{metrics['cv_r2_mean']:.3f}",
                 help="Cross-validation R² score")
    
    # Feature Importance
    st.subheader("📊 Feature Importance")
    feature_imp = forecaster.get_feature_importance(10)
    
    fig = px.bar(
        feature_imp, x='importance', y='feature',
        orientation='h',
        title='Top 10 Most Important Features for Energy Prediction',
        labels={'importance': 'Importance Score', 'feature': 'Feature'},
        color='importance',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Predictions vs Actual
    st.subheader("🎯 Predictions vs Actual Consumption")
    predictions = forecaster.predict(df)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['energy_consumption'],
        mode='markers', name='Actual',
        marker=dict(size=8, color='blue', opacity=0.6)
    ))
    fig.add_trace(go.Scatter(
        x=df['date'], y=predictions,
        mode='markers', name='Predicted',
        marker=dict(size=8, color='red', opacity=0.6, symbol='x')
    ))
    fig.update_layout(
        title='Model Predictions vs Actual Energy Consumption',
        xaxis_title='Date',
        yaxis_title='Energy Consumption (MWh)',
        hovermode='x unified',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Interactive Forecast Tool
    st.subheader("🔮 Interactive Forecast Tool")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        forecast_city = st.selectbox("Select City", df['city'].unique())
    with col2:
        forecast_temp = st.slider("Forecast Temperature (°F)", 
                                  min_value=int(df['avg_temp'].min()),
                                  max_value=int(df['avg_temp'].max()),
                                  value=int(df['avg_temp'].mean()))
    with col3:
        is_weekend = st.checkbox("Weekend?", value=False)
    
    forecast = forecaster.forecast_next_day(df, forecast_temp, forecast_city, is_weekend)
    
    st.info(f"""
    **Forecast for {forecast_city} at {forecast_temp}°F:**
    - Predicted Consumption: **{forecast['predicted_consumption']:,.0f} MWh**
    - 95% Confidence Interval: [{forecast['lower_bound']:,.0f}, {forecast['upper_bound']:,.0f}] MWh
    - Forecast Date: {forecast['forecast_date']}
    """)
    
    # Business Impact
    st.subheader("💰 Business Impact Analysis")
    impact = forecaster.calculate_business_impact(df, cost_per_mwh=50)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Improvement", f"{impact['improvement_pct']:.1f}%",
                 help="Improvement over naive baseline forecast")
    with col2:
        st.metric("Estimated Annual Savings", f"${impact['estimated_annual_savings']:,.0f}",
                 help="Potential cost savings from accurate forecasting")
    with col3:
        st.metric("Total Energy", f"{impact['total_energy_mwh']:,.0f} MWh",
                 help="Total energy in dataset")
    
    with st.expander("📘 About the Forecasting Model"):
        st.markdown("""
        **Gradient Boosting Regressor** trained on:
        - Temperature features (avg, max, min, range, squared terms)
        - Time-based features (day of week, month, cyclical encoding)
        - Location features (city one-hot encoding)
        - Interaction terms (temperature ratios, weekend indicators)
        
        **Use Cases:**
        - Day-ahead energy demand forecasting
        - Capacity planning and resource allocation
        - Cost optimization through accurate predictions
        - Risk management for over/under generation
        """)


# =============================================================================
# RECOMMENDATIONS SECTION
# =============================================================================
def display_recommendations(df):
    """Display actionable recommendations"""
    st.header("💡 Actionable Recommendations")
    
    with st.spinner("Generating data-driven recommendations..."):
        engine = EnergyRecommendationEngine()
        recommendations = engine.generate_recommendations(df)
        summary = engine.generate_executive_summary(recommendations)
    
    # Executive Summary
    st.success(f"""
    ### Executive Summary
    {summary['summary']}
    
    **Total Potential Annual Savings:** ${summary['total_estimated_cost_savings']:,.0f}  
    **Total Energy Reduction:** {summary['total_estimated_savings_mwh']:,.0f} MWh/year  
    **Estimated Payback Period:** {summary['payback_period_months']} months
    """)
    
    # Filter by priority
    priority_filter = st.multiselect(
        "Filter by Priority",
        options=['HIGH', 'MEDIUM', 'LOW'],
        default=['HIGH', 'MEDIUM', 'LOW']
    )
    
    filtered_recs = [r for r in recommendations if r['priority'] in priority_filter]
    
    # Display recommendations
    for i, rec in enumerate(filtered_recs, 1):
        priority_colors = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}
        
        with st.expander(f"{priority_colors[rec['priority']]} {i}. {rec['title']}", expanded=(i <= 2)):
            st.markdown(f"**Category:** {rec['category']}")
            st.markdown(f"**Priority:** {rec['priority']}")
            st.markdown(f"**Description:** {rec['description']}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Estimated Annual Savings", f"${rec['estimated_cost_savings']:,.0f}")
            with col2:
                st.metric("Energy Reduction", f"{rec['estimated_savings_mwh']:,.0f} MWh/year")
            
            st.markdown("**Action Items:**")
            for action in rec['action_items']:
                st.markdown(f"- {action}")
    
    # Savings Breakdown Chart
    st.subheader("💵 Savings Potential by Category")
    
    rec_df = pd.DataFrame(filtered_recs)
    category_savings = rec_df.groupby('category')['estimated_cost_savings'].sum().sort_values(ascending=False)
    
    fig = px.bar(
        x=category_savings.values,
        y=category_savings.index,
        orientation='h',
        title='Potential Annual Savings by Category',
        labels={'x': 'Annual Savings ($)', 'y': 'Category'},
        color=category_savings.values,
        color_continuous_scale='Greens'
    )
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Analysis Insights
    st.subheader("📈 Supporting Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Temperature Sensitivity", "Weekend Patterns", "City Benchmarking"])
    
    with tab1:
        temp_analysis = engine.temperature_sensitivity_analysis(df)
        temp_df = pd.DataFrame(temp_analysis).T
        temp_df = temp_df.sort_values('temp_sensitivity', ascending=False)
        
        st.dataframe(temp_df.style.format({
            'correlation': '{:.3f}',
            'avg_consumption': '{:,.0f}',
            'cold_day_avg': '{:,.0f}',
            'hot_day_avg': '{:,.0f}',
            'temp_sensitivity': '{:.3f}'
        }), use_container_width=True)
    
    with tab2:
        weekend_analysis = engine.weekend_vs_weekday_analysis(df)
        st.dataframe(weekend_analysis.style.format({
            'weekday_avg': '{:,.0f}',
            'weekend_avg': '{:,.0f}',
            'savings_pct': '{:.2f}%',
            'potential_annual_savings_mwh': '{:,.0f}'
        }), use_container_width=True)
    
    with tab3:
        city_perf = engine.analyze_city_performance(df)
        st.dataframe(city_perf, use_container_width=True)


# =============================================================================
# MAIN APP WITH NEW SECTIONS
# =============================================================================
if __name__ == "__main__":
    main()