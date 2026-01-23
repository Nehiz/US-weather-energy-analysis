# 🌡️ US Weather-Energy Analysis Pipeline

A production-ready data engineering and data science pipeline that analyzes the relationship between weather patterns and electrical energy consumption across major US cities, featuring **machine learning forecasting** and **actionable business recommendations**.

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.8+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📊 Project Overview

This project demonstrates comprehensive data engineering and data science skills including:
- **API Integration**: NOAA Climate Data & EIA Energy APIs
- **ETL Pipeline**: Automated data fetching, cleaning, and transformation
- **Machine Learning**: Energy demand forecasting with Gradient Boosting (99.3% R²)
- **Statistical Analysis**: Correlation analysis, regression modeling, hypothesis testing
- **Business Intelligence**: Actionable recommendations with cost-benefit analysis
- **Interactive Visualization**: Real-time Streamlit dashboard with 4 tabs
- **Production Best Practices**: Logging, error handling, configuration management, testing

## 🎯 Business Problem

Energy grid operators need to understand how temperature affects electricity demand to:
- **Forecast** day-ahead energy consumption for capacity planning
- **Optimize** power generation scheduling and reduce costs
- **Predict** peak demand periods to prevent grid overload
- **Identify** efficiency opportunities across cities
- **Reduce** operational costs through data-driven insights

**This project solves these problems** by providing predictive models (5.63% MAPE) and prioritized recommendations with estimated annual savings of **$31.8M+**.

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────────┐
│  NOAA API   │────▶│   Pipeline   │────▶│  Dashboard          │
│  EIA API    │     │  (Orchestrator)│     │ 📊 Analysis         │
└─────────────┘     └──────────────┘     │ 🤖 ML Forecasting   │
                           │              │ 💡 Recommendations  │
                    ┌──────┴──────┐       └─────────────────────┘
                    │             │
              ┌─────▼────┐  ┌────▼─────┐
              │  Fetcher │  │ Processor│
              └──────────┘  └──────────┘
                    │             │
         ┌──────────┴─────────────┴──────────────┐
         │                                        │
   ┌─────▼────────┐  ┌──────────────┐  ┌────────▼────────┐
   │   Analysis   │  │  Forecasting │  │ Recommendations │
   │   Module     │  │  (ML Model)  │  │     Engine      │
   └──────────────┘  └──────────────┘  └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.14+
- API Keys: [NOAA CDO](https://www.ncdc.noaa.gov/cdo-web/token) & [EIA](https://www.eia.gov/opendata/)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/weather-energy-analysis.git
cd weather-energy-analysis

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Configure API keys
echo "NOAA_TOKEN=your_noaa_token" > .env
echo "EIA_API_KEY=your_eia_key" >> .env

# Run pipeline
cd src
python pipeline.py

# Launch dashboard
cd ..
streamlit run dashboards/app.py
```

Access dashboard at: **http://localhost:8501**

## 📁 Project Structure

```
project1-energy-analysis/
├── src/
│   ├── data_fetcher.py      # API integration
│   ├── data_processor.py    # Data cleaning & transformation
│   ├── analysis.py          # Statistical analysis
│   ├── forecasting.py       # ML prediction model (NEW! 🤖)
│   ├── recommendations.py   # Business insights engine (NEW! 💡)
│   ├── pipeline.py          # Orchestration
│   ├── data_quality.py      # Quality monitoring
│   └── performance.py       # Performance tracking
├── dashboards/
│   └── app.py               # Streamlit dashboard (4 tabs!)
├── config/
│   └── config.yaml          # Configuration
├── data/
│   ├── raw/                 # Raw API data
│   └── processed/           # Cleaned data
├── tests/                   # Unit tests
├── logs/                    # Application logs
└── notebooks/               # Exploratory analysis
```

## 📈 Key Features

### 1. Automated Data Pipeline
- Fetches data from NOAA (weather) and EIA (energy) APIs
- Handles rate limiting, retries, and error recovery
- Validates and cleans data automatically
- Generates comprehensive quality reports

### 2. Machine Learning Forecasting 🤖
- **Gradient Boosting Regressor** for energy demand prediction
- **99.3% R² score** with **5.63% MAPE** on test data
- Feature importance analysis (temperature, time, location)
- Interactive forecast tool with confidence intervals
- Business impact analysis: **$219M+ annual savings potential**

### 3. Actionable Recommendations 💡
- **5 prioritized recommendations** based on data analysis
- Temperature sensitivity analysis by city
- Weekend vs weekday efficiency patterns
- Peak demand reduction strategies
- Cost-benefit analysis for each recommendation
- **Total potential savings: $31.8M annually**

### 4. Statistical Analysis 📊
- Correlation analysis (Pearson coefficient)
- Linear regression with R² scoring
- Hypothesis testing for weekend patterns
- Outlier detection and handling
- Cross-city benchmarking

### 5. Interactive Dashboard 🎨
**4 comprehensive tabs:**
- **📊 Exploratory Analysis**: Geographic maps, time series, correlations, heatmaps
- **🤖 ML Forecasting**: Model performance, predictions, interactive forecasting tool
- **💡 Recommendations**: Prioritized action items with cost-benefit analysis
- **📖 About**: Project documentation and technical details

## 🔍 Key Results

### Statistical Findings
- Average temperature-energy correlation: **-0.42** (moderate negative)
- Houston shows strongest correlation: **-0.81**
- R² score: **0.18** (temperature explains 18% of variance)
- Energy consumption 0.7% lower on weekends

### Machine Learning Performance 🎯
- **Test R² Score**: 0.9933 (99.3% variance explained)
- **Test MAE**: 43,333 MWh (Mean Absolute Error)
- **Test MAPE**: 5.63% (highly accurate predictions)
- **Cross-Validation R²**: 0.9889 ± 0.0111
- **Model Improvement**: 94.5% better than naive baseline

### Business Impact 💰
- **Estimated Annual Savings**: $219.6M from accurate forecasting
- **Total Recommendation Value**: $31.8M+ annually
- **Payback Period**: 6 months
- **Top Opportunity**: Peak demand reduction (76.4M MWh/year savings)

## 🛠️ Technologies Used

- **Languages**: Python 3.14
- **Data Processing**: pandas, numpy, scipy
- **Machine Learning**: scikit-learn (Gradient Boosting, Random Forest)
- **Visualization**: Plotly, Streamlit
- **APIs**: NOAA CDO, EIA
- **Testing**: pytest
- **Configuration**: YAML, python-dotenv
- **Statistical Analysis**: statsmodels, scipy.stats

## 📊 Sample Output

```python
# Correlation by City
Chicago:  -0.569
Houston:  -0.805
New York: -0.788
Phoenix:  -0.590
Seattle:  -0.788
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_data_fetcher.py
```

## 📝 Future Enhancements

- [ ] Add machine learning predictions
- [ ] Implement real-time alerts
- [ ] Add more cities (50+ coverage)
- [ ] Deploy to cloud (AWS/Azure)
- [ ] Add CI/CD pipeline
- [ ] Create REST API endpoint
- [ ] Add email notifications

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)
- Portfolio: [yourwebsite.com](https://yourwebsite.com)

## 🙏 Acknowledgments

- NOAA for climate data API
- EIA for energy consumption data
- Streamlit for dashboard framework