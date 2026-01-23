# 🌡️ US Weather-Energy Analysis Pipeline

A production-ready data engineering pipeline that analyzes the relationship between weather patterns and electrical energy consumption across major US cities.

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📊 Project Overview

This project demonstrates end-to-end data engineering skills including:
- **API Integration**: NOAA Climate Data & EIA Energy APIs
- **ETL Pipeline**: Automated data fetching, cleaning, and transformation
- **Statistical Analysis**: Correlation analysis, regression modeling
- **Interactive Visualization**: Real-time Streamlit dashboard
- **Production Best Practices**: Logging, error handling, configuration management

## 🎯 Business Problem

Energy grid operators need to understand how temperature affects electricity demand to:
- Optimize power generation scheduling
- Predict peak demand periods
- Reduce operational costs
- Improve grid reliability

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  NOAA API   │────▶│   Pipeline   │────▶│  Dashboard  │
│  EIA API    │     │  (Orchestrator)│     │ (Streamlit) │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                    ┌──────┴──────┐
                    │             │
              ┌─────▼────┐  ┌────▼─────┐
              │  Fetcher │  │ Processor│
              └──────────┘  └──────────┘
                    │             │
              ┌─────▼─────────────▼────┐
              │    Analysis Module     │
              └────────────────────────┘
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

## 📁 Project Structure

```
project1-energy-analysis/
├── src/
│   ├── data_fetcher.py      # API integration
│   ├── data_processor.py    # Data cleaning & transformation
│   ├── analysis.py          # Statistical analysis
│   └── pipeline.py          # Orchestration
├── dashboards/
│   └── app.py               # Streamlit dashboard
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
- Handles rate limiting and retries
- Validates and cleans data
- Generates quality reports

### 2. Statistical Analysis
- Correlation analysis (Pearson coefficient)
- Linear regression modeling
- Outlier detection
- Weekend/weekday patterns

### 3. Interactive Dashboard
- Real-time data visualization
- Geographic heat maps
- Time series analysis
- Correlation scatter plots
- Configurable date ranges and city filters

## 🔍 Analysis Results

**Key Findings:**
- Average temperature-energy correlation: **-0.42** (moderate negative)
- Houston shows strongest correlation: **-0.81**
- R² score: **0.18** (temperature explains 18% of variance)
- Energy consumption 0.7% lower on weekends

## 🛠️ Technologies Used

- **Languages**: Python 3.14
- **Data Processing**: pandas, numpy, scipy
- **Visualization**: Plotly, Streamlit
- **APIs**: NOAA CDO, EIA
- **Testing**: pytest
- **Configuration**: YAML, python-dotenv

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