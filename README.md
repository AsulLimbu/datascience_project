# Smart Energy Consumption Forecasting & Optimization System

A comprehensive data science project for analyzing and predicting electricity consumption patterns in Nepal using real-world NEA data and machine learning.

## Project Structure

```
datascience_project/
├── data/
│   ├── NMOR*.pdf          # NEA Monthly Operation Reports (raw)
│   ├── raw/               # Scraped weather & holiday data
│   └── processed/         # Cleaned datasets
├── notebooks/
│   └── 01_eda_energy_data.ipynb
├── src/
│   ├── scraper/
│   │   ├── weather_scraper.py
│   │   └── holiday_scraper.py
│   └── models/            # ML models (to be added)
├── reports/               # Generated visualizations
└── requirements.txt
```

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Run Web Scraping

```python
# Fetch weather data
from src.scraper.weather_scraper import fetch_nepal_weather_for_energy_analysis
weather_df = fetch_nepal_weather_for_energy_analysis()

# Fetch holidays
from src.scraper.holiday_scraper import fetch_nepal_holidays
holidays_df = fetch_nepal_holidays()
```

### 2. Run EDA

Open `notebooks/01_eda_energy_data.ipynb` in Jupyter.

## Team Responsibilities

| Member | Role | Focus Areas |
|--------|------|-------------|
| Member 1 | Data Engineer | ETL pipeline, database, scheduling |
| Member 2 | Data Wrangler | Feature engineering, cleaning, optimization |
| Member 3 | ML Engineer | Model building, training, evaluation |
| Member 4 | Visualization | Dashboard, reports, storytelling |

## Data Sources

- **Energy Data**: Nepal Electricity Authority (NEA) Monthly Operation Reports
- **Weather Data**: Open-Meteo API (free, no key required)
- **Holidays**: timeanddate.com + fallback data