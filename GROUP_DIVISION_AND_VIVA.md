# Smart Energy Consumption Forecasting & Optimization System
## Group Division & Viva Preparation Guide

---

# Table of Contents
1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Group Division](#3-group-division)
   - [Member 1: Data Collection & Data Cleaning](#member-1-data-collection--data-cleaning)
   - [Member 2: EDA & Visualization](#member-2-eda--visualization)
   - [Member 3: Feature Engineering & Model Development](#member-3-feature-engineering--model-development)
   - [Member 4: Model Evaluation & Documentation](#member-4-model-evaluation--documentation)
4. [Technical Implementation Details](#4-technical-implementation-details)
5. [Viva Questions & Answers](#5-viva-questions--answers)

---

# 1. Project Overview

## Problem Statement
"Can we analyze and predict electricity consumption patterns and optimize energy usage using real-world data and machine learning?"

## Objectives
| Objective | Description |
|-----------|-------------|
| Data Collection | Gather energy consumption data from NEA reports, weather data, and holiday calendars |
| EDA | Understand patterns, trends, and relationships in energy consumption |
| Feature Engineering | Create predictive features from temporal, weather, and historical data |
| Model Building | Develop and compare multiple machine learning models for demand forecasting |
| Deployment | Create a reusable model for future predictions |

## Key Results
- **Best Model**: Ridge Regression
- **Accuracy**: 89.91% R² (variance explained)
- **Error Rate**: 4.32% MAPE (Mean Absolute Percentage Error)
- **Data Processed**: 606 daily energy records from 18 PDF reports

---

# 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION LAYER                         │
├─────────────────┬─────────────────┬─────────────────────────────────┤
│   NEA Reports   │   Weather API   │      Holiday Calendar          │
│   (PDF Files)   │  (Open-Meteo)   │    (timeanddate.com)           │
└────────┬────────┴────────┬────────┴───────────────┬─────────────────┘
         │                 │                        │
         ▼                 ▼                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     DATA PROCESSING LAYER                           │
├─────────────────┬─────────────────┬─────────────────────────────────┤
│  PDF Extraction │  API Scraping   │   Data Cleaning & Merging      │
│  (pdfplumber)   │  (requests)     │   (pandas)                     │
└────────┬────────┴────────┬────────┴───────────────┬─────────────────┘
         │                 │                        │
         ▼                 ▼                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINEERING LAYER                         │
├─────────────────┬─────────────────┬─────────────────────────────────┤
│ Time Features   │ Weather Features│     Lag Features               │
│ (day, month,    │ (temp, humidity,│  (previous day,                │
│  season)        │  precipitation) │   rolling average)             │
└────────┬────────┴────────┬────────┴───────────────┬─────────────────┘
         │                 │                        │
         ▼                 ▼                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     MACHINE LEARNING LAYER                          │
├─────────────────┬─────────────────┬─────────────────────────────────┤
│    Linear/Ridge │  Random Forest  │        XGBoost                 │
│    Regression   │   Regressor     │        Regressor               │
└────────┬────────┴────────┬────────┴───────────────┬─────────────────┘
         │                 │                        │
         ▼                 ▼                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      OUTPUT & EVALUATION                            │
├─────────────────┬─────────────────┬─────────────────────────────────┤
│   Predictions   │   Visualizations│      Model Metrics             │
│   (MWh/day)     │   (matplotlib)  │   (RMSE, MAE, R²)              │
└─────────────────┴─────────────────┴─────────────────────────────────┘
```

---

# 3. Group Division

## Member 1: Data Collection & Data Cleaning

### Scope
All data acquisition, extraction, and preprocessing tasks.

### Responsibilities

#### 1. PDF Data Extraction
**Task**: Extract tabular energy data from NEA Monthly Operation Reports

**Files**: `data/data_nea.ipynb`

**Implementation**:
```python
import pdfplumber
from pathlib import Path

# Open PDF files and extract tables
for pdf_path in pdf_files:
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[:10]:
            tables = page.extract_tables()
            for row in table:
                # Parse daily energy data and peak demand data
```

**Data Extracted**:
| Variable | Description | Unit |
|----------|-------------|------|
| Date(Nepali) | Nepali calendar date (Bikram Sambat) | - |
| Date(English) | Gregorian calendar date | - |
| Energy_generation_NEA | Energy generated by NEA power plants | MWh |
| Energy_generation_NEA Subsidiary | Energy from NEA subsidiary companies | MWh |
| Energy_generation_IPP | Energy from Independent Power Producers | MWh |
| Energy_generation_Import | Energy imported from India | MWh |
| Energy Export | Energy exported to India | MWh |
| Net Energy Met (INPS Demand) | Net domestic demand met | MWh |
| Energy Interruption | Energy lost due to outages | MWh |
| Energy Requirement | Total energy demand | MWh |

**Output**: 606 daily records from 18 PDF reports (NMOR 2079_04 to NMOR 2080_11)

---

#### 2. Weather Data Collection
**Task**: Fetch historical weather data from Open-Meteo API

**File**: `src/scraper/weather_scraper.py`

**Implementation**:
```python
class WeatherScraper:
    NEPAL_CITIES = {
        'Kathmandu': {'lat': 27.7172, 'lon': 85.3240},
        'Pokhara': {'lat': 28.2667, 'lon': 83.9667},
    }

    def get_historical_weather(self, city, start_date, end_date):
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            'latitude': coords['lat'],
            'longitude': coords['lon'],
            'start_date': start_date,
            'end_date': end_date,
            'hourly': 'temperature_2m,relative_humidity_2m,precipitation,windspeed_10m',
            'timezone': 'Asia/Kathmandu'
        }
        return response.json()
```

**Variables Collected**:
- Temperature (°C)
- Relative Humidity (%)
- Precipitation (mm)
- Wind Speed (km/h)

**Output**: 24,864 hourly records → aggregated to 518 daily records

---

#### 3. Holiday Calendar Compilation
**Task**: Compile Nepal public holidays for 2022-2024

**File**: `src/scraper/holiday_scraper.py`

**Implementation**:
```python
class HolidayScraper:
    def scrape_from_timeanddate(self, year):
        url = f"https://www.timeanddate.com/holidays/nepal/{year}"
        # Parse HTML table for holiday dates

    def _get_fallback_holidays(self, year):
        # Hardcoded fallback with known Nepali holidays
```

**Output**: 30 holiday records including Dashain, Tihar, Holi, etc.

---

#### 4. Data Cleaning & Preprocessing
**Task**: Clean and merge all datasets

**Steps**:
1. **Type Conversion**: Convert numeric columns from string (handling comma separators)
2. **Date Parsing**: Parse dates to datetime objects
3. **Deduplication**: Remove duplicate records based on date
4. **Sorting**: Sort chronologically for time-series analysis
5. **Missing Value Handling**: Forward fill for weather gaps, drop rows with missing lag features
6. **Dataset Merging**: Merge energy, weather, and holiday data

**Implementation**:
```python
def clean_energy_data(df):
    # Convert numeric columns
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')

    # Parse dates
    df['date'] = pd.to_datetime(df['Date(English)'], format='%d/%m/%Y')

    # Remove duplicates and sort
    df = df.drop_duplicates(subset=['date']).sort_values('date')
    return df
```

### Deliverables
| Deliverable | Location | Records |
|-------------|----------|---------|
| Raw weather data | `data/raw/weather_data.csv` | 24,864 hourly |
| Holiday calendar | `data/raw/holidays.csv` | 30 holidays |
| Cleaned energy data | Processed DataFrame | 606 daily |
| Weather scraper script | `src/scraper/weather_scraper.py` | - |
| Holiday scraper script | `src/scraper/holiday_scraper.py` | - |

---

## Member 2: EDA & Visualization

### Scope
Data exploration, pattern discovery, and visualization creation.

### Responsibilities

#### 1. Statistical Summary
**Task**: Compute and present basic statistics of energy data

**Key Statistics**:
| Metric | Value |
|--------|-------|
| Mean Daily Demand | 32,191 MWh |
| Maximum Demand | 42,190 MWh |
| Minimum Demand | 18,936 MWh |
| Standard Deviation | 4,613 MWh |

---

#### 2. Temporal Pattern Analysis
**Task**: Analyze energy consumption patterns over time

**Analyses Performed**:
- Daily energy requirement trend over 1.5 years
- 7-day rolling average to smooth daily volatility
- Day of week patterns (weekday vs weekend)
- Monthly patterns
- Seasonal variations

**Key Findings**:
| Season | Avg Demand (MWh) | Observation |
|--------|------------------|-------------|
| Summer | Highest (~35,000) | Cooling demand peak |
| Winter | Moderate (~32,000) | Heating needs |
| Monsoon | Variable | Hydro generation impact |
| Autumn | Festival spike | Dashain/Tihar effect |
| Spring | Lower (~30,000) | Mild weather |

---

#### 3. Generation Mix Analysis
**Task**: Analyze energy generation sources

**Key Findings**:
| Source | Avg Contribution (MWh) | Share |
|--------|------------------------|-------|
| IPP | 16,464 | 43% |
| NEA Subsidiary | 7,180 | 30% |
| NEA Own Generation | 7,935 | 23% |
| Import from India | 4,607 | 4% |

---

#### 4. Correlation Analysis
**Task**: Analyze relationships between variables

**Key Correlations**:
- Energy Requirement ↔ Total Energy Available: 0.99
- Energy Requirement ↔ INPS Demand: 0.95
- Temperature ↔ Energy Demand: Positive correlation
- Previous day demand ↔ Current demand: Strongest predictor

---

#### 5. Anomaly Detection
**Task**: Identify unusual consumption days using IQR method

**Method**:
```python
def detect_anomalies(series, method='iqr'):
    Q1, Q3 = series.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return (series < lower) | (series > upper)
```

**Output**: 7 anomalous days detected

---

#### 6. Visualizations Created
| Visualization | File | Purpose |
|---------------|------|---------|
| Energy Requirement Trend | `reports/energy_requirement_trend.png` | Show daily demand over time |
| Generation Mix Analysis | `reports/generation_analysis.png` | Energy source breakdown |
| Seasonal Patterns | `reports/seasonal_patterns.png` | Season and day-of-week patterns |
| Energy Heatmap | `reports/energy_heatmap.png` | Day-of-week vs Month patterns |
| Peak Demand Analysis | `reports/peak_demand_analysis.png` | Peak demand distribution |
| Correlation Matrix | `reports/correlation_matrix.png` | Variable relationships |
| Anomaly Detection | `reports/anomaly_detection.png` | Unusual consumption days |

### Deliverables
| Deliverable | Location |
|-------------|----------|
| EDA Notebook | `notebooks/01_eda_energy_data.ipynb` |
| All visualizations | `reports/*.png` |
| Cleaned data exports | `data/processed/` |

---

## Member 3: Feature Engineering & Model Development

### Scope
Feature creation and ML model implementation.

### Responsibilities

#### 1. Temporal Feature Engineering
**Task**: Create time-based features

**Features Created**:
| Feature | Description | Type |
|---------|-------------|------|
| day_of_week | Monday=0, Sunday=6 | Integer (0-6) |
| month | Calendar month | Integer (1-12) |
| day_of_year | Day number in year | Integer (1-366) |
| week_of_year | ISO week number | Integer (1-52) |
| quarter | Financial quarter | Integer (1-4) |
| season | Nepal season (0-4) | Categorical |

**Nepal Season Mapping**:
```python
season_map = {
    12: 0, 1: 0, 2: 0,  # Winter
    3: 1, 4: 1,         # Spring
    5: 2, 6: 2, 7: 2,   # Summer
    8: 3, 9: 3,         # Monsoon
    10: 4, 11: 4        # Autumn
}
```

---

#### 2. Weather Feature Integration
**Task**: Integrate weather data with energy data

**Features Created**:
| Feature | Description | Purpose |
|---------|-------------|---------|
| temp_mean | Daily average temperature | Overall temperature effect |
| temp_max | Daily maximum | Peak cooling demand |
| temp_min | Daily minimum | Heating requirements |
| temp_range | max - min | Temperature variability |
| humidity | Average humidity | Cooling load |
| precipitation | Daily sum | Hydro generation impact |
| windspeed | Average wind speed | Weather factor |

---

#### 3. Lag Feature Creation
**Task**: Create lag features for time-series prediction

**Features Created**:
| Feature | Description | Importance |
|---------|-------------|------------|
| demand_lag_1 | Previous day's demand | Very High |
| demand_lag_7 | Same day last week | High |
| demand_rolling_7 | 7-day moving average | Very High |

**Implementation**:
```python
df['demand_lag_1'] = df['Energy Requirement'].shift(1)
df['demand_lag_7'] = df['Energy Requirement'].shift(7)
df['demand_rolling_7'] = df['Energy Requirement'].rolling(window=7).mean()
```

---

#### 4. Binary Flag Creation
**Task**: Create categorical flags

**Features Created**:
| Feature | Values | Purpose |
|---------|--------|---------|
| is_weekend | 0/1 | Weekend demand patterns |
| is_holiday | 0/1 | Holiday demand patterns |

---

#### 5. Train-Test Split
**Task**: Split data for model training

**Method**: Time-based split (not random shuffle)

| Set | Size | Date Range |
|-----|------|------------|
| Training | 396 samples (80%) | July 2022 - August 2023 |
| Testing | 99 samples (20%) | August 2023 - November 2023 |

**Rationale**: Time-based split preserves temporal dependencies and simulates real-world forecasting scenarios.

---

#### 6. Feature Scaling
**Task**: Normalize features for linear models

**Method**: StandardScaler (Z-score normalization)

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Note**: Applied only to Linear Regression and Ridge Regression. Not applied to Random Forest and XGBoost (tree-based models are scale-invariant).

---

#### 7. ML Model Implementation
**Task**: Implement and train 4 machine learning models

##### Model 1: Linear Regression (Baseline)
```python
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
```
- Assumes linear relationship between features and target
- Fast training and prediction
- Prone to underfitting on complex patterns

##### Model 2: Ridge Regression
```python
from sklearn.linear_model import Ridge
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)
```
- L2 regularization penalizes large coefficients
- More robust than standard linear regression
- Handles multicollinearity better

##### Model 3: Random Forest Regressor
```python
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
```
- Ensemble method capturing non-linear relationships
- Robust to outliers
- Provides feature importance

##### Model 4: XGBoost Regressor
```python
import xgboost as xgb
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train)
```
- Gradient boosting for high accuracy
- State-of-the-art for tabular data
- Can overfit if not tuned properly

### Deliverables
| Deliverable | Description |
|-------------|-------------|
| Feature-engineered dataset | 495 records, 18 features |
| 4 trained models | Linear, Ridge, Random Forest, XGBoost |
| Feature scaling pipeline | StandardScaler fitted on training data |

---

## Member 4: Model Evaluation & Documentation

### Scope
Model comparison, analysis, and final documentation.

### Responsibilities

#### 1. Evaluation Metrics Calculation
**Task**: Calculate performance metrics for all models

**Metrics Used**:
| Metric | Formula | Interpretation |
|--------|---------|----------------|
| RMSE | √(Σ(y_actual - y_pred)² / n) | Lower is better, penalizes large errors |
| MAE | Σ\|y_actual - y_pred\| / n | Average absolute error |
| MAPE | Σ\|y_actual - y_pred\| / y_actual × 100 | Percentage error |
| R² | 1 - (SS_res / SS_tot) | Variance explained (0-1) |

---

#### 2. Model Performance Comparison
**Task**: Compare and rank all models

**Results**:
| Model | Test RMSE (MWh) | Test MAE (MWh) | Test MAPE (%) | Test R² |
|-------|-----------------|----------------|---------------|---------|
| **Ridge Regression** | **1,660.94** | **1,324.15** | **4.32%** | **0.8991** |
| Linear Regression | 1,662.57 | 1,328.28 | 4.34% | 0.8989 |
| Random Forest | 1,760.68 | 1,396.45 | 4.55% | 0.8866 |
| XGBoost | 1,964.87 | 1,594.81 | 5.06% | 0.8588 |

---

#### 3. Best Model Selection & Justification
**Task**: Select and justify the best model

**Selected Model**: Ridge Regression

**Justification**:
| Factor | Explanation |
|--------|-------------|
| Dataset Size | 495 records - relatively small for complex models |
| Feature Linearity | Energy demand has strong linear relationships with lag features |
| Regularization | Ridge's L2 penalty prevented overfitting |
| Tree Overfitting | Random Forest and XGBoost showed signs of overfitting (low train RMSE, higher test RMSE) |

---

#### 4. Residual Analysis
**Task**: Analyze prediction residuals

**Residual Statistics**:
| Metric | Value |
|--------|-------|
| Mean | ~0 MWh (unbiased predictions) |
| Std Dev | ~1,600 MWh |
| Min | -4,500 MWh |
| Max | +4,200 MWh |

**Finding**: Residuals are approximately normally distributed around zero, indicating good model calibration.

---

#### 5. Feature Importance Interpretation
**Task**: Identify and interpret most important features

**Top Predictors**:
| Rank | Feature | Importance |
|------|---------|------------|
| 1 | demand_rolling_7 | 7-day average demand - captures medium-term trend |
| 2 | demand_lag_1 | Previous day demand - captures short-term momentum |
| 3 | demand_lag_7 | Same day last week - captures weekly seasonality |
| 4 | temp_mean | Temperature - affects heating/cooling demand |
| 5 | day_of_week | Weekly pattern |

---

#### 6. Sample Predictions
**Task**: Demonstrate model predictions

**Sample Predictions from Test Set**:
| Date | Actual (MWh) | Predicted (MWh) | Error (%) |
|------|--------------|-----------------|-----------|
| 2023-08-24 | 33,052 | 34,930 | -5.68% |
| 2023-08-25 | 32,788 | 33,717 | -2.83% |
| 2023-08-26 | 30,524 | 32,357 | -6.01% |
| 2023-08-27 | 32,978 | 31,868 | +3.36% |
| 2023-08-28 | 35,623 | 33,159 | +6.92% |

---

#### 7. Model Persistence
**Task**: Save model for future use

```python
import joblib

# Save model and scaler
joblib.dump(ridge_model, 'models/best_model.joblib')
joblib.dump(scaler, 'models/scaler.joblib')
```

---

#### 8. Visualization Creation
**Task**: Create evaluation visualizations

| Visualization | File | Purpose |
|---------------|------|---------|
| Model Comparison | `reports/model_comparison.png` | Compare RMSE and R² across models |
| Prediction Results | `reports/prediction_results.png` | Actual vs Predicted comparison |
| Residual Analysis | `reports/residual_analysis.png` | Residual distribution and time plot |

### Deliverables
| Deliverable | Location |
|-------------|----------|
| Model comparison analysis | `notebooks/02_prediction_model.ipynb` |
| Residual analysis | `notebooks/02_prediction_model.ipynb` |
| Saved model | `models/best_model.joblib` |
| Saved scaler | `models/scaler.joblib` |
| Evaluation visualizations | `reports/*.png` |

---

# 4. Technical Implementation Details

## Technologies Used

| Category | Tools |
|----------|-------|
| Programming | Python 3.11 |
| Data Manipulation | pandas, numpy |
| PDF Extraction | pdfplumber |
| Web Scraping | requests, BeautifulSoup |
| Visualization | matplotlib, seaborn, plotly |
| Machine Learning | scikit-learn, XGBoost |
| Time Series | statsmodels, prophet |
| Model Persistence | joblib |
| IDE | Jupyter Notebook, VS Code |

## Project Structure

```
datascience_project/
├── data/
│   ├── NMOR*.pdf              # NEA Monthly Reports (18 files)
│   ├── raw/
│   │   ├── weather_data.csv   # Scraped weather data
│   │   └── holidays.csv       # Holiday calendar
│   └── processed/             # Cleaned datasets
├── notebooks/
│   ├── data_nea.ipynb         # PDF extraction
│   ├── 01_eda_energy_data.ipynb    # EDA notebook
│   └── 02_prediction_model.ipynb   # ML modeling notebook
├── src/
│   └── scraper/
│       ├── weather_scraper.py      # Weather API scraper
│       └── holiday_scraper.py      # Holiday scraper
├── models/                    # Saved model artifacts
│   ├── best_model.joblib
│   └── scaler.joblib
├── reports/                   # Generated visualizations
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## Final Dataset Summary

| Metric | Value |
|--------|-------|
| Total Records | 495 |
| Features | 18 |
| Target | Energy Requirement (MWh) |
| Date Range | July 2022 - November 2023 |
| Training Set | 396 samples (80%) |
| Test Set | 99 samples (20%) |

**Feature List**:
```
['temp_mean', 'temp_max', 'temp_min', 'humidity', 'precipitation',
 'windspeed', 'temp_range', 'day_of_week', 'month', 'day_of_year',
 'week_of_year', 'quarter', 'is_weekend', 'is_holiday', 'season',
 'demand_lag_1', 'demand_lag_7', 'demand_rolling_7']
```

---

# 5. Viva Questions & Answers

## Section A: Data Collection & Data Cleaning (Member 1)

### Q1: What data sources did you use for this project?
**Answer**: We used three primary data sources:
1. **NEA Monthly Operation Reports (PDFs)**: 18 PDF files containing daily energy generation, consumption, import/export data
2. **Open-Meteo Weather API**: Historical weather data including temperature, humidity, precipitation, and wind speed for Kathmandu and Pokhara
3. **Holiday Calendar**: Nepal's public holidays from 2022-2024 from timeanddate.com and hardcoded fallback data

---

### Q2: Why did you choose pdfplumber for PDF extraction?
**Answer**: pdfplumber was chosen because:
- It excels at extracting tables from PDFs while preserving structure
- It handles multi-page tables well
- It provides both text and positional information
- It's more reliable than alternatives like PyPDF2 for tabular data
- Free and open-source

---

### Q3: What challenges did you face during PDF extraction?
**Answer**: Key challenges included:
1. **Inconsistent date formats**: Some dates had single-digit day/month, others had leading zeros. Solved by normalizing to DD/MM/YYYY format
2. **Tables spanning multiple pages**: Concatenated rows from all pages
3. **Mixed data types in columns**: Used regex validation to classify rows (peak demand vs daily energy)
4. **Nepali vs English dates**: Stored both, used English dates for analysis

---

### Q4: How did you handle missing values in the dataset?
**Answer**:
- For weather data: Used forward fill for small gaps in continuous data
- For lag features: Dropped rows where lag features couldn't be calculated (first 7 days)
- For numeric conversion: Used `errors='coerce'` to convert invalid values to NaN, then handled appropriately
- No imputation was needed for the main energy dataset as NEA reports had complete data

---

### Q5: Why did you collect weather data for only Kathmandu and Pokhara?
**Answer**:
- These two cities represent major population and industrial centers
- Kathmandu Valley accounts for significant portion of Nepal's energy consumption
- API rate limiting and processing efficiency
- Weather patterns are reasonably representative of the central region where most demand centers are located
- Future improvement: Expand to more cities for better spatial coverage

---

### Q6: What is the Open-Meteo API and why did you choose it?
**Answer**: Open-Meteo is a free, open-source weather API that provides:
- Historical weather data without requiring an API key
- Hourly and daily weather variables
- Global coverage with reasonable accuracy
- No rate limiting for moderate use
- We chose it because it's completely free and requires no authentication, unlike OpenWeatherMap which requires an API key

---

### Q7: How many data records did you extract and what is the date range?
**Answer**:
- **Energy data**: 606 daily records from July 17, 2022 to March 13, 2024
- **Weather data**: 24,864 hourly records, aggregated to 518 daily records
- **Holiday data**: 30 holidays for 2022-2024
- **Final merged dataset**: 495 records (after dropping NaN from lag features)

---

### Q8: What is the difference between daily energy data and peak demand data?
**Answer**:
- **Daily energy data**: Total energy (MWh) generated/consumed over 24 hours - used for demand forecasting
- **Peak demand data**: Maximum power (MW) drawn at a specific time during the day - used for capacity planning
- Energy is measured in MWh (energy), peak demand in MW (power)
- Our project focused on daily energy forecasting for operational planning

---

## Section B: EDA & Visualization (Member 2)

### Q9: What key patterns did you discover in the energy consumption data?
**Answer**:
1. **Seasonal patterns**: Summer (May-July) has highest demand due to cooling needs; Spring has lowest demand
2. **Weekly patterns**: Weekends have lower demand due to reduced industrial activity
3. **Festival impact**: Major festivals like Dashain and Tihar show distinct consumption changes
4. **Upward trend**: Energy demand shows gradual increase over the study period
5. **Generation mix**: IPPs contribute 43% of total generation - the largest share

---

### Q10: Why is 7-day rolling average used in the analysis?
**Answer**:
- Daily data has inherent volatility due to random fluctuations
- 7-day rolling average smooths out daily noise while preserving trends
- Weekly window aligns with the natural 7-day cycle in energy consumption
- Helps identify underlying patterns more clearly
- Used both as analysis tool and as a predictive feature (demand_rolling_7)

---

### Q11: What does the correlation matrix reveal about energy demand?
**Answer**:
- **Strong positive correlations**:
  - Energy Requirement ↔ Total Energy Available (0.99) - demand is almost always met
  - Energy Requirement ↔ INPS Demand (0.95) - strong relationship
  - NEA Generation ↔ NEA Subsidiary (0.85) - correlated generation patterns
- **Weather correlations**: Temperature shows positive correlation with demand (higher temp → higher demand)
- **Key insight**: Lag features (previous day demand) are the strongest predictors

---

### Q12: How did you detect anomalies in the data?
**Answer**: We used the IQR (Interquartile Range) method:
1. Calculate Q1 (25th percentile) and Q3 (75th percentile)
2. Compute IQR = Q3 - Q1
3. Define bounds: Lower = Q1 - 1.5×IQR, Upper = Q3 + 1.5×IQR
4. Flag points outside these bounds as anomalies

Found 7 anomalous days, often corresponding to major holidays or grid disturbances.

---

### Q13: What is the generation mix in Nepal and why does it matter?
**Answer**:
| Source | Share |
|--------|-------|
| IPP (Independent Power Producers) | 43% |
| NEA Subsidiary | 30% |
| NEA Own Generation | 23% |
| Import from India | 4% |

This matters because:
- IPPs are crucial for grid stability - their contribution must be considered in planning
- Low import dependency (4%) indicates good self-sufficiency
- Generation mix affects forecasting - different sources have different patterns

---

### Q14: Why do weekends have lower energy demand?
**Answer**:
- **Industrial shutdown**: Most factories and industries operate on weekdays only
- **Commercial reduction**: Offices, banks, and businesses are closed
- **Schools closed**: Educational institutions consume less energy
- **Government offices closed**: Significant portion of institutional consumption reduced
- The weekend effect (Saturday-Sunday in Nepal) is consistent and predictable

---

### Q15: What visualization libraries did you use and why?
**Answer**:
- **matplotlib**: Primary plotting library, highly customizable, good for static visualizations
- **seaborn**: Built on matplotlib, provides better default aesthetics, excellent for statistical visualizations like heatmaps
- **plotly**: Listed in requirements for potential interactive visualizations

We chose seaborn for correlation matrices and statistical plots due to its built-in support for such visualizations.

---

## Section C: Feature Engineering & Model Development (Member 3)

### Q16: What is feature engineering and why is it important?
**Answer**: Feature engineering is the process of creating new input variables from raw data to improve model performance. It's important because:
- Raw data often doesn't capture all relevant patterns
- Good features can dramatically improve model accuracy
- Domain knowledge can be encoded into features
- Machine learning models learn from features, not raw data directly

In our project, feature engineering transformed raw dates and weather into 18 predictive features.

---

### Q17: Why are lag features important for time series forecasting?
**Answer**: Lag features are crucial because:
- **Autocorrelation**: Energy demand has strong temporal dependencies - today's demand is correlated with yesterday's
- **Pattern capture**: demand_lag_1 captures short-term momentum, demand_lag_7 captures weekly seasonality
- **Trend information**: Rolling averages capture medium-term trends
- Our analysis showed lag features are the strongest predictors of energy demand

---

### Q18: Explain the different lag features you created.
**Answer**:
| Feature | Description | Captures |
|---------|-------------|----------|
| demand_lag_1 | Previous day's demand | Short-term momentum, immediate trends |
| demand_lag_7 | Same day last week | Weekly seasonality (Monday↔Monday patterns) |
| demand_rolling_7 | 7-day moving average | Medium-term trend, smooths volatility |

The 7-day rolling average was the most important predictor in our model.

---

### Q19: Why did you use a time-based train-test split instead of random split?
**Answer**:
- **Temporal dependencies**: Random split would break time-series patterns
- **Real-world simulation**: In practice, we forecast future based on past - time split simulates this
- **Prevents data leakage**: Random split could have future data in training set
- **Valid evaluation**: Time-based split gives realistic performance estimates

We used 80% oldest data for training, 20% newest for testing.

---

### Q20: Why did you apply StandardScaler only to some models?
**Answer**:
- **Linear Regression and Ridge Regression**: These models are sensitive to feature scales. Features with larger ranges would dominate the model. StandardScaler normalizes all features to mean=0, std=1.
- **Random Forest and XGBoost**: These are tree-based models that split data based on ordering, not absolute values. They are scale-invariant and don't require normalization.

---

### Q21: What is the difference between Ridge Regression and Linear Regression?
**Answer**:
- **Linear Regression**: Minimizes sum of squared errors. Can overfit when features are correlated or dataset is small.
- **Ridge Regression**: Adds L2 regularization term (α × Σβ²) to the loss function. This penalizes large coefficients, reducing overfitting.

Ridge performed best in our project because:
- Small dataset (495 records)
- Regularization prevented overfitting
- Better handles multicollinearity among features

---

### Q22: Why did you choose these four specific ML models?
**Answer**:
1. **Linear Regression**: Simple baseline, interpretable, fast
2. **Ridge Regression**: Regularized linear model, handles multicollinearity
3. **Random Forest**: Captures non-linear relationships, provides feature importance
4. **XGBoost**: State-of-the-art for tabular data, handles complex patterns

This covers the spectrum from simple to complex models, allowing fair comparison.

---

### Q23: What hyperparameters did you tune for each model?
**Answer**:
- **Linear Regression**: No hyperparameters to tune
- **Ridge Regression**: alpha=1.0 (default regularization strength)
- **Random Forest**: n_estimators=100, max_depth=15, min_samples_split=5, min_samples_leaf=2
- **XGBoost**: n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8

These were chosen based on best practices for similar problems.

---

### Q24: What is season encoding and how did you implement it?
**Answer**: Season encoding maps months to seasons:
```python
season_map = {
    12: 0, 1: 0, 2: 0,  # Winter
    3: 1, 4: 1,         # Spring
    5: 2, 6: 2, 7: 2,   # Summer
    8: 3, 9: 3,         # Monsoon
    10: 4, 11: 4        # Autumn
}
```

Nepal has 5 distinct seasons based on climate patterns, which affect energy consumption differently.

---

## Section D: Model Evaluation & Results (Member 4)

### Q25: What evaluation metrics did you use and why?
**Answer**:
| Metric | Why Used |
|--------|----------|
| RMSE | Penalizes large errors more heavily, good for operational planning where large errors are costly |
| MAE | Average absolute error, easy to interpret in MWh units |
| MAPE | Percentage error, allows comparison across different scales |
| R² | Variance explained, indicates how well model captures patterns |

Using multiple metrics provides comprehensive evaluation.

---

### Q26: Why did Ridge Regression outperform XGBoost and Random Forest?
**Answer**:
1. **Dataset size**: 495 records is small for complex models
2. **Feature linearity**: Energy demand has strong linear relationships with lag features
3. **Regularization benefit**: Ridge's L2 penalty prevented overfitting
4. **Tree model overfitting**: Random Forest and XGBoost showed lower training RMSE but higher test RMSE (overfitting signs)
5. **Simplicity principle**: Occam's razor - simpler models generalize better on small datasets

---

### Q27: What does R² = 0.90 mean?
**Answer**: R² = 0.90 means:
- The model explains 90% of the variance in energy demand
- Only 10% of variation is unexplained (random noise or missing features)
- This is considered good performance for energy forecasting
- Industry standard: R² > 0.85 is acceptable for operational planning

---

### Q28: What is residual analysis and why is it important?
**Answer**: Residual analysis examines the difference between actual and predicted values:
- **Purpose**: Check if model is well-calibrated and assumptions are met
- **Ideal**: Residuals should be normally distributed around zero
- **Our results**: Mean ~0 MWh (unbiased), Std ~1,600 MWh, approximately normal distribution
- **Importance**: Identifies systematic biases, heteroscedasticity, or model inadequacies

---

### Q29: How do you interpret the MAPE of 4.32%?
**Answer**:
- MAPE (Mean Absolute Percentage Error) = 4.32%
- On average, predictions deviate from actual values by about 4.32%
- For energy demand of ~32,000 MWh, this means average error of ~1,380 MWh
- Industry standard: MAPE < 5% is considered good for short-term forecasting
- This accuracy is suitable for operational planning decisions

---

### Q30: What are the most important features in the model?
**Answer**:
| Rank | Feature | Why Important |
|------|---------|---------------|
| 1 | demand_rolling_7 | Captures medium-term demand trend |
| 2 | demand_lag_1 | Short-term momentum in consumption |
| 3 | demand_lag_7 | Weekly seasonality pattern |
| 4 | temp_mean | Temperature affects cooling/heating demand |
| 5 | day_of_week | Captures weekday/weekend pattern |

Lag features dominate because energy demand has strong temporal autocorrelation.

---

### Q31: How would you use this model for prediction in practice?
**Answer**:
```python
import joblib

# Load model and scaler
model = joblib.load('models/best_model.joblib')
scaler = joblib.load('models/scaler.joblib')

# Prepare input features for tomorrow
sample_data = {
    'temp_mean': 25.0,      # Weather forecast
    'temp_max': 30.0,
    'temp_min': 20.0,
    'humidity': 75.0,
    'precipitation': 5.0,
    'windspeed': 3.5,
    'temp_range': 10.0,
    'day_of_week': 2,       # Wednesday
    'month': 8,
    'day_of_year': 220,
    'week_of_year': 32,
    'quarter': 3,
    'is_weekend': 0,
    'is_holiday': 0,
    'season': 3,
    'demand_lag_1': 32000,  # Today's demand
    'demand_lag_7': 31500,  # Last week same day
    'demand_rolling_7': 31800
}

sample_scaled = scaler.transform(sample_df)
predicted_demand = model.predict(sample_scaled)
```

---

### Q32: What are the limitations of this project?
**Answer**:
1. **Limited data period**: 1.5 years may not capture all seasonal variations
2. **Weather data scope**: Only 2 cities; more comprehensive coverage would improve accuracy
3. **Peak time data**: Not fully extracted due to PDF format inconsistencies
4. **No real-time integration**: Currently uses historical data only
5. **No uncertainty quantification**: Model doesn't provide prediction intervals
6. **Single-country focus**: Could be extended to regional analysis

---

### Q33: What future improvements would you suggest?
**Answer**:
1. **Deep Learning**: Explore LSTM/GRU for sequence modeling
2. **Prophet**: Facebook's time-series library with built-in holiday handling
3. **Hyperparameter tuning**: GridSearchCV or Bayesian optimization
4. **Real-time dashboard**: Plotly Dash or Streamlit for interactive visualization
5. **Automated pipeline**: Apache Airflow for scheduled ETL
6. **Prediction intervals**: Add uncertainty quantification
7. **Extended data collection**: More historical data for better pattern capture

---

### Q34: How can this model be deployed in a real-world scenario?
**Answer**:
1. **Daily batch prediction**: Run model each morning with latest data
2. **Integration with SCADA**: Connect to Nepal's grid management systems
3. **API service**: Expose model as REST API for other systems
4. **Dashboard**: Visual interface for operators to view predictions
5. **Alert system**: Trigger warnings when predicted demand approaches capacity
6. **Scheduled updates**: Retrain model periodically with new data

---

### Q35: What is the business value of this forecasting system?
**Answer**:
| Application | Benefit |
|-------------|---------|
| Demand Forecasting | Plan generation schedules 1-7 days ahead |
| Import Planning | Optimize cross-border energy exchange with India |
| Peak Management | Prepare for high-demand periods |
| Cost Reduction | Minimize expensive peak-time imports |
| Grid Stability | Ensure adequate generation to meet demand |
| Policy Support | Data-driven insights for energy policy |

---

## Section E: General Project Questions

### Q36: What was the biggest challenge in this project?
**Answer**: PDF extraction was the most challenging aspect:
- Tables spanning multiple pages
- Inconsistent date formats
- Mixed data types in columns
- Nepali and English date systems

Solved by using pdfplumber with careful row classification logic based on column content patterns.

---

### Q37: How does this project help Nepal's energy sector?
**Answer**:
- **Demand forecasting**: Enables better planning for power generation
- **Import optimization**: Helps schedule cross-border energy exchange
- **Grid reliability**: Reduces risk of supply-demand mismatch
- **Cost savings**: Better planning reduces expensive emergency imports
- **Renewable integration**: Helps integrate more variable renewable sources

---

### Q38: What tools and technologies did you use?
**Answer**:
| Category | Tools |
|----------|-------|
| Programming | Python 3.11 |
| Data Processing | pandas, numpy |
| PDF Extraction | pdfplumber |
| Web Scraping | requests, BeautifulSoup |
| Visualization | matplotlib, seaborn |
| Machine Learning | scikit-learn, XGBoost |
| IDE | Jupyter Notebook |
| Version Control | Git |

---

### Q39: How would you validate that the model is working correctly?
**Answer**:
1. **Cross-validation**: Time-series cross-validation on training data
2. **Hold-out testing**: Final evaluation on unseen test data
3. **Residual analysis**: Check for systematic patterns in errors
4. **Business validation**: Compare predictions with domain expert expectations
5. **Backtesting**: Test model on historical periods to see how it would have performed

---

### Q40: What did you learn from this project?
**Answer**:
1. **Data extraction**: PDF parsing is complex but manageable with right tools
2. **Feature engineering**: Domain knowledge is crucial for creating effective features
3. **Model selection**: Complex models aren't always better - match model to data
4. **Time series**: Temporal dependencies require special handling
5. **Real-world data**: Requires significant preprocessing effort
6. **Nepal's energy sector**: Growing demand, strong IPP contribution, improving self-sufficiency

---

# Quick Reference: Model Performance Summary

| Model | RMSE (MWh) | MAE (MWh) | MAPE (%) | R² | Rank |
|-------|------------|-----------|----------|-----|------|
| Ridge Regression | 1,660.94 | 1,324.15 | 4.32 | 0.8991 | 1st |
| Linear Regression | 1,662.57 | 1,328.28 | 4.34 | 0.8989 | 2nd |
| Random Forest | 1,760.68 | 1,396.45 | 4.55 | 0.8866 | 3rd |
| XGBoost | 1,964.87 | 1,594.81 | 5.06 | 0.8588 | 4th |

---

*Document prepared for Smart Energy Consumption Forecasting Project Viva*
*Group of 4 Members*