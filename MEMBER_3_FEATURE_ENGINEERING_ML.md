# Member 3: Feature Engineering & Model Development

## Smart Energy Consumption Forecasting & Optimization System
### Nepal Electricity Authority (NEA) Daily Operations Analysis

---

## Table of Contents
1. [Role Overview](#1-role-overview)
2. [Responsibilities](#2-responsibilities)
3. [Feature Engineering](#3-feature-engineering)
4. [Model Development](#4-model-development)
5. [Deliverables](#5-deliverables)
6. [Viva Questions & Answers](#6-viva-questions--answers)

---

# 1. Role Overview

## Scope
Feature creation, data transformation, and machine learning model implementation.

## Primary Tasks
- Create temporal features from date data
- Integrate weather features with energy data
- Design and implement lag features for time-series prediction
- Create binary flags for categorical patterns
- Perform train-test split using time-based methodology
- Apply feature scaling for appropriate models
- Implement and train 4 machine learning models

## Models Implemented
1. Linear Regression (Baseline)
2. Ridge Regression (Regularized)
3. Random Forest Regressor (Ensemble)
4. XGBoost Regressor (Gradient Boosting)

---

# 2. Responsibilities

## 2.1 Temporal Feature Engineering

### Task Description
Create time-based features from date column.

### Features Created

| Feature | Description | Type | Values |
|---------|-------------|------|--------|
| day_of_week | Day of the week | Integer | 0-6 (Mon-Sun) |
| month | Calendar month | Integer | 1-12 |
| day_of_year | Day number in year | Integer | 1-366 |
| week_of_year | ISO week number | Integer | 1-52 |
| quarter | Financial quarter | Integer | 1-4 |
| season | Nepal season | Categorical | 0-4 |

### Nepal Season Encoding
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

## 2.2 Weather Feature Integration

### Task Description
Merge and transform weather data with energy data.

### Features Created

| Feature | Description | Purpose |
|---------|-------------|---------|
| temp_mean | Daily average temperature | Overall temperature effect |
| temp_max | Daily maximum temperature | Peak cooling demand indicator |
| temp_min | Daily minimum temperature | Heating requirements |
| temp_range | temp_max - temp_min | Temperature variability |
| humidity | Average humidity | Cooling load indicator |
| precipitation | Daily precipitation sum | Hydro generation impact |
| windspeed | Average wind speed | Weather factor |

---

## 2.3 Lag Feature Creation

### Task Description
Create lag features to capture temporal autocorrelation in energy demand.

### Features Created

| Feature | Description | Importance |
|---------|-------------|------------|
| demand_lag_1 | Previous day's demand | Very High |
| demand_lag_7 | Same day last week's demand | High |
| demand_rolling_7 | 7-day moving average | Very High |

### Why Lag Features Matter
- Energy demand has strong temporal dependencies
- Previous day's demand is the strongest predictor
- 7-day lag captures weekly seasonality
- Rolling average captures medium-term trends

---

## 2.4 Binary Flag Creation

### Task Description
Create categorical flags for special days.

### Features Created

| Feature | Values | Purpose |
|---------|--------|---------|
| is_weekend | 0/1 | Capture weekend demand reduction |
| is_holiday | 0/1 | Capture holiday demand patterns |

---

## 2.5 Train-Test Split

### Task Description
Split data for model training using time-based methodology.

### Split Strategy
| Set | Size | Date Range |
|-----|------|------------|
| Training | 396 samples (80%) | July 2022 - August 2023 |
| Testing | 99 samples (20%) | August 2023 - November 2023 |

### Why Time-Based Split?
- Preserves temporal dependencies
- Simulates real-world forecasting scenario
- Prevents data leakage from future to past
- More realistic performance evaluation

---

## 2.6 Feature Scaling

### Task Description
Normalize features for linear models.

### Scaling Method
**StandardScaler** (Z-score normalization):
- Mean = 0, Standard Deviation = 1
- Formula: z = (x - μ) / σ

### Application
| Model | Scaling Applied? | Reason |
|-------|------------------|--------|
| Linear Regression | Yes | Sensitive to feature scales |
| Ridge Regression | Yes | Sensitive to feature scales |
| Random Forest | No | Tree-based, scale-invariant |
| XGBoost | No | Tree-based, scale-invariant |

---

# 3. Feature Engineering

## 3.1 Complete Feature Engineering Code

```python
import pandas as pd
import numpy as np

# Load all data sources
energy_df = pd.read_csv('data/processed/energy_clean.csv')
weather_df = pd.read_csv('data/raw/weather_data.csv')
holidays_df = pd.read_csv('data/raw/holidays.csv')

# Parse dates
energy_df['date'] = pd.to_datetime(energy_df['date'])
weather_df['date'] = pd.to_datetime(weather_df['date'])
holidays_df['date'] = pd.to_datetime(holidays_df['date'])

# Aggregate weather to daily level
daily_weather = weather_df.groupby('date').agg({
    'temperature_c': ['mean', 'max', 'min', 'std'],
    'humidity_percent': 'mean',
    'precipitation_mm': 'sum',
    'windspeed_kmh': 'mean'
}).reset_index()

daily_weather.columns = ['date', 'temp_mean', 'temp_max', 'temp_min', 'temp_std',
                         'humidity', 'precipitation', 'windspeed']
daily_weather['temp_range'] = daily_weather['temp_max'] - daily_weather['temp_min']

# Merge datasets
df = energy_df[['date', 'Energy Requirement']].copy()
df = df.merge(daily_weather, on='date', how='left')

# === TEMPORAL FEATURES ===
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['day_of_year'] = df['date'].dt.dayofyear
df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
df['quarter'] = df['date'].dt.quarter

# Season encoding (Nepal-specific)
season_map = {
    12: 0, 1: 0, 2: 0,  # Winter
    3: 1, 4: 1,         # Spring
    5: 2, 6: 2, 7: 2,   # Summer
    8: 3, 9: 3,         # Monsoon
    10: 4, 11: 4        # Autumn
}
df['season'] = df['month'].map(season_map)

# === BINARY FLAGS ===
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
holiday_dates = set(holidays_df['date'])
df['is_holiday'] = df['date'].isin(holiday_dates).astype(int)

# === LAG FEATURES ===
df['demand_lag_1'] = df['Energy Requirement'].shift(1)
df['demand_lag_7'] = df['Energy Requirement'].shift(7)
df['demand_rolling_7'] = df['Energy Requirement'].rolling(window=7).mean()

# Drop rows with NaN from lag features
df = df.dropna().reset_index(drop=True)

print(f"Final dataset: {len(df)} records, {len(df.columns)} columns")
```

---

## 3.2 Feature List Summary

### Final Feature Set (18 Features)

```python
feature_cols = [
    # Weather Features (7)
    'temp_mean', 'temp_max', 'temp_min', 'humidity',
    'precipitation', 'windspeed', 'temp_range',

    # Temporal Features (6)
    'day_of_week', 'month', 'day_of_year',
    'week_of_year', 'quarter', 'season',

    # Binary Flags (2)
    'is_weekend', 'is_holiday',

    # Lag Features (3)
    'demand_lag_1', 'demand_lag_7', 'demand_rolling_7'
]

target_col = 'Energy Requirement'
```

---

# 4. Model Development

## 4.1 Data Preparation

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Prepare features and target
X = df[feature_cols]
y = df[target_col]

# Time-based split (80-20)
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Training: {len(X_train)} samples")
print(f"Testing: {len(X_test)} samples")

# Scale features for linear models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## 4.2 Model 1: Linear Regression (Baseline)

### Description
Simple linear model assuming linear relationship between features and target.

### Implementation
```python
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
```

### Characteristics
- Assumes linear relationship
- Fast training and prediction
- Prone to underfitting on complex patterns
- Good baseline for comparison

---

## 4.3 Model 2: Ridge Regression

### Description
Linear model with L2 regularization to prevent overfitting.

### Implementation
```python
from sklearn.linear_model import Ridge

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)
y_pred_ridge = ridge_model.predict(X_test_scaled)
```

### Characteristics
- L2 regularization: penalty = α × Σβ²
- Prevents overfitting by shrinking coefficients
- Handles multicollinearity better
- Robust on small datasets

### Hyperparameter
- **alpha = 1.0**: Regularization strength (default)

---

## 4.4 Model 3: Random Forest Regressor

### Description
Ensemble of decision trees using bagging (bootstrap aggregating).

### Implementation
```python
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(
    n_estimators=100,      # Number of trees
    max_depth=15,          # Maximum tree depth
    min_samples_split=5,   # Minimum samples to split node
    min_samples_leaf=2,    # Minimum samples in leaf
    random_state=42,
    n_jobs=-1              # Use all CPU cores
)
rf_model.fit(X_train, y_train)  # No scaling needed
y_pred_rf = rf_model.predict(X_test)
```

### Characteristics
- Captures non-linear relationships
- Robust to outliers
- Provides feature importance
- Can overfit if not tuned

### Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| n_estimators | 100 | Number of trees in forest |
| max_depth | 15 | Limit tree complexity |
| min_samples_split | 5 | Prevent overfitting |
| min_samples_leaf | 2 | Ensure meaningful leaves |

---

## 4.5 Model 4: XGBoost Regressor

### Description
Gradient boosting algorithm, state-of-the-art for tabular data.

### Implementation
```python
import xgboost as xgb

xgb_model = xgb.XGBRegressor(
    n_estimators=100,      # Number of boosting rounds
    max_depth=6,           # Maximum tree depth
    learning_rate=0.1,     # Step size shrinkage
    subsample=0.8,         # Row sampling ratio
    colsample_bytree=0.8,  # Column sampling ratio
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)  # No scaling needed
y_pred_xgb = xgb_model.predict(X_test)
```

### Characteristics
- Sequential ensemble (boosting)
- Handles missing values
- Built-in regularization
- Can overfit if not tuned

### Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| n_estimators | 100 | Number of boosting rounds |
| max_depth | 6 | Control tree complexity |
| learning_rate | 0.1 | Shrink step size |
| subsample | 0.8 | Random row sampling |
| colsample_bytree | 0.8 | Random column sampling |

---

## 4.6 Evaluation Function

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name, scaled=False):
    """Train and evaluate a model, returning metrics."""

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100

    print(f"\n{'='*50}")
    print(f"{model_name} Results")
    print(f"{'='*50}")
    print(f"Train RMSE: {train_rmse:.2f} MWh")
    print(f"Test RMSE:  {test_rmse:.2f} MWh")
    print(f"Test MAE:   {test_mae:.2f} MWh")
    print(f"Test MAPE:  {test_mape:.2f}%")
    print(f"Test R²:    {test_r2:.4f}")

    return {
        'model': model,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_mape': test_mape,
        'test_r2': test_r2,
        'y_test_pred': y_test_pred
    }
```

---

# 5. Deliverables

## Files Modified

| File | Description |
|------|-------------|
| `notebooks/02_prediction_model.ipynb` | Complete modeling notebook |

## Datasets Created

| Dataset | Records | Features | Description |
|---------|---------|----------|-------------|
| Training Set | 396 | 18 | For model training |
| Test Set | 99 | 18 | For model evaluation |

## Models Trained

| Model | Status | Notes |
|-------|--------|-------|
| Linear Regression | Trained | Baseline model |
| Ridge Regression | Trained | Best performer |
| Random Forest | Trained | Ensemble model |
| XGBoost | Trained | Gradient boosting |

---

# 6. Viva Questions & Answers

## Q1: What is feature engineering and why is it important?
**Answer**: Feature engineering is the process of creating new input variables from raw data to improve model performance. It's important because:
- Raw data often doesn't capture all relevant patterns
- Good features can dramatically improve model accuracy
- Domain knowledge can be encoded into features
- Machine learning models learn from features, not raw data directly

In our project, feature engineering transformed raw dates and weather into 18 predictive features.

---

## Q2: Why are lag features important for time series forecasting?
**Answer**: Lag features are crucial because:
- **Autocorrelation**: Energy demand has strong temporal dependencies - today's demand is correlated with yesterday's
- **Pattern capture**: demand_lag_1 captures short-term momentum, demand_lag_7 captures weekly seasonality
- **Trend information**: Rolling averages capture medium-term trends
- Our analysis showed lag features are the strongest predictors of energy demand

---

## Q3: Explain the different lag features you created.
**Answer**:

| Feature | Description | Captures |
|---------|-------------|----------|
| demand_lag_1 | Previous day's demand | Short-term momentum, immediate trends |
| demand_lag_7 | Same day last week | Weekly seasonality (Monday↔Monday patterns) |
| demand_rolling_7 | 7-day moving average | Medium-term trend, smooths volatility |

The 7-day rolling average was the most important predictor in our model.

---

## Q4: Why did you use a time-based train-test split instead of random split?
**Answer**:
- **Temporal dependencies**: Random split would break time-series patterns
- **Real-world simulation**: In practice, we forecast future based on past - time split simulates this
- **Prevents data leakage**: Random split could have future data in training set
- **Valid evaluation**: Time-based split gives realistic performance estimates

We used 80% oldest data for training, 20% newest for testing.

---

## Q5: Why did you apply StandardScaler only to some models?
**Answer**:
- **Linear Regression and Ridge Regression**: These models are sensitive to feature scales. Features with larger ranges would dominate the model. StandardScaler normalizes all features to mean=0, std=1.
- **Random Forest and XGBoost**: These are tree-based models that split data based on ordering, not absolute values. They are scale-invariant and don't require normalization.

---

## Q6: What is the difference between Ridge Regression and Linear Regression?
**Answer**:
- **Linear Regression**: Minimizes sum of squared errors. Can overfit when features are correlated or dataset is small.
- **Ridge Regression**: Adds L2 regularization term (α × Σβ²) to the loss function. This penalizes large coefficients, reducing overfitting.

Ridge performed best in our project because:
- Small dataset (495 records)
- Regularization prevented overfitting
- Better handles multicollinearity among features

---

## Q7: Why did you choose these four specific ML models?
**Answer**:
1. **Linear Regression**: Simple baseline, interpretable, fast
2. **Ridge Regression**: Regularized linear model, handles multicollinearity
3. **Random Forest**: Captures non-linear relationships, provides feature importance
4. **XGBoost**: State-of-the-art for tabular data, handles complex patterns

This covers the spectrum from simple to complex models, allowing fair comparison.

---

## Q8: What hyperparameters did you tune for each model?
**Answer**:
- **Linear Regression**: No hyperparameters to tune
- **Ridge Regression**: alpha=1.0 (default regularization strength)
- **Random Forest**: n_estimators=100, max_depth=15, min_samples_split=5, min_samples_leaf=2
- **XGBoost**: n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8

These were chosen based on best practices for similar problems.

---

## Q9: What is season encoding and how did you implement it?
**Answer**: Season encoding maps months to numerical seasons:

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

## Q10: How does the number of features (18) relate to the dataset size (495)?
**Answer**:
- **Rule of thumb**: At least 10-20 samples per feature
- **Our ratio**: 495/18 ≈ 27.5 samples per feature
- **Implication**: Adequate for linear models, but may be tight for complex models
- **Result**: Ridge Regression (simpler model) performed best, confirming this analysis

---

## Q11: Why is max_depth limited in Random Forest and XGBoost?
**Answer**:
- **Prevent overfitting**: Deeper trees memorize training data
- **Generalization**: Shallower trees generalize better to new data
- **Our choice**: max_depth=15 for RF, max_depth=6 for XGBoost
- XGBoost typically needs shallower trees due to boosting nature

---

## Q12: What is the role of learning_rate in XGBoost?
**Answer**:
- **Shrinkage**: Smaller learning_rate means each tree contributes less
- **Trade-off**: Lower learning_rate needs more trees (n_estimators)
- **Our choice**: 0.1 is a balanced value
- Lower values (0.01) would require more boosting rounds

---

## Quick Reference: Model Configurations

### Linear Regression
```python
LinearRegression()  # No hyperparameters
```

### Ridge Regression
```python
Ridge(alpha=1.0)
```

### Random Forest
```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
```

### XGBoost
```python
XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
```

---

## Quick Reference: Feature Summary

| Category | Features | Count |
|----------|----------|-------|
| Weather | temp_mean, temp_max, temp_min, humidity, precipitation, windspeed, temp_range | 7 |
| Temporal | day_of_week, month, day_of_year, week_of_year, quarter, season | 6 |
| Binary | is_weekend, is_holiday | 2 |
| Lag | demand_lag_1, demand_lag_7, demand_rolling_7 | 3 |
| **Total** | | **18** |

---

*Document for Member 3 - Feature Engineering & Model Development*
*Smart Energy Consumption Forecasting Project*