"""
Data Loader Utility
Loads and caches processed energy data, weather data, and holidays.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from functools import lru_cache
from typing import Optional, Tuple


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = PROJECT_ROOT / "models"


@lru_cache(maxsize=1)
def load_energy_data() -> pd.DataFrame:
    """
    Load processed energy data from CSV.

    Returns:
        DataFrame with energy generation and consumption data
    """
    energy_path = PROCESSED_DIR / "energy_clean.csv"

    if not energy_path.exists():
        raise FileNotFoundError(
            f"Energy data not found at {energy_path}. "
            "Please run the data processing notebook first."
        )

    df = pd.read_csv(energy_path)

    # Convert date column
    df['date'] = pd.to_datetime(df['date'])

    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)

    return df


@lru_cache(maxsize=1)
def load_weather_data() -> pd.DataFrame:
    """
    Load weather data from CSV.

    Returns:
        DataFrame with weather data
    """
    weather_path = RAW_DIR / "weather_data.csv"

    if not weather_path.exists():
        raise FileNotFoundError(
            f"Weather data not found at {weather_path}. "
            "Please run the weather scraper first."
        )

    df = pd.read_csv(weather_path)
    df['date'] = pd.to_datetime(df['date'])

    return df


@lru_cache(maxsize=1)
def load_holidays() -> pd.DataFrame:
    """
    Load holiday calendar from CSV.

    Returns:
        DataFrame with holiday data
    """
    holidays_path = RAW_DIR / "holidays.csv"

    if not holidays_path.exists():
        # Return empty DataFrame if no holidays file
        return pd.DataFrame(columns=['date', 'holiday_name', 'type', 'year'])

    df = pd.read_csv(holidays_path)
    df['date'] = pd.to_datetime(df['date'])

    return df


def get_holiday_dates() -> set:
    """
    Get set of holiday dates.

    Returns:
        Set of datetime dates that are holidays
    """
    holidays_df = load_holidays()
    return set(pd.to_datetime(holidays_df['date']).dt.date)


def aggregate_weather_daily(city: str = 'Kathmandu') -> pd.DataFrame:
    """
    Aggregate hourly weather data to daily level.

    Args:
        city: City name to filter by

    Returns:
        DataFrame with daily weather aggregates
    """
    weather_df = load_weather_data()

    if city and 'city' in weather_df.columns:
        weather_df = weather_df[weather_df['city'] == city]

    # Group by date and aggregate
    daily = weather_df.groupby(weather_df['date'].dt.date).agg({
        'temperature_c': ['mean', 'max', 'min', 'std'],
        'humidity_percent': 'mean',
        'precipitation_mm': 'sum',
        'windspeed_kmh': 'mean'
    }).reset_index()

    # Flatten column names
    daily.columns = ['date', 'temp_mean', 'temp_max', 'temp_min', 'temp_std',
                     'humidity', 'precipitation', 'windspeed']
    daily['temp_range'] = daily['temp_max'] - daily['temp_min']
    daily['date'] = pd.to_datetime(daily['date'])

    return daily


def prepare_features_for_prediction(
    date: pd.Timestamp,
    temp_mean: float,
    temp_max: float,
    temp_min: float,
    humidity: float,
    precipitation: float,
    windspeed: float,
    demand_lag_1: float,
    demand_lag_7: float,
    demand_rolling_7: float,
    is_holiday: int = 0
) -> pd.DataFrame:
    """
    Prepare feature DataFrame for model prediction.

    Args:
        date: Prediction date
        temp_mean: Mean temperature
        temp_max: Max temperature
        temp_min: Min temperature
        humidity: Humidity percentage
        precipitation: Precipitation in mm
        windspeed: Wind speed in km/h
        demand_lag_1: Previous day's demand
        demand_lag_7: Same day last week demand
        demand_rolling_7: 7-day rolling average demand
        is_holiday: Whether it's a holiday

    Returns:
        DataFrame with features ready for prediction
    """
    temp_range = temp_max - temp_min
    day_of_week = date.dayofweek
    month = date.month
    day_of_year = date.dayofyear
    week_of_year = date.isocalendar().week
    quarter = date.quarter
    is_weekend = 1 if day_of_week in [5, 6] else 0

    # Season mapping
    season_map = {
        12: 0, 1: 0, 2: 0,  # Winter
        3: 1, 4: 1,         # Spring
        5: 2, 6: 2, 7: 2,   # Summer
        8: 3, 9: 3,         # Monsoon
        10: 4, 11: 4        # Autumn
    }
    season = season_map.get(month, 0)

    features = {
        'temp_mean': temp_mean,
        'temp_max': temp_max,
        'temp_min': temp_min,
        'humidity': humidity,
        'precipitation': precipitation,
        'windspeed': windspeed,
        'temp_range': temp_range,
        'day_of_week': day_of_week,
        'month': month,
        'day_of_year': day_of_year,
        'week_of_year': week_of_year,
        'quarter': quarter,
        'is_weekend': is_weekend,
        'is_holiday': is_holiday,
        'season': season,
        'demand_lag_1': demand_lag_1,
        'demand_lag_7': demand_lag_7,
        'demand_rolling_7': demand_rolling_7
    }

    return pd.DataFrame([features])


def get_summary_stats(df: pd.DataFrame) -> dict:
    """
    Calculate summary statistics from energy data.

    Args:
        df: Energy DataFrame

    Returns:
        Dictionary with summary statistics
    """
    energy_col = 'Energy Requirement'

    stats = {
        'total_energy_mwh': df[energy_col].sum(),
        'avg_daily_demand': df[energy_col].mean(),
        'peak_demand': df[energy_col].max(),
        'peak_date': df.loc[df[energy_col].idxmax(), 'date'],
        'min_demand': df[energy_col].min(),
        'std_demand': df[energy_col].std(),
        'total_import': df['Energy_generation_Import'].sum() if 'Energy_generation_Import' in df.columns else 0,
        'total_export': df['Energy Export'].sum() if 'Energy Export' in df.columns else 0,
        'total_generation': df['Energy_generation_Total Energy Available'].sum() if 'Energy_generation_Total Energy Available' in df.columns else 0,
        'record_count': len(df),
        'date_start': df['date'].min(),
        'date_end': df['date'].max()
    }

    # Calculate import dependency
    if stats['total_generation'] > 0:
        stats['import_dependency'] = (stats['total_import'] / stats['total_generation']) * 100
    else:
        stats['import_dependency'] = 0

    return stats


def clear_cache():
    """Clear all cached data."""
    load_energy_data.cache_clear()
    load_weather_data.cache_clear()
    load_holidays.cache_clear()


if __name__ == "__main__":
    # Test data loading
    print("Testing data loading...")

    energy_df = load_energy_data()
    print(f"Energy data: {len(energy_df)} records")
    print(f"Date range: {energy_df['date'].min()} to {energy_df['date'].max()}")

    weather_df = load_weather_data()
    print(f"\nWeather data: {len(weather_df)} records")

    holidays_df = load_holidays()
    print(f"Holidays: {len(holidays_df)} records")

    stats = get_summary_stats(energy_df)
    print(f"\nSummary Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")