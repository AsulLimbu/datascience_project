"""
Weather API Utility
Fetches real-time weather data and forecasts from Open-Meteo API.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time


class WeatherAPI:
    """
    Fetches weather data from Open-Meteo API.
    Free API - no key required.
    """

    # Nepal cities with coordinates
    NEPAL_CITIES = {
        'Kathmandu': {'lat': 27.7172, 'lon': 85.3240},
        'Pokhara': {'lat': 28.2667, 'lon': 83.9667},
        'Biratnagar': {'lat': 26.4833, 'lon': 87.2833},
        'Birgunj': {'lat': 27.0100, 'lon': 84.8800},
        'Butwal': {'lat': 27.7000, 'lon': 83.4500},
    }

    ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"
    FORECAST_API = "https://api.open-meteo.com/v1/forecast"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'NEA-Energy-Dashboard/1.0'
        })

    def get_forecast(
        self,
        city: str = 'Kathmandu',
        days: int = 7
    ) -> pd.DataFrame:
        """
        Get weather forecast for the next N days.

        Args:
            city: City name in Nepal
            days: Number of forecast days (max 16)

        Returns:
            DataFrame with daily weather forecast
        """
        if city not in self.NEPAL_CITIES:
            raise ValueError(f"City '{city}' not supported. "
                           f"Choose from: {list(self.NEPAL_CITIES.keys())}")

        coords = self.NEPAL_CITIES[city]

        params = {
            'latitude': coords['lat'],
            'longitude': coords['lon'],
            'daily': 'temperature_2m_max,temperature_2m_min,'
                     'temperature_2m_mean,precipitation_sum,'
                     'windspeed_10m_max,relative_humidity_2m_mean',
            'timezone': 'Asia/Kathmandu',
            'forecast_days': min(days, 16)
        }

        try:
            response = self.session.get(self.FORECAST_API, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to fetch weather forecast: {e}")

        # Parse response
        daily = data.get('daily', {})

        df = pd.DataFrame({
            'date': pd.to_datetime(daily.get('time', [])),
            'temp_max': daily.get('temperature_2m_max', []),
            'temp_min': daily.get('temperature_2m_min', []),
            'temp_mean': daily.get('temperature_2m_mean', []),
            'precipitation': daily.get('precipitation_sum', []),
            'windspeed': daily.get('windspeed_10m_max', []),
            'humidity': daily.get('relative_humidity_2m_mean', [])
        })

        df['temp_range'] = df['temp_max'] - df['temp_min']
        df['city'] = city

        return df

    def get_current_weather(self, city: str = 'Kathmandu') -> Dict:
        """
        Get current weather conditions for a city.

        Args:
            city: City name in Nepal

        Returns:
            Dictionary with current weather data
        """
        if city not in self.NEPAL_CITIES:
            raise ValueError(f"City '{city}' not supported")

        coords = self.NEPAL_CITIES[city]

        params = {
            'latitude': coords['lat'],
            'longitude': coords['lon'],
            'current_weather': 'true',
            'timezone': 'Asia/Kathmandu'
        }

        try:
            response = self.session.get(self.FORECAST_API, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to fetch current weather: {e}")

        return data.get('current_weather', {})

    def get_historical(
        self,
        city: str = 'Kathmandu',
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Get historical weather data.

        Args:
            city: City name in Nepal
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with historical weather data
        """
        if city not in self.NEPAL_CITIES:
            raise ValueError(f"City '{city}' not supported")

        coords = self.NEPAL_CITIES[city]

        params = {
            'latitude': coords['lat'],
            'longitude': coords['lon'],
            'daily': 'temperature_2m_max,temperature_2m_min,'
                     'temperature_2m_mean,precipitation_sum,'
                     'windspeed_10m_max,relative_humidity_2m_mean',
            'timezone': 'Asia/Kathmandu',
            'start_date': start_date,
            'end_date': end_date
        }

        try:
            response = self.session.get(self.ARCHIVE_API, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to fetch historical weather: {e}")

        daily = data.get('daily', {})

        df = pd.DataFrame({
            'date': pd.to_datetime(daily.get('time', [])),
            'temp_max': daily.get('temperature_2m_max', []),
            'temp_min': daily.get('temperature_2m_min', []),
            'temp_mean': daily.get('temperature_2m_mean', []),
            'precipitation': daily.get('precipitation_sum', []),
            'windspeed': daily.get('windspeed_10m_max', []),
            'humidity': daily.get('relative_humidity_2m_mean', [])
        })

        df['temp_range'] = df['temp_max'] - df['temp_min']
        df['city'] = city

        return df


def fetch_forecast_for_date(
    target_date: datetime,
    city: str = 'Kathmandu'
) -> Dict:
    """
    Fetch weather forecast for a specific future date.

    Args:
        target_date: Target date for prediction
        city: City name

    Returns:
        Dictionary with weather data for the date
    """
    api = WeatherAPI()

    # Calculate days ahead
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    days_ahead = (target_date - today).days

    if days_ahead < 0:
        # Date is in the past - fetch historical
        end_date = target_date.strftime('%Y-%m-%d')
        start_date = end_date
        df = api.get_historical(city, start_date, end_date)
    elif days_ahead > 16:
        raise ValueError("Cannot forecast more than 16 days ahead")
    else:
        # Fetch forecast
        df = api.get_forecast(city, days=days_ahead + 1)

    # Find the specific date
    target_str = target_date.strftime('%Y-%m-%d')
    match = df[df['date'].dt.strftime('%Y-%m-%d') == target_str]

    if len(match) == 0:
        raise ValueError(f"No weather data available for {target_str}")

    row = match.iloc[0]

    return {
        'temp_mean': float(row['temp_mean']) if pd.notna(row['temp_mean']) else 25.0,
        'temp_max': float(row['temp_max']) if pd.notna(row['temp_max']) else 30.0,
        'temp_min': float(row['temp_min']) if pd.notna(row['temp_min']) else 20.0,
        'humidity': float(row['humidity']) if pd.notna(row['humidity']) else 75.0,
        'precipitation': float(row['precipitation']) if pd.notna(row['precipitation']) else 0.0,
        'windspeed': float(row['windspeed']) if pd.notna(row['windspeed']) else 5.0,
        'temp_range': float(row['temp_range']) if pd.notna(row['temp_range']) else 10.0
    }


def get_available_cities() -> List[str]:
    """Return list of supported cities."""
    return list(WeatherAPI.NEPAL_CITIES.keys())


if __name__ == "__main__":
    # Test weather API
    print("Testing Weather API...")

    api = WeatherAPI()

    # Test forecast
    print("\nFetching 7-day forecast for Kathmandu...")
    forecast_df = api.get_forecast('Kathmandu', days=7)
    print(forecast_df)

    # Test specific date
    print("\nFetching weather for tomorrow...")
    tomorrow = datetime.now() + timedelta(days=1)
    weather = fetch_forecast_for_date(tomorrow, 'Kathmandu')
    print(weather)