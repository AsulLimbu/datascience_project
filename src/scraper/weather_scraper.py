"""
Weather Data Scraper
Fetches historical and current weather data for Nepal using OpenWeatherMap API.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import time
import json
from pathlib import Path


class WeatherScraper:
    """
    Scraper for weather data from OpenWeatherMap API.
    Focuses on Kathmandu, Nepal for energy correlation analysis.
    """

    BASE_URL = "https://api.openweathermap.org/data/2.5"
    GEO_URL = "http://api.openweathermap.org/geo/1.0"

    # Major cities in Nepal for energy analysis
    NEPAL_CITIES = {
        'Kathmandu': {'lat': 27.7172, 'lon': 85.3240},
        'Pokhara': {'lat': 28.2667, 'lon': 83.9667},
        'Biratnagar': {'lat': 26.4833, 'lon': 87.2833},
        'Birgunj': {'lat': 27.0100, 'lon': 84.8800},
        'Butwal': {'lat': 27.7000, 'lon': 83.4500},
    }

    def __init__(self, api_key: str):
        """
        Initialize the weather scraper.

        Args:
            api_key: OpenWeatherMap API key (free at openweathermap.org)
        """
        self.api_key = api_key
        self.session = requests.Session()

    def get_current_weather(self, city: str = 'Kathmandu') -> Dict:
        """
        Get current weather for a Nepali city.

        Args:
            city: Name of the city in Nepal

        Returns:
            Dictionary with current weather data
        """
        if city not in self.NEPAL_CITIES:
            raise ValueError(f"City {city} not in supported cities: {list(self.NEPAL_CITIES.keys())}")

        coords = self.NEPAL_CITIES[city]
        url = f"{self.BASE_URL}/weather"
        params = {
            'lat': coords['lat'],
            'lon': coords['lon'],
            'appid': self.api_key,
            'units': 'metric'
        }

        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_historical_weather(
        self,
        city: str = 'Kathmandu',
        start_date: str = None,
        end_date: str = None,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get historical weather data using Open-Meteo API (free, no API key needed).

        Args:
            city: Name of the city in Nepal
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            save_path: Optional path to save the data as CSV

        Returns:
            DataFrame with historical weather data
        """
        if city not in self.NEPAL_CITIES:
            raise ValueError(f"City {city} not in supported cities")

        coords = self.NEPAL_CITIES[city]

        # Use Open-Meteo API (free historical data)
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            'latitude': coords['lat'],
            'longitude': coords['lon'],
            'start_date': start_date,
            'end_date': end_date,
            'hourly': 'temperature_2m,relative_humidity_2m,precipitation,windspeed_10m',
            'timezone': 'Asia/Kathmandu'
        }

        print(f"Fetching weather data for {city} from {start_date} to {end_date}...")

        response = self.session.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # Parse into DataFrame
        hourly = data.get('hourly', {})
        df = pd.DataFrame({
            'datetime': pd.to_datetime(hourly.get('time', [])),
            'temperature_c': hourly.get('temperature_2m', []),
            'humidity_percent': hourly.get('relative_humidity_2m', []),
            'precipitation_mm': hourly.get('precipitation', []),
            'windspeed_kmh': hourly.get('windspeed_10m', [])
        })

        df['city'] = city
        df['date'] = df['datetime'].dt.date

        if save_path:
            df.to_csv(save_path, index=False)
            print(f"Weather data saved to {save_path}")

        return df

    def get_weather_forecast(self, city: str = 'Kathmandu', days: int = 7) -> pd.DataFrame:
        """
        Get weather forecast for a city (free, no API key needed via Open-Meteo).

        Args:
            city: Name of the city in Nepal
            days: Number of forecast days (max 16)

        Returns:
            DataFrame with forecast data
        """
        if city not in self.NEPAL_CITIES:
            raise ValueError(f"City {city} not in supported cities")

        coords = self.NEPAL_CITIES[city]

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': coords['lat'],
            'longitude': coords['lon'],
            'hourly': 'temperature_2m,relative_humidity_2m,precipitation,windspeed_10m',
            'forecast_days': days,
            'timezone': 'Asia/Kathmandu'
        }

        print(f"Fetching {days}-day forecast for {city}...")

        response = self.session.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        hourly = data.get('hourly', {})
        df = pd.DataFrame({
            'datetime': pd.to_datetime(hourly.get('time', [])),
            'temperature_c': hourly.get('temperature_2m', []),
            'humidity_percent': hourly.get('relative_humidity_2m', []),
            'precipitation_mm': hourly.get('precipitation', []),
            'windspeed_kmh': hourly.get('windspeed_10m', [])
        })

        df['city'] = city
        return df

    def aggregate_daily_weather(self, hourly_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate hourly weather data to daily level for energy correlation.

        Args:
            hourly_df: DataFrame with hourly weather data

        Returns:
            DataFrame with daily aggregated weather metrics
        """
        daily = hourly_df.groupby('date').agg({
            'temperature_c': ['mean', 'max', 'min', 'std'],
            'humidity_percent': ['mean', 'max', 'min'],
            'precipitation_mm': ['sum', 'max'],
            'windspeed_kmh': ['mean', 'max']
        }).reset_index()

        # Flatten column names
        daily.columns = ['_'.join(col).strip('_') for col in daily.columns.values]
        daily = daily.rename(columns={'date_': 'date'})

        return daily


def fetch_nepal_weather_for_energy_analysis(
    start_date: str = '2022-07-01',
    end_date: str = '2023-11-30',
    cities: List[str] = ['Kathmandu'],
    save_path: str = 'data/raw/weather_data.csv'
) -> pd.DataFrame:
    """
    Convenience function to fetch weather data matching your energy dataset period.

    Your NEA data covers: 2079/04 - 2080/11 (July 2022 - Nov 2023 approximately)

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        cities: List of Nepali cities to fetch data for
        save_path: Where to save the combined data

    Returns:
        Combined DataFrame with weather data for all cities
    """
    scraper = WeatherScraper(api_key="")  # Open-Meteo doesn't need API key

    all_data = []
    for city in cities:
        df = scraper.get_historical_weather(
            city=city,
            start_date=start_date,
            end_date=end_date,
            save_path=None
        )
        all_data.append(df)
        time.sleep(1)  # Be respectful to the API

    combined = pd.concat(all_data, ignore_index=True)

    # Save the data
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(save_path, index=False)
    print(f"\nTotal records: {len(combined)}")
    print(f"Data saved to: {save_path}")

    return combined


if __name__ == "__main__":
    # Example usage - fetches weather data for your energy analysis period
    print("=" * 60)
    print("Weather Data Scraper for Energy Analysis")
    print("=" * 60)

    # Fetch data for the period covered by your NEA reports
    weather_df = fetch_nepal_weather_for_energy_analysis(
        start_date='2022-07-01',
        end_date='2023-11-30',
        cities=['Kathmandu', 'Pokhara'],
        save_path='data/raw/weather_data.csv'
    )

    print("\nSample data:")
    print(weather_df.head(10))

    print("\nDaily aggregation:")
    scraper = WeatherScraper("")
    daily_weather = scraper.aggregate_daily_weather(weather_df)
    print(daily_weather.head())