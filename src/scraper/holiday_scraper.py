"""
Holiday Scraper for Nepal
Fetches Nepali public holidays which affect energy consumption patterns.
"""

import requests
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from pathlib import Path
import re


class HolidayScraper:
    """
    Scrapes Nepali public holidays from various sources.
    Holidays significantly impact energy consumption patterns.
    """

    # Known Nepali holidays with approximate dates (for fallback)
    NEPALI_HOLIDAYS = {
        # Major festivals that shift dates annually (Lunar calendar)
        'Dashain': {'type': 'major', 'impact': 'high'},
        'Tihar': {'type': 'major', 'impact': 'high'},
        'Chhath': {'type': 'major', 'impact': 'medium'},
        # Fixed date holidays
        'New Year': {'date': '01-01', 'type': 'minor', 'impact': 'low'},
        'Democracy Day': {'date': '02-19', 'type': 'national', 'impact': 'medium'},
        'Holi': {'type': 'major', 'impact': 'medium'},
        'Buddha Jayanti': {'type': 'major', 'impact': 'low'},
        'Independence Day': {'date': '08-15', 'type': 'minor', 'impact': 'low'},
        'Constitution Day': {'date': '09-20', 'type': 'national', 'impact': 'medium'},
        'Vijaya Dashami': {'type': 'major', 'impact': 'high'},
        'Indra Jatra': {'type': 'major', 'impact': 'low'},
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def scrape_from_timeanddate(self, year: int = 2023) -> pd.DataFrame:
        """
        Scrape Nepal holidays from timeanddate.com

        Args:
            year: Year to fetch holidays for

        Returns:
            DataFrame with holiday information
        """
        url = f"https://www.timeanddate.com/holidays/nepal/{year}"

        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching holidays: {e}")
            return self._get_fallback_holidays(year)

        soup = BeautifulSoup(response.content, 'html.parser')

        holidays = []
        table = soup.find('table', {'class': 'holidays'})
        if table:
            for row in table.find_all('tr')[1:]:  # Skip header
                cols = row.find_all('th') + row.find_all('td')
                if len(cols) >= 2:
                    date_str = cols[0].get_text(strip=True)
                    name = cols[1].get_text(strip=True)
                    holiday_type = cols[2].get_text(strip=True) if len(cols) > 2 else ''

                    # Parse date
                    try:
                        date = datetime.strptime(f"{date_str} {year}", "%b %d %Y")
                    except ValueError:
                        continue

                    holidays.append({
                        'date': date.date(),
                        'holiday_name': name,
                        'type': holiday_type,
                        'year': year
                    })

        if holidays:
            return pd.DataFrame(holidays)
        else:
            return self._get_fallback_holidays(year)

    def _get_fallback_holidays(self, year: int) -> pd.DataFrame:
        """
        Fallback holiday data when scraping fails.
        Contains major Nepali holidays with fixed or known dates.
        """
        # Nepali holidays with known dates for 2022-2024
        holidays_data = []

        # Fixed date holidays
        fixed_holidays = [
            (f'{year}-01-01', 'New Year', 'National'),
            (f'{year}-01-15', 'Maghe Sankranti', 'Festival'),
            (f'{year}-02-19', 'Democracy Day', 'National'),
            (f'{year}-03-08', 'Maha Shivaratri', 'Festival'),
            (f'{year}-03-14', 'Holi', 'Festival'),
            (f'{year}-04-14', 'Nepali New Year', 'National'),
            (f'{year}-05-01', 'Labor Day', 'International'),
            (f'{year}-05-16', 'Buddha Jayanti', 'Festival'),
            (f'{year}-08-15', 'Independence Day', 'International'),
            (f'{year}-09-20', 'Constitution Day', 'National'),
        ]

        for date_str, name, h_type in fixed_holidays:
            try:
                date = datetime.strptime(date_str, '%Y-%m-%d').date()
                holidays_data.append({
                    'date': date,
                    'holiday_name': name,
                    'type': h_type,
                    'year': year
                })
            except ValueError:
                continue

        return pd.DataFrame(holidays_data)

    def get_holiday_calendar(
        self,
        start_year: int = 2022,
        end_year: int = 2024,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get holiday calendar for multiple years.

        Args:
            start_year: Start year
            end_year: End year
            save_path: Optional path to save the calendar

        Returns:
            DataFrame with holidays for all years
        """
        all_holidays = []

        for year in range(start_year, end_year + 1):
            print(f"Fetching holidays for {year}...")
            df = self.scrape_from_timeanddate(year)
            all_holidays.append(df)

        combined = pd.concat(all_holidays, ignore_index=True)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(save_path, index=False)
            print(f"Holiday calendar saved to {save_path}")

        return combined

    def add_holiday_features(
        self,
        energy_df: pd.DataFrame,
        date_column: str = 'date',
        holiday_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Add holiday-related features to energy dataset.

        Args:
            energy_df: DataFrame with energy data
            date_column: Name of the date column
            holiday_df: DataFrame with holiday data (optional)

        Returns:
            Energy DataFrame with added holiday features
        """
        df = energy_df.copy()

        # Ensure date column is datetime
        df[date_column] = pd.to_datetime(df[date_column])
        df['is_weekend'] = df[date_column].dt.dayofweek.isin([5, 6]).astype(int)

        if holiday_df is not None:
            holiday_dates = set(pd.to_datetime(holiday_df['date']).dt.date)
            df['is_holiday'] = df[date_column].dt.date.isin(holiday_dates).astype(int)
        else:
            df['is_holiday'] = 0

        # Day of week one-hot encoding
        df['day_of_week'] = df[date_column].dt.dayofweek
        df['month'] = df[date_column].dt.month
        df['quarter'] = df[date_column].dt.quarter

        # Season (Nepal has 4 main seasons)
        df['season'] = df['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring',
            5: 'Summer', 6: 'Summer', 7: 'Summer',
            8: 'Monsoon', 9: 'Monsoon',
            10: 'Autumn', 11: 'Autumn'
        })

        return df


def fetch_nepal_holidays(
    start_year: int = 2022,
    end_year: int = 2024,
    save_path: str = 'data/raw/holidays.csv'
) -> pd.DataFrame:
    """
    Convenience function to fetch Nepal holidays.

    Args:
        start_year: Start year
        end_year: End year
        save_path: Where to save the holiday calendar

    Returns:
        DataFrame with holiday data
    """
    scraper = HolidayScraper()
    holidays = scraper.get_holiday_calendar(
        start_year=start_year,
        end_year=end_year,
        save_path=save_path
    )
    return holidays


if __name__ == "__main__":
    print("=" * 60)
    print("Nepal Holiday Scraper for Energy Analysis")
    print("=" * 60)

    # Fetch holidays for the period covered by NEA data
    holidays_df = fetch_nepal_holidays(
        start_year=2022,
        end_year=2024,
        save_path='data/raw/holidays.csv'
    )

    print("\nFetched holidays:")
    print(holidays_df)

    # Show how to merge with energy data
    print("\nExample: Adding holiday features to energy data")
    print("Use: HolidayScraper().add_holiday_features(energy_df, date_column='Date(English)')")