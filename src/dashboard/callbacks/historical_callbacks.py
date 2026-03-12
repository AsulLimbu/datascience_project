"""
Historical Page Callbacks
Interactive callbacks for the historical data visualization page.
"""

import pandas as pd
import numpy as np
from dash import Output, Input, State, callback, dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from ..utils.data_loader import load_energy_data, load_weather_data, get_summary_stats, aggregate_weather_daily
from ..components.charts import (
    create_demand_trend_chart,
    create_generation_mix_chart,
    create_seasonal_pattern_chart,
    create_weekly_pattern_chart,
    create_weather_correlation_chart,
    create_generation_pie_chart
)
from ..components.cards import create_summary_cards


def register_callbacks(app):
    """Register all historical page callbacks."""

    @app.callback(
        Output('summary-cards', 'children'),
        Input('date-range-picker', 'start_date'),
        Input('date-range-picker', 'end_date')
    )
    def update_summary_cards(start_date, end_date):
        """Update KPI summary cards based on date range."""
        df = load_energy_data()

        # Filter by date range
        if start_date and end_date:
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

        cards = create_summary_cards(df)

        return dbc.Row([
            dbc.Col(card, width=3) for card in cards
        ], className='g-3')

    @app.callback(
        Output('demand-trend-chart', 'figure'),
        Input('date-range-picker', 'start_date'),
        Input('date-range-picker', 'end_date')
    )
    def update_demand_trend(start_date, end_date):
        """Update demand trend chart."""
        df = load_energy_data()

        date_range = (start_date, end_date) if start_date and end_date else None

        fig = create_demand_trend_chart(df, date_range)

        return fig

    @app.callback(
        Output('generation-mix-chart', 'figure'),
        Input('date-range-picker', 'start_date'),
        Input('date-range-picker', 'end_date')
    )
    def update_generation_mix(start_date, end_date):
        """Update generation mix chart."""
        df = load_energy_data()

        date_range = (start_date, end_date) if start_date and end_date else None

        fig = create_generation_mix_chart(df, date_range)

        return fig

    @app.callback(
        Output('seasonal-chart', 'figure'),
        Input('date-range-picker', 'start_date'),
        Input('date-range-picker', 'end_date')
    )
    def update_seasonal_pattern(start_date, end_date):
        """Update seasonal pattern chart."""
        df = load_energy_data()

        if start_date and end_date:
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

        fig = create_seasonal_pattern_chart(df)

        return fig

    @app.callback(
        Output('weekly-chart', 'figure'),
        Input('date-range-picker', 'start_date'),
        Input('date-range-picker', 'end_date')
    )
    def update_weekly_pattern(start_date, end_date):
        """Update weekly pattern chart."""
        df = load_energy_data()

        if start_date and end_date:
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

        fig = create_weekly_pattern_chart(df)

        return fig

    @app.callback(
        Output('weather-correlation-chart', 'figure'),
        Input('date-range-picker', 'start_date'),
        Input('date-range-picker', 'end_date'),
        Input('city-selector', 'value'),
        Input('weather-variable', 'value')
    )
    def update_weather_correlation(start_date, end_date, city, weather_var):
        """Update weather correlation chart."""
        energy_df = load_energy_data()

        # Get weather data
        try:
            weather_df = aggregate_weather_daily(city)

            # Merge energy and weather data
            merged = energy_df.merge(
                weather_df[['date', weather_var]],
                on='date',
                how='inner'
            )

            # Add season info
            if 'season' not in merged.columns:
                season_map = {
                    12: 'Winter', 1: 'Winter', 2: 'Winter',
                    3: 'Spring', 4: 'Spring',
                    5: 'Summer', 6: 'Summer', 7: 'Summer',
                    8: 'Monsoon', 9: 'Monsoon',
                    10: 'Autumn', 11: 'Autumn'
                }
                merged['season'] = merged['month'].map(season_map)

            if start_date and end_date:
                merged = merged[(merged['date'] >= start_date) & (merged['date'] <= end_date)]

            fig = create_weather_correlation_chart(merged, weather_var)
        except Exception as e:
            # Return empty figure if weather data unavailable
            fig = go.Figure()
            fig.update_layout(
                title=f'Weather data unavailable: {str(e)}',
                height=350
            )

        return fig

    @app.callback(
        Output('data-table', 'children'),
        Input('date-range-picker', 'start_date'),
        Input('date-range-picker', 'end_date')
    )
    def update_data_table(start_date, end_date):
        """Update data table."""
        df = load_energy_data()

        if start_date and end_date:
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

        # Select relevant columns
        display_cols = [
            'date', 'Energy Requirement',
            'Energy_generation_NEA', 'Energy_generation_IPP',
            'Energy_generation_Import', 'Energy Export'
        ]

        display_cols = [c for c in display_cols if c in df.columns]

        # Format for display
        display_df = df[display_cols].copy()
        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')

        # Round numeric columns
        for col in display_df.select_dtypes(include=[np.number]).columns:
            display_df[col] = display_df[col].round(0).astype(int)

        return dbc.Table.from_dataframe(
            display_df.head(20),
            striped=True,
            bordered=True,
            hover=True,
            size='sm',
            className='mt-2'
        )