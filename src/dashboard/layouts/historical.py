"""
Historical Data Layout
Page layout for historical energy data visualization.
"""

import dash_bootstrap_components as dbc
from dash import html, dcc
import pandas as pd
from datetime import datetime, timedelta

from ..components.cards import create_summary_cards
from ..utils.data_loader import load_energy_data, get_summary_stats


def create_historical_layout() -> dbc.Container:
    """
    Create the historical data visualization page layout.

    Returns:
        Bootstrap Container with the page layout
    """
    # Load data for initial stats
    try:
        df = load_energy_data()
        stats = get_summary_stats(df)
        min_date = stats['date_start']
        max_date = stats['date_end']
    except Exception:
        df = pd.DataFrame()
        min_date = datetime(2022, 7, 1)
        max_date = datetime(2023, 11, 30)

    layout = dbc.Container([
        # Page Header
        dbc.Row([
            dbc.Col([
                html.H2('[Historical] Energy Analysis', className='mb-1'),
                html.P(
                    'Explore historical energy demand patterns and generation trends in Nepal',
                    className='text-muted'
                )
            ], width=8),
            dbc.Col([
                html.P([
                    html.Small(f"Data: {min_date.strftime('%b %Y')} - {max_date.strftime('%b %Y')}")
                ], className='text-end text-muted mt-3')
            ], width=4)
        ], className='mb-4'),

        # Filters Row
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    # Date Range Picker
                    dbc.Col([
                        html.Label('Date Range', className='fw-bold'),
                        dcc.DatePickerRange(
                            id='date-range-picker',
                            min_date_allowed=min_date,
                            max_date_allowed=max_date,
                            start_date=min_date,
                            end_date=max_date,
                            display_format='MMM DD, YYYY',
                            className='mt-1'
                        )
                    ], width=4),

                    # City Selector (for weather correlation)
                    dbc.Col([
                        html.Label('City (Weather Data)', className='fw-bold'),
                        dcc.Dropdown(
                            id='city-selector',
                            options=[
                                {'label': 'Kathmandu', 'value': 'Kathmandu'},
                                {'label': 'Pokhara', 'value': 'Pokhara'}
                            ],
                            value='Kathmandu',
                            clearable=False,
                            className='mt-1'
                        )
                    ], width=3),

                    # Variable Selector
                    dbc.Col([
                        html.Label('Weather Variable', className='fw-bold'),
                        dcc.Dropdown(
                            id='weather-variable',
                            options=[
                                {'label': 'Mean Temperature', 'value': 'temp_mean'},
                                {'label': 'Max Temperature', 'value': 'temp_max'},
                                {'label': 'Precipitation', 'value': 'precipitation'},
                                {'label': 'Humidity', 'value': 'humidity'}
                            ],
                            value='temp_mean',
                            clearable=False,
                            className='mt-1'
                        )
                    ], width=3),

                    # Refresh Button
                    dbc.Col([
                        html.Label('Actions', className='fw-bold'),
                        dbc.Button(
                            '[Refresh]',
                            id='refresh-historical',
                            color='secondary',
                            className='mt-1'
                        )
                    ], width=2)
                ])
            ])
        ], className='mb-4 shadow-sm'),

        # KPI Summary Cards
        html.Div(id='summary-cards', className='mb-4'),

        # Main Trend Chart
        dbc.Card([
            dbc.CardHeader(html.H5('Energy Demand Trend', className='mb-0')),
            dbc.CardBody([
                dcc.Loading(
                    dcc.Graph(id='demand-trend-chart', config={'displayModeBar': False}),
                    type='default'
                )
            ])
        ], className='mb-4 shadow-sm'),

        # Charts Row 1: Generation Mix + Seasonal Patterns
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5('Generation Mix', className='mb-0')),
                    dbc.CardBody([
                        dcc.Loading(
                            dcc.Graph(id='generation-mix-chart', config={'displayModeBar': False}),
                            type='default'
                        )
                    ])
                ], className='shadow-sm h-100')
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5('Seasonal Patterns', className='mb-0')),
                    dbc.CardBody([
                        dcc.Loading(
                            dcc.Graph(id='seasonal-chart', config={'displayModeBar': False}),
                            type='default'
                        )
                    ])
                ], className='shadow-sm h-100')
            ], width=6)
        ], className='mb-4'),

        # Charts Row 2: Weekly Patterns + Weather Correlation
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5('Weekly Patterns', className='mb-0')),
                    dbc.CardBody([
                        dcc.Loading(
                            dcc.Graph(id='weekly-chart', config={'displayModeBar': False}),
                            type='default'
                        )
                    ])
                ], className='shadow-sm h-100')
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5('Weather Correlation', className='mb-0')),
                    dbc.CardBody([
                        dcc.Loading(
                            dcc.Graph(id='weather-correlation-chart', config={'displayModeBar': False}),
                            type='default'
                        )
                    ])
                ], className='shadow-sm h-100')
            ], width=6)
        ], className='mb-4'),

        # Data Table
        dbc.Card([
            dbc.CardHeader(html.H5('[Data] Table', className='mb-0')),
            dbc.CardBody([
                dcc.Loading(
                    html.Div(id='data-table'),
                    type='default'
                )
            ])
        ], className='shadow-sm'),

        # Interval for auto-refresh (optional)
        dcc.Interval(
            id='historical-interval',
            interval=60000 * 5,  # 5 minutes
            n_intervals=0,
            disabled=True  # Disabled by default
        )

    ], fluid=True, className='px-4')

    return layout


def get_page_title() -> str:
    """Return page title for navigation."""
    return 'Historical Analysis'


def get_page_id() -> str:
    """Return unique page identifier."""
    return 'historical-page'