"""
Demand Forecasting Layout
Page layout for energy demand prediction interface.
"""

import dash_bootstrap_components as dbc
from dash import html, dcc
from datetime import datetime, timedelta


def create_forecasting_layout() -> dbc.Container:
    """
    Create the demand forecasting page layout.

    Returns:
        Bootstrap Container with the page layout
    """
    today = datetime.now()
    tomorrow = today + timedelta(days=1)
    max_forecast_date = today + timedelta(days=14)

    layout = dbc.Container([
        # Page Header
        dbc.Row([
            dbc.Col([
                html.H2('[Forecast] Energy Demand Forecasting', className='mb-1'),
                html.P(
                    'Predict future energy demand using machine learning and real-time weather data',
                    className='text-muted'
                )
            ])
        ], className='mb-4'),

        # Main Content Row
        dbc.Row([
            # Left Column: Input Form
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5('[Input] Parameters', className='mb-0')),
                    dbc.CardBody([
                        # Date Selection
                        html.Div([
                            html.Label('Prediction Date', className='fw-bold'),
                            dcc.DatePickerSingle(
                                id='prediction-date',
                                min_date_allowed=tomorrow,
                                max_date_allowed=max_forecast_date,
                                initial_visible_month=tomorrow,
                                date=tomorrow.date(),
                                display_format='MMM DD, YYYY',
                                className='mt-1 w-100'
                            )
                        ], className='mb-3'),

                        # City Selection
                        html.Div([
                            html.Label('City', className='fw-bold'),
                            dcc.Dropdown(
                                id='forecast-city',
                                options=[
                                    {'label': 'Kathmandu', 'value': 'Kathmandu'},
                                    {'label': 'Pokhara', 'value': 'Pokhara'}
                                ],
                                value='Kathmandu',
                                clearable=False,
                                className='mt-1'
                            )
                        ], className='mb-3'),

                        # Fetch Weather Button
                        dbc.Button(
                            '[Fetch Weather]',
                            id='fetch-weather-btn',
                            color='info',
                            className='w-100 mb-3',
                            outline=True
                        ),

                        html.Hr(),

                        # Weather Inputs
                        html.H6('Weather Parameters', className='mt-3 mb-2'),

                        dbc.Row([
                            dbc.Col([
                                html.Label('Temp Mean (°C)', className='small'),
                                dcc.Input(
                                    id='input-temp-mean',
                                    type='number',
                                    value=25.0,
                                    className='form-control form-control-sm'
                                )
                            ], width=6),
                            dbc.Col([
                                html.Label('Temp Max (°C)', className='small'),
                                dcc.Input(
                                    id='input-temp-max',
                                    type='number',
                                    value=30.0,
                                    className='form-control form-control-sm'
                                )
                            ], width=6)
                        ], className='mb-2'),

                        dbc.Row([
                            dbc.Col([
                                html.Label('Temp Min (°C)', className='small'),
                                dcc.Input(
                                    id='input-temp-min',
                                    type='number',
                                    value=20.0,
                                    className='form-control form-control-sm'
                                )
                            ], width=6),
                            dbc.Col([
                                html.Label('Humidity (%)', className='small'),
                                dcc.Input(
                                    id='input-humidity',
                                    type='number',
                                    value=75.0,
                                    className='form-control form-control-sm'
                                )
                            ], width=6)
                        ], className='mb-2'),

                        dbc.Row([
                            dbc.Col([
                                html.Label('Precipitation (mm)', className='small'),
                                dcc.Input(
                                    id='input-precipitation',
                                    type='number',
                                    value=0.0,
                                    className='form-control form-control-sm'
                                )
                            ], width=6),
                            dbc.Col([
                                html.Label('Wind Speed (km/h)', className='small'),
                                dcc.Input(
                                    id='input-windspeed',
                                    type='number',
                                    value=5.0,
                                    className='form-control form-control-sm'
                                )
                            ], width=6)
                        ], className='mb-3'),

                        html.Hr(),

                        # Historical Demand Inputs
                        html.H6('Historical Demand (MWh)', className='mt-3 mb-2'),

                        dbc.Row([
                            dbc.Col([
                                html.Label('Previous Day', className='small'),
                                dcc.Input(
                                    id='input-lag-1',
                                    type='number',
                                    value=32000,
                                    className='form-control form-control-sm'
                                )
                            ], width=4),
                            dbc.Col([
                                html.Label('Same Day Last Week', className='small'),
                                dcc.Input(
                                    id='input-lag-7',
                                    type='number',
                                    value=31500,
                                    className='form-control form-control-sm'
                                )
                            ], width=4),
                            dbc.Col([
                                html.Label('7-Day Average', className='small'),
                                dcc.Input(
                                    id='input-rolling-7',
                                    type='number',
                                    value=31800,
                                    className='form-control form-control-sm'
                                )
                            ], width=4)
                        ], className='mb-3'),

                        # Holiday Checkbox
                        dbc.Checkbox(
                            id='input-is-holiday',
                            label='Is Holiday',
                            value=False,
                            className='mb-3'
                        ),

                        # Predict Button
                        dbc.Button(
                            '[PREDICT DEMAND]',
                            id='predict-btn',
                            color='primary',
                            size='lg',
                            className='w-100 mt-2'
                        )
                    ])
                ], className='shadow-sm')
            ], width=5),

            # Right Column: Prediction Results
            dbc.Col([
                # Prediction Result Card
                dbc.Card([
                    dbc.CardHeader(html.H5('[Prediction] Result', className='mb-0')),
                    dbc.CardBody([
                        dcc.Loading(
                            html.Div(id='prediction-result'),
                            type='default'
                        )
                    ])
                ], className='shadow-sm mb-4'),

                # Feature Importance Card
                dbc.Card([
                    dbc.CardHeader(html.H5('[Features] Importance', className='mb-0')),
                    dbc.CardBody([
                        dcc.Loading(
                            dcc.Graph(
                                id='feature-importance-chart',
                                config={'displayModeBar': False},
                                style={'height': '300px'}
                            ),
                            type='default'
                        )
                    ])
                ], className='shadow-sm mb-4'),

                # Recent Trend Card
                dbc.Card([
                    dbc.CardHeader(html.H5('[Trend] Recent Demand', className='mb-0')),
                    dbc.CardBody([
                        dcc.Loading(
                            dcc.Graph(
                                id='recent-trend-chart',
                                config={'displayModeBar': False},
                                style={'height': '250px'}
                            ),
                            type='default'
                        )
                    ])
                ], className='shadow-sm')
            ], width=7)
        ]),

        # Hidden divs for storing data
        dcc.Store(id='weather-store'),
        dcc.Store(id='prediction-store'),

        # Alert for errors
        html.Div(id='forecast-alert')

    ], fluid=True, className='px-4')

    return layout


def get_page_title() -> str:
    """Return page title for navigation."""
    return 'Demand Forecasting'


def get_page_id() -> str:
    """Return unique page identifier."""
    return 'forecasting-page'