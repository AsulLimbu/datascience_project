"""
Forecasting Page Callbacks
Interactive callbacks for the demand forecasting page.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dash import Output, Input, State, callback, dcc, html, ctx
import dash_bootstrap_components as dbc

from ..utils.data_loader import load_energy_data, prepare_features_for_prediction
from ..utils.predictor import predict_demand, get_feature_importance, validate_features
from ..utils.weather_api import fetch_forecast_for_date, WeatherAPI
from ..components.charts import create_feature_importance_chart, create_demand_trend_chart
from ..components.cards import create_prediction_card


def register_callbacks(app):
    """Register all forecasting page callbacks."""

    @app.callback(
        Output('weather-store', 'data'),
        Output('input-temp-mean', 'value'),
        Output('input-temp-max', 'value'),
        Output('input-temp-min', 'value'),
        Output('input-humidity', 'value'),
        Output('input-precipitation', 'value'),
        Output('input-windspeed', 'value'),
        Output('forecast-alert', 'children'),
        Input('fetch-weather-btn', 'n_clicks'),
        State('prediction-date', 'date'),
        State('forecast-city', 'value')
    )
    def fetch_weather(n_clicks, prediction_date, city):
        """Fetch weather forecast for selected date."""
        if not n_clicks:
            # Return default values
            return None, 25.0, 30.0, 20.0, 75.0, 0.0, 5.0, None

        if not prediction_date:
            return None, 25.0, 30.0, 20.0, 75.0, 0.0, 5.0, dbc.Alert(
                'Please select a date first', color='warning'
            )

        try:
            # Parse date
            if isinstance(prediction_date, str):
                target_date = datetime.strptime(prediction_date, '%Y-%m-%d')
            else:
                target_date = prediction_date

            # Fetch weather
            weather_data = fetch_forecast_for_date(target_date, city)

            alert = dbc.Alert(
                f'[OK] Weather data fetched for {target_date.strftime("%b %d, %Y")}',
                color='success',
                dismissable=True
            )

            return (
                weather_data,
                weather_data['temp_mean'],
                weather_data['temp_max'],
                weather_data['temp_min'],
                weather_data['humidity'],
                weather_data['precipitation'],
                weather_data['windspeed'],
                alert
            )

        except Exception as e:
            alert = dbc.Alert(
                f'[Error] Error fetching weather: {str(e)}',
                color='danger',
                dismissable=True
            )
            return None, 25.0, 30.0, 20.0, 75.0, 0.0, 5.0, alert

    @app.callback(
        Output('prediction-store', 'data'),
        Output('prediction-result', 'children'),
        Output('feature-importance-chart', 'figure'),
        Input('predict-btn', 'n_clicks'),
        State('prediction-date', 'date'),
        State('input-temp-mean', 'value'),
        State('input-temp-max', 'value'),
        State('input-temp-min', 'value'),
        State('input-humidity', 'value'),
        State('input-precipitation', 'value'),
        State('input-windspeed', 'value'),
        State('input-lag-1', 'value'),
        State('input-lag-7', 'value'),
        State('input-rolling-7', 'value'),
        State('input-is-holiday', 'value')
    )
    def make_prediction(
        n_clicks, prediction_date,
        temp_mean, temp_max, temp_min, humidity, precipitation, windspeed,
        lag_1, lag_7, rolling_7, is_holiday
    ):
        """Make energy demand prediction."""
        if not n_clicks:
            # Return default state
            importance_df = get_feature_importance()
            fig = create_feature_importance_chart(importance_df)

            return None, html.Div([
                html.P('Enter parameters and click "PREDICT DEMAND"', className='text-muted text-center mt-4')
            ]), fig

        # Validate inputs
        if not all([temp_mean is not None, lag_1 is not None, lag_7 is not None, rolling_7 is not None]):
            importance_df = get_feature_importance()
            fig = create_feature_importance_chart(importance_df)
            return None, dbc.Alert(
                '[Warning] Please fill in all required fields',
                color='warning'
            ), fig

        try:
            # Parse date
            if isinstance(prediction_date, str):
                target_date = datetime.strptime(prediction_date, '%Y-%m-%d')
            else:
                target_date = prediction_date

            # Prepare features
            features = {
                'temp_mean': float(temp_mean),
                'temp_max': float(temp_max or temp_mean + 5),
                'temp_min': float(temp_min or temp_mean - 5),
                'humidity': float(humidity or 75),
                'precipitation': float(precipitation or 0),
                'windspeed': float(windspeed or 5),
                'temp_range': float((temp_max or temp_mean + 5) - (temp_min or temp_mean - 5)),
                'day_of_week': target_date.weekday(),
                'month': target_date.month,
                'day_of_year': target_date.timetuple().tm_yday,
                'week_of_year': target_date.isocalendar().week,
                'quarter': (target_date.month - 1) // 3 + 1,
                'is_weekend': 1 if target_date.weekday() in [5, 6] else 0,
                'is_holiday': 1 if is_holiday else 0,
                'season': get_season(target_date.month),
                'demand_lag_1': float(lag_1),
                'demand_lag_7': float(lag_7),
                'demand_rolling_7': float(rolling_7)
            }

            # Make prediction
            prediction, info = predict_demand(features)

            # Get feature importance
            importance_df = get_feature_importance()
            fig = create_feature_importance_chart(importance_df)

            # Create result card
            result_card = create_prediction_card(
                prediction=info['prediction_mwh'],
                confidence_low=info['confidence_low'],
                confidence_high=info['confidence_high'],
                model_r2=info['model_r2'],
                model_mape=info['model_mape']
            )

            # Add prediction details
            result_with_details = html.Div([
                result_card,
                html.Hr(),
                html.Small([
                    f'Date: {target_date.strftime("%B %d, %Y")} | ',
                    f'Temp: {temp_mean}C | ',
                    f'Model: {info["model_type"]}'
                ], className='text-muted d-block mt-2')
            ])

            return info, result_with_details, fig

        except Exception as e:
            importance_df = get_feature_importance()
            fig = create_feature_importance_chart(importance_df)
            return None, dbc.Alert(
                f'[Error] Prediction error: {str(e)}',
                color='danger'
            ), fig

    @app.callback(
        Output('recent-trend-chart', 'figure'),
        Input('prediction-date', 'date')
    )
    def update_recent_trend(prediction_date):
        """Update recent demand trend chart."""
        df = load_energy_data()

        # Get last 30 days
        if len(df) > 30:
            recent_df = df.tail(30)
        else:
            recent_df = df

        fig = create_demand_trend_chart(recent_df, show_rolling_avg=True)

        # Update title
        fig.update_layout(
            title='Last 30 Days Demand',
            height=250,
            margin=dict(l=30, r=30, t=40, b=30)
        )

        return fig

    @app.callback(
        Output('input-lag-1', 'value'),
        Output('input-lag-7', 'value'),
        Output('input-rolling-7', 'value'),
        Input('prediction-date', 'date')
    )
    def suggest_demand_values(prediction_date):
        """Suggest demand values based on historical data."""
        df = load_energy_data()

        if df.empty:
            return 32000, 31500, 31800

        # Get recent values
        recent = df['Energy Requirement'].tail(7)

        if len(recent) >= 1:
            lag_1 = recent.iloc[-1]
        else:
            lag_1 = 32000

        if len(recent) >= 7:
            lag_7 = recent.iloc[0]
        else:
            lag_7 = lag_1

        rolling_7 = recent.mean() if len(recent) > 0 else 31800

        return int(lag_1), int(lag_7), int(rolling_7)


def get_season(month: int) -> int:
    """Map month to season number."""
    season_map = {
        12: 0, 1: 0, 2: 0,  # Winter
        3: 1, 4: 1,         # Spring
        5: 2, 6: 2, 7: 2,   # Summer
        8: 3, 9: 3,         # Monsoon
        10: 4, 11: 4        # Autumn
    }
    return season_map.get(month, 0)