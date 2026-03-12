"""
Dashboard Chart Components
Reusable Plotly chart functions for the dashboard.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple


# Color palette for consistency
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#28A745',
    'danger': '#DC3545',
    'info': '#17A2B8',
    'nea': '#1f77b4',
    'subsidiary': '#ff7f0e',
    'ipp': '#2ca02c',
    'import': '#d62728',
    'export': '#9467bd',
    'background': '#f8f9fa'
}

# Generation source colors
GEN_COLORS = {
    'NEA': '#1f77b4',
    'Subsidiary': '#ff7f0e',
    'IPP': '#2ca02c',
    'Import': '#d62728',
    'Export': '#9467bd'
}


def create_demand_trend_chart(
    df: pd.DataFrame,
    date_range: Optional[Tuple] = None,
    show_rolling_avg: bool = True
) -> go.Figure:
    """
    Create energy demand trend line chart.

    Args:
        df: Energy DataFrame with 'date' and 'Energy Requirement' columns
        date_range: Optional tuple of (start_date, end_date)
        show_rolling_avg: Whether to show 7-day rolling average

    Returns:
        Plotly Figure object
    """
    # Filter by date range
    if date_range:
        start_date, end_date = date_range
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

    fig = go.Figure()

    # Main demand line
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['Energy Requirement'],
        mode='lines',
        name='Daily Demand',
        line=dict(color=COLORS['primary'], width=1.5),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Demand: %{y:,.0f} MWh<extra></extra>'
    ))

    # Rolling average
    if show_rolling_avg and len(df) > 7:
        rolling = df['Energy Requirement'].rolling(window=7).mean()
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=rolling,
            mode='lines',
            name='7-Day Average',
            line=dict(color=COLORS['danger'], width=2, dash='dot'),
            hovertemplate='<b>7-Day Avg</b><br>%{y:,.0f} MWh<extra></extra>'
        ))

    fig.update_layout(
        title='Energy Demand Trend',
        xaxis_title='Date',
        yaxis_title='Energy Demand (MWh)',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=0, r=0, t=50, b=0),
        height=350
    )

    return fig


def create_generation_mix_chart(
    df: pd.DataFrame,
    date_range: Optional[Tuple] = None
) -> go.Figure:
    """
    Create stacked area chart showing generation mix over time.

    Args:
        df: Energy DataFrame with generation columns
        date_range: Optional date filter

    Returns:
        Plotly Figure object
    """
    if date_range:
        start_date, end_date = date_range
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

    fig = go.Figure()

    # Add traces for each generation source
    sources = [
        ('Energy_generation_NEA', 'NEA Generation', GEN_COLORS['NEA']),
        ('Energy_generation_NEA Subsidiary', 'NEA Subsidiary', GEN_COLORS['Subsidiary']),
        ('Energy_generation_IPP', 'IPP', GEN_COLORS['IPP']),
        ('Energy_generation_Import', 'Import', GEN_COLORS['Import'])
    ]

    for col, name, color in sources:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df[col],
                mode='lines',
                name=name,
                stackgroup='one',
                line=dict(color=color, width=0.5),
                hovertemplate=f'<b>{name}</b><br>%{{y:,.0f}} MWh<extra></extra>'
            ))

    fig.update_layout(
        title='Generation Mix Over Time',
        xaxis_title='Date',
        yaxis_title='Energy (MWh)',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=0, r=0, t=50, b=0),
        height=350
    )

    return fig


def create_seasonal_pattern_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create box plot showing demand distribution by season.

    Args:
        df: Energy DataFrame with 'season' column

    Returns:
        Plotly Figure object
    """
    season_order = ['Winter', 'Spring', 'Summer', 'Monsoon', 'Autumn']
    season_colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']

    # Map season numbers to names if needed
    if 'season' in df.columns and df['season'].dtype in ['int64', 'float64']:
        season_map = {0: 'Winter', 1: 'Spring', 2: 'Summer', 3: 'Monsoon', 4: 'Autumn'}
        df = df.copy()
        df['season_name'] = df['season'].map(season_map)
        season_col = 'season_name'
    else:
        season_col = 'season'

    fig = go.Figure()

    for i, season in enumerate(season_order):
        if season_col in df.columns:
            data = df[df[season_col] == season]['Energy Requirement']
            if len(data) > 0:
                fig.add_trace(go.Box(
                    y=data,
                    name=season,
                    marker_color=season_colors[i],
                    boxpoints='outliers'
                ))

    fig.update_layout(
        title='Seasonal Demand Distribution',
        yaxis_title='Energy Demand (MWh)',
        showlegend=False,
        margin=dict(l=0, r=0, t=50, b=0),
        height=350
    )

    return fig


def create_weekly_pattern_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create bar chart showing average demand by day of week.

    Args:
        df: Energy DataFrame with 'day_of_week' column

    Returns:
        Plotly Figure object
    """
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Calculate average by day
    weekly_avg = df.groupby('day_of_week')['Energy Requirement'].mean()

    fig = go.Figure(data=[
        go.Bar(
            x=day_names,
            y=[weekly_avg.get(i, 0) for i in range(7)],
            marker_color=[COLORS['primary'] if i < 5 else COLORS['secondary'] for i in range(7)],
            text=[f'{weekly_avg.get(i, 0):,.0f}' for i in range(7)],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Avg: %{y:,.0f} MWh<extra></extra>'
        )
    ])

    fig.update_layout(
        title='Average Demand by Day of Week',
        yaxis_title='Average Energy Demand (MWh)',
        showlegend=False,
        margin=dict(l=0, r=0, t=50, b=0),
        height=350
    )

    return fig


def create_weather_correlation_chart(
    df: pd.DataFrame,
    weather_col: str = 'temp_mean'
) -> go.Figure:
    """
    Create scatter plot showing weather-demand correlation.

    Args:
        df: DataFrame with weather and energy data
        weather_col: Weather variable to plot

    Returns:
        Plotly Figure object
    """
    fig = px.scatter(
        df,
        x=weather_col,
        y='Energy Requirement',
        color='season',
        opacity=0.6,
        trendline='ols',
        labels={
            weather_col: weather_col.replace('_', ' ').title(),
            'Energy Requirement': 'Energy Demand (MWh)'
        }
    )

    fig.update_layout(
        title=f'Weather Correlation: {" ".join(weather_col.split("_")).title()} vs Demand',
        margin=dict(l=0, r=0, t=50, b=0),
        height=350
    )

    return fig


def create_prediction_chart(
    dates: pd.Series,
    actual: pd.Series,
    predicted: np.ndarray,
    title: str = 'Actual vs Predicted Demand'
) -> go.Figure:
    """
    Create line chart comparing actual and predicted values.

    Args:
        dates: Date series
        actual: Actual values
        predicted: Predicted values
        title: Chart title

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=actual,
        mode='lines',
        name='Actual',
        line=dict(color=COLORS['primary'], width=2),
        hovertemplate='<b>Actual</b><br>%{y:,.0f} MWh<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=dates,
        y=predicted,
        mode='lines',
        name='Predicted',
        line=dict(color=COLORS['danger'], width=2, dash='dot'),
        hovertemplate='<b>Predicted</b><br>%{y:,.0f} MWh<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Energy Demand (MWh)',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=0, r=0, t=50, b=0),
        height=350
    )

    return fig


def create_forecast_chart(
    historical: pd.DataFrame,
    forecast_dates: List,
    forecast_values: List,
    confidence_low: List = None,
    confidence_high: List = None
) -> go.Figure:
    """
    Create chart showing historical data and forecast.

    Args:
        historical: Historical DataFrame
        forecast_dates: List of forecast dates
        forecast_values: List of forecast values
        confidence_low: Lower confidence bounds
        confidence_high: Upper confidence bounds

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Historical data
    fig.add_trace(go.Scatter(
        x=historical['date'],
        y=historical['Energy Requirement'],
        mode='lines',
        name='Historical',
        line=dict(color=COLORS['primary'], width=2),
        hovertemplate='<b>Historical</b><br>%{y:,.0f} MWh<extra></extra>'
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_values,
        mode='lines',
        name='Forecast',
        line=dict(color=COLORS['accent'], width=2, dash='dash'),
        hovertemplate='<b>Forecast</b><br>%{y:,.0f} MWh<extra></extra>'
    ))

    # Confidence interval
    if confidence_low and confidence_high:
        fig.add_trace(go.Scatter(
            x=forecast_dates + forecast_dates[::-1],
            y=confidence_high + confidence_low[::-1],
            fill='toself',
            fillcolor='rgba(241, 143, 1, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=True,
            name='Confidence Interval'
        ))

    fig.update_layout(
        title='Energy Demand Forecast',
        xaxis_title='Date',
        yaxis_title='Energy Demand (MWh)',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=0, r=0, t=50, b=0),
        height=400
    )

    return fig


def create_feature_importance_chart(importance_df: pd.DataFrame) -> go.Figure:
    """
    Create horizontal bar chart for feature importance.

    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns

    Returns:
        Plotly Figure object
    """
    # Sort and take top 10
    df = importance_df.nlargest(10, 'importance').sort_values('importance')

    # Clean feature names
    df['feature_clean'] = df['feature'].str.replace('_', ' ').str.title()

    fig = go.Figure(data=[
        go.Bar(
            x=df['importance'],
            y=df['feature_clean'],
            orientation='h',
            marker_color=COLORS['primary'],
            text=df['importance'].round(3),
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>'
        )
    ])

    fig.update_layout(
        title='Feature Importance',
        xaxis_title='Importance',
        yaxis_title='',
        margin=dict(l=100, r=0, t=50, b=0),
        height=350
    )

    return fig


def create_generation_pie_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create pie chart showing generation source distribution.

    Args:
        df: Energy DataFrame

    Returns:
        Plotly Figure object
    """
    totals = {
        'NEA': df['Energy_generation_NEA'].sum(),
        'Subsidiary': df['Energy_generation_NEA Subsidiary'].sum(),
        'IPP': df['Energy_generation_IPP'].sum(),
        'Import': df['Energy_generation_Import'].sum()
    }

    fig = go.Figure(data=[
        go.Pie(
            labels=list(totals.keys()),
            values=list(totals.values()),
            marker_colors=list(GEN_COLORS.values())[:4],
            hole=0.3,
            hovertemplate='<b>%{label}</b><br>%{value:,.0f} MWh<br>%{percent}<extra></extra>'
        )
    ])

    fig.update_layout(
        title='Generation Source Distribution',
        margin=dict(l=0, r=0, t=50, b=0),
        height=350
    )

    return fig