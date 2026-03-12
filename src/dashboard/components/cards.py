"""
Dashboard KPI Card Components
Bootstrap cards for displaying key performance indicators.
"""

import dash_bootstrap_components as dbc
from dash import html
import pandas as pd
from typing import Optional


def create_stat_card(
    title: str,
    value: str,
    delta: Optional[str] = None,
    delta_color: str = 'success',
    icon: str = '📊',
    description: str = ''
) -> dbc.Card:
    """
    Create a statistics card component.

    Args:
        title: Card title
        value: Main value to display
        delta: Optional change indicator
        delta_color: Color for delta (success/danger/warning/info)
        icon: Icon to display
        description: Optional description text

    Returns:
        Bootstrap Card component
    """
    children = [
        dbc.CardBody([
            html.Div([
                html.Span(icon, className='me-2', style={'fontSize': '1.5rem'}),
                html.Small(title, className='text-muted d-block')
            ], className='d-flex align-items-center'),
            html.H3(value, className='mt-2 mb-1'),
        ])
    ]

    if delta:
        children[0].children.append(
            html.Span(
                delta,
                className=f'badge bg-{delta_color} mt-1'
            )
        )

    if description:
        children[0].children.append(
            html.Small(description, className='text-muted d-block mt-1')
        )

    return dbc.Card(children, className='shadow-sm h-100')


def create_summary_cards(df: pd.DataFrame) -> list:
    """
    Create a row of summary KPI cards.

    Args:
        df: Energy DataFrame

    Returns:
        List of Card components
    """
    energy_col = 'Energy Requirement'

    # Calculate statistics
    total_energy = df[energy_col].sum()
    avg_demand = df[energy_col].mean()
    peak_demand = df[energy_col].max()

    # Calculate import dependency
    if 'Energy_generation_Import' in df.columns and 'Energy_generation_Total Energy Available' in df.columns:
        total_import = df['Energy_generation_Import'].sum()
        total_available = df['Energy_generation_Total Energy Available'].sum()
        import_pct = (total_import / total_available * 100) if total_available > 0 else 0
    else:
        import_pct = 0

    # Format numbers
    def format_mwh(value):
        if value >= 1_000_000:
            return f"{value/1_000_000:.1f}M"
        elif value >= 1_000:
            return f"{value/1_000:.0f}K"
        else:
            return f"{value:.0f}"

    cards = [
        create_stat_card(
            title='Total Energy',
            value=f'{format_mwh(total_energy)} MWh',
            icon='*',
            description='All time generation'
        ),
        create_stat_card(
            title='Avg Daily Demand',
            value=f'{avg_demand:,.0f} MWh',
            icon='^',
            description='Daily average'
        ),
        create_stat_card(
            title='Peak Demand',
            value=f'{peak_demand:,.0f} MWh',
            icon='#',
            description='Highest recorded'
        ),
        create_stat_card(
            title='Import Dependency',
            value=f'{import_pct:.1f}%',
            icon='>',
            description='From India'
        )
    ]

    return cards


def create_prediction_card(
    prediction: float,
    confidence_low: float,
    confidence_high: float,
    model_r2: float,
    model_mape: float
) -> dbc.Card:
    """
    Create a card displaying prediction results.

    Args:
        prediction: Predicted demand value
        confidence_low: Lower bound of confidence interval
        confidence_high: Upper bound of confidence interval
        model_r2: Model R² score
        model_mape: Model MAPE score

    Returns:
        Bootstrap Card component
    """
    return dbc.Card([
        dbc.CardBody([
            html.H5('Predicted Energy Demand', className='text-center text-muted mb-3'),
            html.H1(
                f'{prediction:,.0f}',
                className='text-center mb-2',
                style={'fontSize': '3rem', 'fontWeight': 'bold', 'color': '#2E86AB'}
            ),
            html.P('MWh', className='text-center text-muted mb-3'),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.Small('Confidence Range', className='text-muted d-block'),
                    html.Strong(f'{confidence_low:,.0f} - {confidence_high:,.0f} MWh')
                ], width=6),
                dbc.Col([
                    html.Small('Model Accuracy', className='text-muted d-block'),
                    html.Strong(f'R² = {model_r2:.2f} ({model_mape:.1f}% MAPE)')
                ], width=6)
            ], className='mt-2')
        ])
    ], className='shadow-lg', style={'borderWidth': '2px', 'borderColor': '#2E86AB'})


def create_info_card(
    title: str,
    content: str,
    icon: str = 'ℹ️'
) -> dbc.Card:
    """
    Create an information card.

    Args:
        title: Card title
        content: Card content text
        icon: Icon to display

    Returns:
        Bootstrap Card component
    """
    return dbc.Card([
        dbc.CardBody([
            html.H5([
                html.Span(icon, className='me-2'),
                title
            ], className='card-title'),
            html.P(content, className='card-text')
        ])
    ], className='shadow-sm')


def create_weather_card(weather_data: dict) -> dbc.Card:
    """
    Create a card displaying weather information.

    Args:
        weather_data: Dictionary with weather metrics

    Returns:
        Bootstrap Card component
    """
    return dbc.Card([
        dbc.CardBody([
            html.H5('[Weather] Data', className='mb-3'),
            dbc.Row([
                dbc.Col([
                    html.Small('Temperature', className='text-muted d-block'),
                    html.Strong(f"{weather_data.get('temp_mean', 0):.1f}°C")
                ], width=4),
                dbc.Col([
                    html.Small('Humidity', className='text-muted d-block'),
                    html.Strong(f"{weather_data.get('humidity', 0):.0f}%")
                ], width=4),
                dbc.Col([
                    html.Small('Precipitation', className='text-muted d-block'),
                    html.Strong(f"{weather_data.get('precipitation', 0):.1f} mm")
                ], width=4)
            ])
        ])
    ], className='shadow-sm')