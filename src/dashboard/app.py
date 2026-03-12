"""
NEA Energy Dashboard - Main Application
Interactive dashboard for Nepal Electricity Authority energy data visualization and forecasting.

Usage:
    python run_dashboard.py

Or:
    python -m src.dashboard.app
"""

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from pathlib import Path

# Import layouts
from .layouts.historical import create_historical_layout
from .layouts.forecasting import create_forecasting_layout

# Import callbacks
from .callbacks.historical_callbacks import register_callbacks as register_historical_callbacks
from .callbacks.forecasting_callbacks import register_callbacks as register_forecasting_callbacks


# Initialize Dash app with Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.FLATLY,  # Clean, modern theme
        'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css'
    ],
    title='NEA Energy Dashboard',
    suppress_callback_exceptions=True,
    use_pages=False
)

# Application metadata
app.title = 'NEA Energy Dashboard'
app.description = 'Interactive dashboard for Nepal electricity demand analysis and forecasting'


def create_navbar():
    """Create the navigation bar."""
    return dbc.NavbarSimple(
        brand=[
            html.I(className='fas fa-bolt me-2', style={'color': '#F18F01'}),
            html.Span('NEA Energy Dashboard', className='fw-bold')
        ],
        brand_href='/',
        color='primary',
        dark=True,
        className='mb-3 shadow',
        children=[
            dbc.Nav([
                dbc.NavItem(
                    dbc.NavLink(
                        [html.I(className='fas fa-chart-line me-1'), ' Historical'],
                        href='/',
                        id='nav-historical',
                        active=True
                    )
                ),
                dbc.NavItem(
                    dbc.NavLink(
                        [html.I(className='fas fa-crystal-ball me-1'), ' Forecast'],
                        href='/forecast',
                        id='nav-forecast'
                    )
                )
            ], navbar=True)
        ]
    )


def create_footer():
    """Create the page footer."""
    return dbc.Container([
        html.Hr(),
        dbc.Row([
            dbc.Col([
                html.P([
                    html.Small([
                        'NEA Energy Dashboard | ',
                        html.A('Project Report', href='#', className='text-decoration-none'),
                        ' | ',
                        html.A('GitHub', href='#', className='text-decoration-none'),
                        ' | ',
                        f'Built with Dash & Plotly'
                    ], className='text-muted')
                ], className='text-center py-2')
            ])
        ])
    ], fluid=True)


# Main app layout
app.layout = html.Div([
    # URL tracking
    dcc.Location(id='url', refresh=False),

    # Navigation bar
    create_navbar(),

    # Page content container
    html.Div(id='page-content'),

    # Footer
    create_footer()
])


# Register callbacks
register_historical_callbacks(app)
register_forecasting_callbacks(app)


# Page routing callback
@app.callback(
    dash.Output('page-content', 'children'),
    [dash.Input('url', 'pathname')]
)
def display_page(pathname):
    """Route to the appropriate page based on URL."""
    if pathname == '/forecast':
        return create_forecasting_layout()
    else:  # Default to historical page
        return create_historical_layout()


# Navigation active state
@app.callback(
    [dash.Output('nav-historical', 'active'),
     dash.Output('nav-forecast', 'active')],
    [dash.Input('url', 'pathname')]
)
def update_nav_active(pathname):
    """Update active state of navigation links."""
    if pathname == '/forecast':
        return False, True
    return True, False


# Run the application
if __name__ == '__main__':
    print("=" * 60)
    print("NEA Energy Dashboard")
    print("=" * 60)
    print("\nStarting server...")
    print("Open http://localhost:8050 in your browser\n")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)

    app.run(debug=True, port=8050, host='127.0.0.1')