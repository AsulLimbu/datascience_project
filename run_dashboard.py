#!/usr/bin/env python
"""
NEA Energy Dashboard Launcher

Run this script to start the interactive dashboard for:
- Historical energy data visualization
- Demand forecasting with ML models

Usage:
    python run_dashboard.py

The dashboard will be available at http://localhost:8050
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.dashboard.app import app


def main():
    """Launch the dashboard server."""
    print("=" * 60)
    print("NEA Energy Dashboard")
    print("=" * 60)
    print()
    print("[Historical Analysis]")
    print("   - Energy demand trends")
    print("   - Generation mix visualization")
    print("   - Seasonal and weekly patterns")
    print("   - Weather correlation analysis")
    print()
    print("[Demand Forecasting]")
    print("   - Real-time weather integration")
    print("   - ML-powered predictions (Ridge Regression)")
    print("   - Feature importance analysis")
    print()
    print("=" * 60)
    print("Starting server...")
    print()
    print("Open http://localhost:8050 in your browser")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    print()

    # Run the server
    app.run(
        debug=True,
        port=8050,
        host='127.0.0.1'
    )


if __name__ == '__main__':
    main()