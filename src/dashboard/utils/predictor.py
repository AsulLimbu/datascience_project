"""
Model Predictor Utility
Loads saved model and makes energy demand predictions.
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from functools import lru_cache
from typing import Dict, Optional, Tuple

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


@lru_cache(maxsize=1)
def load_model():
    """
    Load the trained Ridge Regression model.

    Returns:
        Trained model object
    """
    model_path = MODELS_DIR / "best_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Please run the prediction notebook first."
        )

    return joblib.load(model_path)


@lru_cache(maxsize=1)
def load_scaler():
    """
    Load the StandardScaler used for feature scaling.

    Returns:
        Fitted scaler object
    """
    scaler_path = MODELS_DIR / "scaler.joblib"

    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler not found at {scaler_path}. "
            "Please run the prediction notebook first."
        )

    return joblib.load(scaler_path)


def predict_demand(features_dict: Dict) -> Tuple[float, Dict]:
    """
    Make energy demand prediction from input features.

    Args:
        features_dict: Dictionary with feature values

    Returns:
        Tuple of (prediction, info_dict)
    """
    model = load_model()
    scaler = load_scaler()

    # Feature order must match training
    feature_order = [
        'temp_mean', 'temp_max', 'temp_min', 'humidity', 'precipitation',
        'windspeed', 'temp_range', 'day_of_week', 'month', 'day_of_year',
        'week_of_year', 'quarter', 'is_weekend', 'is_holiday', 'season',
        'demand_lag_1', 'demand_lag_7', 'demand_rolling_7'
    ]

    # Create DataFrame with correct order
    df = pd.DataFrame([[features_dict.get(f, 0) for f in feature_order]],
                      columns=feature_order)

    # Scale features (Ridge model requires scaling)
    df_scaled = scaler.transform(df)

    # Make prediction
    prediction = model.predict(df_scaled)[0]

    # Model metrics for confidence interval
    model_info = {
        'prediction_mwh': round(prediction, 2),
        'confidence_low': round(prediction - 1660.94, 2),  # Test RMSE
        'confidence_high': round(prediction + 1660.94, 2),
        'model_type': 'Ridge Regression',
        'model_r2': 0.8991,
        'model_mape': 4.32
    }

    return prediction, model_info


def get_feature_importance() -> pd.DataFrame:
    """
    Get feature importance/coefficients from the model.

    Returns:
        DataFrame with feature names and importance scores
    """
    model = load_model()

    # Feature names (must match training order)
    feature_names = [
        'temp_mean', 'temp_max', 'temp_min', 'humidity', 'precipitation',
        'windspeed', 'temp_range', 'day_of_week', 'month', 'day_of_year',
        'week_of_year', 'quarter', 'is_weekend', 'is_holiday', 'season',
        'demand_lag_1', 'demand_lag_7', 'demand_rolling_7'
    ]

    # For Ridge, coefficients represent importance
    if hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)
    else:
        importance = np.zeros(len(feature_names))

    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })

    # Normalize to 0-1 scale
    df['importance_normalized'] = df['importance'] / df['importance'].max()

    # Sort by importance
    df = df.sort_values('importance', ascending=False).reset_index(drop=True)

    return df


def validate_features(features_dict: Dict) -> Tuple[bool, str]:
    """
    Validate that all required features are present and valid.

    Args:
        features_dict: Dictionary with feature values

    Returns:
        Tuple of (is_valid, error_message)
    """
    required_features = [
        'temp_mean', 'humidity', 'precipitation', 'windspeed',
        'demand_lag_1', 'demand_lag_7', 'demand_rolling_7'
    ]

    missing = []
    invalid = []

    for feat in required_features:
        if feat not in features_dict:
            missing.append(feat)
        elif features_dict[feat] is None:
            missing.append(feat)
        elif pd.isna(features_dict[feat]):
            missing.append(feat)

    # Check for reasonable ranges
    ranges = {
        'temp_mean': (-10, 50),
        'humidity': (0, 100),
        'precipitation': (0, 500),
        'windspeed': (0, 100),
        'demand_lag_1': (0, 100000),
        'demand_lag_7': (0, 100000),
        'demand_rolling_7': (0, 100000)
    }

    for feat, (min_val, max_val) in ranges.items():
        if feat in features_dict and features_dict[feat] is not None:
            val = features_dict[feat]
            if not (min_val <= val <= max_val):
                invalid.append(f"{feat}={val} (expected {min_val}-{max_val})")

    if missing:
        return False, f"Missing features: {', '.join(missing)}"
    if invalid:
        return False, f"Invalid values: {', '.join(invalid)}"

    return True, "All features valid"


def batch_predict(features_df: pd.DataFrame) -> np.ndarray:
    """
    Make predictions for multiple samples.

    Args:
        features_df: DataFrame with feature columns

    Returns:
        Array of predictions
    """
    model = load_model()
    scaler = load_scaler()

    # Scale features
    features_scaled = scaler.transform(features_df)

    # Predict
    predictions = model.predict(features_scaled)

    return predictions


def clear_cache():
    """Clear cached model and scaler."""
    load_model.cache_clear()
    load_scaler.cache_clear()


if __name__ == "__main__":
    # Test model loading and prediction
    print("Testing model prediction...")

    # Sample input
    sample_features = {
        'temp_mean': 25.0,
        'temp_max': 30.0,
        'temp_min': 20.0,
        'humidity': 75.0,
        'precipitation': 5.0,
        'windspeed': 3.5,
        'temp_range': 10.0,
        'day_of_week': 2,
        'month': 8,
        'day_of_year': 220,
        'week_of_year': 32,
        'quarter': 3,
        'is_weekend': 0,
        'is_holiday': 0,
        'season': 3,
        'demand_lag_1': 32000,
        'demand_lag_7': 31500,
        'demand_rolling_7': 31800
    }

    prediction, info = predict_demand(sample_features)
    print(f"\nPredicted Demand: {prediction:,.0f} MWh")
    print(f"\nModel Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    print("\nFeature Importance:")
    importance_df = get_feature_importance()
    print(importance_df.head(10))