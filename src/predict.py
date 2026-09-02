import os
import joblib
import numpy as np
import pandas as pd

from src.feature_engineering import feature_engineering


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# Final deployment artifacts
processor = joblib.load(
    os.path.join(
        MODEL_DIR,
        "preprocessor_deploy.pkl"
    )
)

xgb_model = joblib.load(
    os.path.join(
        MODEL_DIR,
        "xgb_deploy_model.pkl"
    )
)

feature_columns = joblib.load(
    os.path.join(
        MODEL_DIR,
        "feature_columns_deploy.pkl"
    )
)


def predict_price(data):
    """
    Predict equipment resale price using
    the lightweight deployment XGBoost model.
    """

    if isinstance(data, dict):
        data = pd.DataFrame([data])
    else:
        data = data.copy()

    # Feature engineering
    data = feature_engineering(data)

    # Preprocessing
    X_processed = processor.transform(data)

    # Ensure exact feature order
    X_processed = X_processed.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # XGBoost prediction
    log_prediction = xgb_model.predict(
        X_processed
    )

    # Convert log prediction back to price
    prediction = np.expm1(log_prediction)

    prediction = np.clip(
        prediction,
        0,
        None
    )

    return float(prediction[0])