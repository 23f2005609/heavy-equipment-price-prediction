import os
import joblib
import numpy as np
import pandas as pd

from src.feature_engineering import feature_engineering


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")


# --------------------------------------------------
# Load trained objects
# --------------------------------------------------

processor = joblib.load(
    os.path.join(MODEL_DIR, "preprocessor.pkl")
)

xgb_model = joblib.load(
    os.path.join(MODEL_DIR, "xgb_model.pkl")
)

cat_model = joblib.load(
    os.path.join(MODEL_DIR, "cat_model.pkl")
)

feature_columns = joblib.load(
    os.path.join(MODEL_DIR, "feature_columns.pkl")
)


# --------------------------------------------------
# Prediction function
# --------------------------------------------------

def predict_price(data):
    """
    Predict equipment resale price from raw input data.
    """

    # 1. Convert dictionary to DataFrame
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    else:
        data = data.copy()

    # 2. Feature Engineering
    data = feature_engineering(data)

    # 3. Preprocessing
    # Your custom Preprocessor already returns a DataFrame
    X_processed = processor.transform(data)

    # 4. Feature Alignment
    X_processed = X_processed.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # 5. XGBoost prediction
    xgb_pred = xgb_model.predict(X_processed)

    # 6. CatBoost prediction
    cat_pred = cat_model.predict(X_processed)

    # 7. 70:30 Ensemble
    log_prediction = (
        0.7 * xgb_pred +
        0.3 * cat_pred
    )

    # 8. Convert log prediction back to original price
    prediction = np.expm1(log_prediction)

    # 9. Prevent negative price
    prediction = np.clip(prediction, 0, None)

    return float(prediction[0])