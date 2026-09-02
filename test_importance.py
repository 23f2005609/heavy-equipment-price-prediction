import os
import sys
import importlib.util
import joblib
import pandas as pd


# -----------------------------------------
# Project paths
# -----------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
MODEL_DIR = os.path.join(BASE_DIR, "models")


# -----------------------------------------
# Load preprocessing module
# -----------------------------------------

preprocessing_path = os.path.join(
    SRC_DIR,
    "preprocessing.py"
)

spec = importlib.util.spec_from_file_location(
    "preprocessing",
    preprocessing_path
)

preprocessing_module = importlib.util.module_from_spec(spec)

sys.modules["preprocessing"] = preprocessing_module

spec.loader.exec_module(preprocessing_module)


# -----------------------------------------
# Load model and feature names
# -----------------------------------------

xgb_model = joblib.load(
    os.path.join(MODEL_DIR, "xgb_model.pkl")
)

feature_columns = joblib.load(
    os.path.join(MODEL_DIR, "feature_columns.pkl")
)


# -----------------------------------------
# Feature importance
# -----------------------------------------

importance = xgb_model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)


# -----------------------------------------
# Display top 30
# -----------------------------------------

print("===================================")
print("TOP 30 XGBOOST FEATURES")
print("===================================")

print(
    importance_df.head(30).to_string(index=False)
)

print("===================================")