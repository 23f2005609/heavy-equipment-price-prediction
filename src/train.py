
import os
import sys
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from feature_engineering import feature_engineering
from preprocessing import Preprocessor
# =========================================================
# Paths
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "train.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)


# =========================================================
# Load Data
# =========================================================

print("Loading training data...")

train = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

print("Dataset Shape:", train.shape)


# =========================================================
# Separate Features and Target
# =========================================================

X = train.drop(columns=["TargetValue"])

y = np.log1p(train["TargetValue"])


# =========================================================
# Feature Engineering
# =========================================================

print("Applying feature engineering...")

X = feature_engineering(X)

print("Feature Engineered Shape:", X.shape)


# =========================================================
# Preprocessing
# =========================================================

print("Fitting preprocessing pipeline...")

processor = Preprocessor()

X_processed = processor.fit_transform(X)

print("Processed Shape:", X_processed.shape)


# =========================================================
# Final XGBoost
# =========================================================

print("\nTraining XGBoost...")

xgb_model = XGBRegressor(
    n_estimators=11084,
    learning_rate=0.008,
    max_depth=10,
    min_child_weight=5,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42,
    eval_metric="rmse"
)

xgb_model.fit(
    X_processed,
    y
)


# =========================================================
# Final CatBoost
# =========================================================

print("\nTraining CatBoost...")

cat_model = CatBoostRegressor(
    iterations=12000,
    learning_rate=0.03,
    depth=10,
    l2_leaf_reg=3,
    random_strength=2,
    bagging_temperature=0,
    loss_function="RMSE",
    random_seed=42,
    verbose=200
)

cat_model.fit(
    X_processed,
    y
)


# =========================================================
# Save Everything
# =========================================================

print("\nSaving models...")

joblib.dump(
    processor,
    os.path.join(MODEL_DIR, "preprocessor.pkl")
)

joblib.dump(
    xgb_model,
    os.path.join(MODEL_DIR, "xgb_model.pkl")
)

joblib.dump(
    cat_model,
    os.path.join(MODEL_DIR, "cat_model.pkl")
)

joblib.dump(
    list(X_processed.columns),
    os.path.join(MODEL_DIR, "feature_columns.pkl")
)

print("\nTraining Completed Successfully!")

print("\nSaved Files:")

print("✓ preprocessor.pkl")
print("✓ xgb_model.pkl")
print("✓ cat_model.pkl")
print("✓ feature_columns.pkl")