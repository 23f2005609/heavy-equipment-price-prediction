import os
import sys
import importlib.util

# Fix pickle compatibility
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")

preprocessing_path = os.path.join(SRC_DIR, "preprocessing.py")

spec = importlib.util.spec_from_file_location(
    "preprocessing",
    preprocessing_path
)

preprocessing_module = importlib.util.module_from_spec(spec)
sys.modules["preprocessing"] = preprocessing_module
spec.loader.exec_module(preprocessing_module)

sys.path.insert(0, BASE_DIR)

from src.input_adapter import prepare_input
from src.predict import predict_price


user_input = {
    "ManufactureYear": 2018,
    "OperationalHoursMeter": 5200,
    "UtilizationTier": "Medium",
    "AssetScaleFactor": "Large / Medium",
    "FunctionalClassification":
        "Hydraulic Excavator, Track - 21.0 to 24.0 Metric Tons",
    "RegionCode": "Texas",
    "CabinType": "EROPS w AC",
    "Forks": "None or Unspecified",
    "DrivetrainType": "Powershift",
    "Spec_FullDescriptor": "140G"
}

# Frontend → raw model input
raw_data = prepare_input(user_input)

print("RAW INPUT")
print("Shape:", raw_data.shape)
print()

# Raw input → model → prediction
prediction = predict_price(raw_data)

print("FRONTEND → MODEL TEST")
print(f"Predicted price : ₹{prediction:,.2f}")