import pandas as pd

from src.feature_engineering import feature_engineering


df = pd.read_csv("data/train.csv")

X = df.drop(columns=["TargetValue"])

X_new = feature_engineering(X)

print("Original shape:", X.shape)
print("New shape:", X_new.shape)

print("\nNew features:")

new_features = [
    "SaleYear",
    "SaleMonth",
    "SaleDay",
    "SaleWeekday",
    "SaleQuarter",
    "SaleWeek",
    "MachineAge",
    "ModelNumber",
    "ModelSuffix",
    "ModelPrefix",
    "HasLC",
    "HasXL",
    "HasGP",
    "HasBL",
    "HasCL",
    "HasELC",
    "ModelNumberNum",
    "Age_x_ModelNumber",
    "CapacityValue",
    "DescriptorLength",
    "DescriptorWords",
    "HasOperationalHours",
    "HasVariantModifier"
]

print(
    X_new[
        [c for c in new_features if c in X_new.columns]
    ].head()
)