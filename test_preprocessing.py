import pandas as pd

from src.feature_engineering import feature_engineering
from src.preprocessing import Preprocessor


# Load data
df = pd.read_csv("data/train.csv")

# Remove target
X = df.drop(columns=["TargetValue"])

# Feature engineering
X = feature_engineering(X)

# Preprocessing
processor = Preprocessor()

X_processed = processor.fit_transform(X)


print("Before preprocessing:")
print(X.shape)

print("\nAfter preprocessing:")
print(X_processed.shape)

print("\nData type:")
print(X_processed.dtypes.value_counts())

print("\nFirst 5 rows:")
print(X_processed.head())