import pandas as pd
import requests


df = pd.read_csv(
    "data/train.csv",
    low_memory=False
)

# Take one complete raw row
sample = (
    df.drop(columns=["TargetValue"])
      .iloc[0]
      .where(lambda x: x.notna(), None)
      .to_dict()
)

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json=sample
)

print("Status:", response.status_code)

print("Response:")
print(response.json())

print("\nActual price:")
print(df["TargetValue"].iloc[0])