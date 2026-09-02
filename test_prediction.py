import sys
import os
import pandas as pd

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "src"
    )
)

from src.predict import predict_price


# Take one real machine from the dataset
df = pd.read_csv(
    "data/train.csv",
    low_memory=False
)

sample = df.drop(
    columns=["TargetValue"]
).iloc[0].to_dict()


prediction = predict_price(sample)

print("Predicted selling price:", prediction)

print(
    "Actual selling price:",
    df["TargetValue"].iloc[0]
)