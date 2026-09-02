import os
import pandas as pd


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "train.csv"
)


def get_reference_row():
    """
    Load one raw training row to provide fallback
    values for fields that are not entered by the user.
    """

    df = pd.read_csv(
        DATA_PATH,
        low_memory=False,
        nrows=1
    )

    df = df.drop(
        columns=["TargetValue"],
        errors="ignore"
    )

    return df.iloc[0].to_dict()