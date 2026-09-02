import pandas as pd
import numpy as np
import re


# ---------------------------------------------------------
# Capacity extraction
# ---------------------------------------------------------

def capacity(text):
    """
    Extract a numeric capacity value from
    FunctionalClassification.
    """
    if pd.isna(text):
        return np.nan

    text = str(text)

    match = re.search(r'(\d+(?:\.\d+)?)', text)

    if match:
        return float(match.group(1))

    return np.nan


# ---------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------

def feature_engineering(df):
    """
    Apply all feature engineering steps used by
    the final Kaggle model.
    """

    df = df.copy()

    # -----------------------------------------------------
    # Remove extremely sparse columns
    # -----------------------------------------------------

    drop_cols = [
        "traction_system_type",
        "fluid_flow_rate",
        "engine_intake_type",
        "extension_arm_layout"
    ]

    df.drop(
        columns=drop_cols,
        errors="ignore",
        inplace=True
    )

    # -----------------------------------------------------
    # Fix invalid ManufactureYear
    # -----------------------------------------------------

    df["ManufactureYear"] = (
        df["ManufactureYear"]
        .replace(1001, np.nan)
    )

    # -----------------------------------------------------
    # Date Features
    # -----------------------------------------------------

    df["TransactionDate"] = pd.to_datetime(
        df["TransactionDate"],
        errors="coerce"
    )

    df["SaleYear"] = df["TransactionDate"].dt.year
    df["SaleMonth"] = df["TransactionDate"].dt.month
    df["SaleDay"] = df["TransactionDate"].dt.day
    df["SaleWeekday"] = df["TransactionDate"].dt.dayofweek
    df["SaleQuarter"] = df["TransactionDate"].dt.quarter

    df["SaleWeek"] = (
        df["TransactionDate"]
        .dt.isocalendar()
        .week
        .astype("float")
    )

    # TransactionDate is no longer needed
    df.drop(
        columns=["TransactionDate"],
        inplace=True
    )

    # -----------------------------------------------------
    # Machine Age
    # -----------------------------------------------------

    df["MachineAge"] = (
        df["SaleYear"] -
        df["ManufactureYear"]
    )

    # Invalid negative ages → missing
    df.loc[
        df["MachineAge"] < 0,
        "MachineAge"
    ] = np.nan

    # -----------------------------------------------------
    # Fix category typo
    # -----------------------------------------------------

    if "DrivetrainType" in df.columns:
        df["DrivetrainType"] = (
            df["DrivetrainType"]
            .replace({
                "Autoshift": "AutoShift"
            })
        )

    # -----------------------------------------------------
    # Model Features
    # -----------------------------------------------------

    descriptor = (
        df["Spec_FullDescriptor"]
        .astype(str)
    )

    df["ModelNumber"] = (
        descriptor
        .str.extract(r"(\d+)")[0]
    )

    df["ModelSuffix"] = (
        descriptor
        .str.extract(r"([A-Za-z]+)$")[0]
    )

    df["ModelPrefix"] = (
        descriptor
        .str.extract(r"^([A-Za-z]+)")[0]
    )

    # -----------------------------------------------------
    # Binary Model Indicators
    # -----------------------------------------------------

    df["HasLC"] = (
        descriptor
        .str.contains("LC", regex=False)
        .astype(int)
    )

    df["HasXL"] = (
        descriptor
        .str.contains("XL", regex=False)
        .astype(int)
    )

    df["HasGP"] = (
        descriptor
        .str.contains("GP", regex=False)
        .astype(int)
    )

    df["HasBL"] = (
        descriptor
        .str.contains("BL", regex=False)
        .astype(int)
    )

    df["HasCL"] = (
        descriptor
        .str.contains("CL", regex=False)
        .astype(int)
    )

    df["HasELC"] = (
        descriptor
        .str.contains("ELC", regex=False)
        .astype(int)
    )

    # -----------------------------------------------------
    # Numeric Model Number
    # -----------------------------------------------------

    df["ModelNumberNum"] = pd.to_numeric(
        df["ModelNumber"],
        errors="coerce"
    )

    # -----------------------------------------------------
    # Interaction Feature
    # -----------------------------------------------------

    df["Age_x_ModelNumber"] = (
        df["MachineAge"] *
        df["ModelNumberNum"]
    )

    # -----------------------------------------------------
    # Capacity Feature
    # -----------------------------------------------------

    df["CapacityValue"] = (
        df["FunctionalClassification"]
        .apply(capacity)
    )

    # -----------------------------------------------------
    # Descriptor Features
    # -----------------------------------------------------

    df["DescriptorLength"] = (
        df["Spec_FullDescriptor"]
        .fillna("")
        .str.len()
    )

    df["DescriptorWords"] = (
        df["Spec_FullDescriptor"]
        .fillna("")
        .str.split()
        .str.len()
    )

    # -----------------------------------------------------
    # Missingness Features
    # -----------------------------------------------------

    df["HasOperationalHours"] = (
        df["OperationalHoursMeter"]
        .notna()
        .astype(int)
    )

    df["HasVariantModifier"] = (
        df["Spec_VariantModifier"]
        .notna()
        .astype(int)
    )

    return df