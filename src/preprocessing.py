import re
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


class Preprocessor:

    def __init__(self):

        # Numerical columns → median
        self.median_cols = [
            "ManufactureYear",
            "OperationalHoursMeter",
            "MachineAge"
        ]

        # Engineered numerical columns → 0
        self.zero_cols = [
            "ModelNumberNum",
            "Age_x_ModelNumber",
            "CapacityValue"
        ]

        # High-cardinality categorical columns
        self.freq_cols = [
            "VendorPartnerID",
            "RegionCode",
            "FunctionalClassification",
            "Spec_SubClass",
            "Spec_ReleaseSeries",
            "Spec_VariantModifier",
            "Spec_BaseClass",
            "Spec_FullDescriptor",
            "ModelNumber",
            "ModelSuffix",
            "ModelPrefix"
        ]

        self.preprocessor = None
        self.freq_maps = {}
        self.feature_names = None


    # -----------------------------------------------------
    # Fit
    # -----------------------------------------------------

    def fit(self, X):

        X = X.copy()

        # Low-cardinality categorical columns
        onehot_cols = [
            c for c in X.select_dtypes(include="object").columns
            if c not in self.freq_cols
        ]

        # Numerical pipeline
        num_pipe = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ])

        # One-hot categorical pipeline
        cat_pipe = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Missing"
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop="first"
                )
            )
        ])

        # Frequency columns
        freq_pipe = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Missing"
                )
            )
        ])

        # Zero-imputation pipeline
        zero_pipe = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value=0
                )
            )
        ])

        # ColumnTransformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    num_pipe,
                    self.median_cols
                ),
                (
                    "cat",
                    cat_pipe,
                    onehot_cols
                ),
                (
                    "freq",
                    freq_pipe,
                    self.freq_cols
                ),
                (
                    "zero",
                    zero_pipe,
                    self.zero_cols
                )
            ],
            remainder="passthrough",
            verbose_feature_names_out=False
        )

        # Fit preprocessing
        self.preprocessor.fit(X)

        # -------------------------------------------------
        # Store frequency maps
        # -------------------------------------------------

        for col in self.freq_cols:

            values = X[col].fillna("Missing")

            self.freq_maps[col] = (
                values.value_counts()
                .to_dict()
            )

        self.feature_names = (
            self.preprocessor
            .get_feature_names_out()
        )

        return self


    # -----------------------------------------------------
    # Transform
    # -----------------------------------------------------

    def transform(self, X):

        X = X.copy()

        # Apply ColumnTransformer
        transformed = self.preprocessor.transform(X)

        processed = pd.DataFrame(
            transformed,
            columns=self.feature_names,
            index=X.index
        )

        # -------------------------------------------------
        # Frequency Encoding
        # -------------------------------------------------

        for col in self.freq_cols:

            values = X[col].fillna("Missing")

            processed[col + "_freq"] = (
                values
                .map(self.freq_maps[col])
                .fillna(0)
            )

        # Remove original high-cardinality columns
        processed.drop(
            columns=[
                col
                for col in self.freq_cols
                if col in processed.columns
            ],
            inplace=True
        )

        # Float32
        processed = processed.astype(np.float32)

        # Clean column names
        processed.columns = [
            re.sub(
                r"[^A-Za-z0-9_]+",
                "_",
                str(col)
            )
            for col in processed.columns
        ]

        return processed


    # -----------------------------------------------------
    # Fit + Transform
    # -----------------------------------------------------

    def fit_transform(self, X):

        self.fit(X)

        return self.transform(X)