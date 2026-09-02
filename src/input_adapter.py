import pandas as pd


def prepare_input(user_input):
    """
    Convert frontend input into the raw feature format
    expected by the trained ML pipeline.
    """

    data = {
        # Required structural columns
        "TransactionID": 0,
        "AssetID": 0,
        "ProductConfigID": 0,

        # Main user inputs
        "DataOriginCode": "web",
        "VendorPartnerID": None,

        "ManufactureYear": user_input.get("ManufactureYear"),
        "OperationalHoursMeter": user_input.get("OperationalHoursMeter"),
        "UtilizationTier": user_input.get("UtilizationTier"),

        "TransactionDate": user_input.get(
            "TransactionDate",
            "2025-08-01"
        ),

        "Spec_FullDescriptor": user_input.get(
            "Spec_FullDescriptor"
        ),

        "Spec_BaseClass": None,
        "Spec_SubClass": None,
        "Spec_ReleaseSeries": None,
        "Spec_VariantModifier": None,

        "AssetScaleFactor": user_input.get(
            "AssetScaleFactor"
        ),

        "FunctionalClassification": user_input.get(
            "FunctionalClassification"
        ),

        "RegionCode": user_input.get(
            "RegionCode"
        ),

        # Not exposed in frontend yet
        "InventoryGroupCategory": None,
        "InventoryGroupDescription": None,

        # Other raw columns
        "col1": None,
        "CabinType": user_input.get("CabinType"),
        "Forks": user_input.get("Forks"),
        "col3": None,
        "col4": None,

        "DrivetrainType": user_input.get(
            "DrivetrainType"
        ),

        "col5": None,
        "col6": None,
        "col7": None,
        "col8": None,
        "col9": None,
        "col10": None,
        "col11": None,
        "col12": None,
        "col13": None,
        "col14": None,
        "col15": None,
        "col16": None,
        "col18": None,
        "col19": None,
        "col20": None,
        "col21": None,
        "col22": None,
        "col23": None,
        "col24": None,
        "col25": None,
        "col27": None,
        "col28": None,
        "col29": None,
        "col30": None,
    }

    return pd.DataFrame([data])