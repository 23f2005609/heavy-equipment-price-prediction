
# uvicorn app.main:app --reload


from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional
import sys
import os
import importlib.util

# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")


# --------------------------------------------------
# Load preprocessing module
# --------------------------------------------------
# Required because the saved preprocessor.pkl was
# created using the module name "preprocessing".

preprocessing_path = os.path.join(
    SRC_DIR,
    "preprocessing.py"
)

spec = importlib.util.spec_from_file_location(
    "preprocessing",
    preprocessing_path
)

preprocessing_module = importlib.util.module_from_spec(spec)

sys.modules["preprocessing"] = preprocessing_module

spec.loader.exec_module(preprocessing_module)


# --------------------------------------------------
# Import prediction pipeline
# --------------------------------------------------

sys.path.insert(0, BASE_DIR)

from src.input_adapter import prepare_input
from src.predict import predict_price


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Heavy Equipment Price Prediction API",
    description="Predicts the resale price of heavy equipment.",
    version="1.0.0"
)

app.mount(
    "/static",
    StaticFiles(directory=os.path.join(BASE_DIR, "app", "static")),
    name="static"
)

# --------------------------------------------------
# Request schema
# --------------------------------------------------

class EquipmentInput(BaseModel):

    ManufactureYear: int = Field(
        ...,
        description="Year the equipment was manufactured"
    )

    OperationalHoursMeter: Optional[float] = Field(
        None,
        description="Total operational hours"
    )

    UtilizationTier: Optional[str] = None

    AssetScaleFactor: Optional[str] = None

    FunctionalClassification: Optional[str] = None

    RegionCode: Optional[str] = None

    CabinType: Optional[str] = None

    Forks: Optional[str] = None

    DrivetrainType: Optional[str] = None

    Spec_FullDescriptor: Optional[str] = None


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/")
def home():
    return FileResponse(
            os.path.join(
                BASE_DIR,
                "app",
                "static",
                "index.html"
            )
        )

# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(input_data: EquipmentInput):

    try:

        # Convert Pydantic object → dictionary
        user_input = input_data.model_dump()

        # Convert frontend input → raw model format
        raw_data = prepare_input(user_input)

        # Generate prediction
        prediction = predict_price(raw_data)

        return {
            "predicted_price": round(prediction, 2)
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )