# Heavy Equipment Resale Price Prediction

## Problem Statement

Predict the selling price of heavy equipment using machine
specifications, age, usage and categorical information.

## Dataset

138,701 training records
15,000 test records

## Feature Engineering

- MachineAge
- ModelNumber
- CapacityValue
- Descriptor features
- Binary specification indicators
- Interaction features

## Models

- Random Forest
- LightGBM
- CatBoost
- XGBoost

## Final Model

70:30 XGBoost + CatBoost Ensemble

## Evaluation

Metric: RMSLE

Kaggle Public RMSLE: 0.1913

## Deployment

FastAPI + Docker

## Architecture

Input
→ Feature Engineering
→ Preprocessing
→ XGBoost/CatBoost
→ Ensemble
→ Prediction