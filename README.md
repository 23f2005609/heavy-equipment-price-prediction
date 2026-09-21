# 🚜 Heavy Equipment Price Prediction

> **An end-to-end machine learning application for predicting the resale price of heavy equipment using XGBoost, feature engineering, FastAPI, and a production-ready web interface.**

🔗 **Live Demo:** https://heavy-equipment-price-prediction.onrender.com/
📦 **GitHub:** https://github.com/23f2005609/heavy-equipment-price-prediction

---

## 📌 Overview

Heavy equipment such as excavators, loaders, graders, and tractors can have significantly different resale values depending on their **age, operating hours, specifications, configuration, utilization, and market characteristics**.

The goal of this project was to build a machine learning system that takes these characteristics as input and provides an estimated resale price.

Rather than stopping at model training, the project was developed as a complete pipeline:

**Raw Data → Data Cleaning → Feature Engineering → Preprocessing → XGBoost → Evaluation → API → Web Interface → Deployment**

---

## 🎯 Problem Statement

The core problem was:

> **Can we predict the selling price of a heavy equipment machine from its available specifications and usage characteristics?**

The dataset contains historical equipment transactions with a mixture of:

* Numerical features
* Categorical features
* Equipment specifications
* Machine configuration
* Operational information
* Date information
* Missing values

The target variable is the equipment's **selling price**.

---

## 📊 Dataset

The training data contains approximately:

* **138,701 transactions**
* **50 original features**
* Numerical and categorical variables
* Significant missing values across several features

Important features included:

* `ManufactureYear`
* `OperationalHoursMeter`
* `Spec_FullDescriptor`
* `Spec_BaseClass`
* `Spec_SubClass`
* `VariantModifier`
* `ReleaseSeries`
* `FunctionalClassification`
* `RegionCode`
* `VendorPartnerID`
* `UtilizationTier`

---

# 🧠 How We Solved the Problem

## 1. Data Exploration

The first challenge was understanding the structure and quality of the data.

We investigated:

* Missing values
* Numerical distributions
* Categorical cardinality
* Target distribution
* Outliers and unusual values
* Equipment specifications
* Relationships between equipment characteristics and price

The target variable showed a right-skewed distribution, making direct regression more difficult.

---

## 2. Data Cleaning

The dataset contained substantial missing information.

We handled missing values differently depending on feature type:

### Numerical features

Missing numerical values were handled using appropriate imputation strategies.

### Categorical features

Missing categorical values were represented explicitly as:

```text
Missing
```

This allowed the model to learn whether missingness itself carried information.

---

## 3. Feature Engineering

A major part of the project was converting raw equipment information into features that a machine learning model could understand.

### Date Features

The transaction date was decomposed into:

```text
SaleYear
SaleMonth
SaleDay
SaleWeekday
SaleQuarter
SaleWeek
```

### Machine Age

Machine age was derived from the manufacturing year and sale year.

```text
MachineAge = SaleYear - ManufactureYear
```

### Specification Features

Equipment descriptors such as:

```text
140G
320CL
PC200LC6
```

were further analyzed to extract useful information such as:

* Model prefix
* Model number
* Model suffix
* Descriptor length
* Descriptor characteristics

### Indicator Features

Additional binary features were created to capture important specification patterns, including indicators such as:

```text
HasLC
HasXL
HasGP
HasBL
HasCL
HasELC
HasOperationalHours
HasVariantModifier
```

---

# 🔧 4. Categorical Encoding

The dataset contained many categorical variables with different cardinalities.

We used a combination of:

### One-Hot Encoding

For suitable categorical features:

```text
UtilizationTier
CabinType
DrivetrainType
InventoryGroup
```

### Frequency Encoding

For high-cardinality features such as:

```text
VendorPartnerID
RegionCode
Spec_FullDescriptor
Spec_BaseClass
Spec_SubClass
```

frequency encoding was used to avoid creating an excessively large feature matrix.

This was particularly important because the original dataset contained many unique equipment specifications.

---

# 📈 5. Target Transformation

The selling price distribution was skewed.

To make the regression problem more stable, the target was transformed using:

```python
np.log1p(y)
```

Predictions were converted back to the original price scale using:

```python
np.expm1(predictions)
```

This also aligns naturally with the use of **RMSLE** as the evaluation metric.

---

# 🤖 6. Machine Learning Model

After experimenting with the modeling pipeline, the final deployment version uses:

## XGBoost Regressor

XGBoost was selected because it performs well on structured/tabular data and can capture nonlinear relationships between equipment characteristics and selling price.

The deployment model was intentionally made smaller than the original experimentation models so that it could operate within the memory limitations of a free deployment environment.

### Evaluation Metric

The primary metric used was:

**RMSLE — Root Mean Squared Logarithmic Error**

A lower RMSLE indicates better relative prediction accuracy.

The smaller deployment model achieved approximately:

```text
Validation RMSLE ≈ 0.212
```

---

# 🧪 7. Validation & Testing

The model was tested using real rows from the training dataset.

Example prediction:

```text
Actual price    : ₹57,000
Predicted price : ₹57,850
Absolute error  : 1.49%
```

This confirmed that the complete preprocessing + feature engineering + XGBoost inference pipeline was working consistently.

---

# 🌐 8. From Model to Application

Instead of keeping the model inside a notebook, we converted it into an actual web application.

### Backend

Built using:

* **FastAPI**
* Python
* XGBoost
* Joblib

The API accepts equipment characteristics and returns a predicted selling price.

Example response:

```json
{
  "predicted_price": 39650.27
}
```

---

# 🖥️ 9. Web Interface

A responsive frontend was developed using:

* HTML
* CSS
* JavaScript

The interface allows users to enter equipment information such as:

* Manufacture year
* Operational hours
* Utilization tier
* Asset scale
* Functional classification
* Region
* Cabin type
* Forks
* Drivetrain
* Equipment specification

The application then sends the information to the FastAPI backend and displays the estimated selling price.

### UI Features

* Responsive design
* Dark / light mode
* Equipment input sections
* Loading state
* Prediction result card
* Error handling
* Searchable equipment fields
* Mobile-friendly layout

---

# 🐳 10. Deployment

The application was containerized using **Docker** and deployed on **Render**.

### Deployment Architecture

```text
                    User
                      │
                      ▼
              Web Interface
             HTML / CSS / JS
                      │
                      ▼
                FastAPI API
                      │
                      ▼
            Feature Engineering
                      │
                      ▼
               Preprocessor
                      │
                      ▼
             XGBoost Regressor
                      │
                      ▼
             Predicted Price
```

### Live Application

🚀 **Try the deployed application:**

`YOUR_RENDER_URL_HERE`

---

# 🏗️ Project Structure

```text
heavy-equipment-price-prediction/
│
├── app/
│   ├── main.py
│   └── static/
│       ├── index.html
│       ├── style.css
│       └── script.js
│
├── models/
│   ├── xgb_deploy_model.pkl
│   ├── preprocessor_deploy.pkl
│   └── feature_columns_deploy.pkl
│
├── src/
│   ├── feature_engineering.py
│   ├── input_adapter.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── train.py
│   └── deployment_defaults.py
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# ⚙️ Getting Started

## Clone the repository

```bash
git clone https://github.com/23f2005609/heavy-equipment-price-prediction.git

cd heavy-equipment-price-prediction
```

## Create an environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run the application

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

---

# 🛠️ Tech Stack

### Machine Learning

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost

### Backend

* FastAPI
* REST API
* Joblib

### Frontend

* HTML
* CSS
* JavaScript

### Deployment

* Docker
* Render
* Git
* GitHub
* Git LFS

---

# 💡 Key Challenges & Solutions

| Challenge                            | Solution                                                    |
| ------------------------------------ | ----------------------------------------------------------- |
| Large number of categorical features | One-hot + frequency encoding                                |
| High missingness                     | Explicit missing categories + numerical imputation          |
| Skewed price distribution            | `log1p` target transformation                               |
| Complex equipment descriptors        | Model/specification feature engineering                     |
| Feature mismatch during inference    | Saved deployment feature columns + consistent preprocessing |
| Large model size                     | Smaller deployment-focused XGBoost model                    |
| Free hosting memory limitations      | Reduced deployment artifact size                            |
| Local vs production differences      | Dockerized FastAPI application                              |
| Browser/application integration      | REST API + JavaScript frontend                              |

---

# 🚀 What I Learned

This project helped me understand that building a machine learning solution is much more than training a model.

The major learning areas were:

* Working with large tabular datasets
* Handling missing and categorical data
* Feature engineering
* Regression modeling
* Log transformations
* RMSLE evaluation
* Model serialization
* Building inference pipelines
* REST API development
* Frontend ↔ ML model integration
* Docker containerization
* Cloud deployment
* Debugging production-specific issues
* Optimizing ML models for limited deployment resources

---

# 🔮 Future Improvements

Potential improvements include:

* Automated model retraining
* Experiment tracking
* Better model monitoring
* Prediction confidence / price ranges
* Feature importance visualization
* More advanced hyperparameter optimization
* Model versioning
* CI/CD automation
* Larger deployment infrastructure for more complex ensemble models

---

## 👨‍💻 Author

**Barun Sharma**

IIT Madras — B.S. Data Science and Programming

Interested in:

**Machine Learning • Data Science • AI • Python • Data Analytics**

### Connect

* GitHub: https://github.com/23f2005609
* LinkedIn: https://www.linkedin.com/in/barun-sharma1508/

---

## ⭐ Project Highlights

**138K+ transactions**
**50 original features**
**XGBoost Regression**
**~0.212 validation RMSLE**
**FastAPI REST API**
**Dockerized Application**
**Live Render Deployment**

> **From raw equipment data to a deployed machine learning application. 🚜🤖**
