# G24AI1112_MLOps_Major_Assignment


## Repository Structure
## Top-Level Contents:

## 📁 Repository Structure

- `.github/` – GitHub workflows and configuration files
- `.gitignore` – Specifies intentionally untracked files to ignore
- `Dockerfile` – Instructions to build the Docker image for the project
- `README.md` – Project overview and documentation
- `requirements.txt` – Python dependencies
- `src/` – Source code for the application
- `tests/` – Unit and integration test scripts
- `venv/` – Python virtual environment (optional for local setup)

  ## 📂 Key Files and Directories in `src/`

- `__pycache__/` – Compiled Python bytecode cache for faster execution
- `model.joblib` – Serialized machine learning model
- `predict.py` – Script for generating predictions using the trained model
- `quant_params.joblib` – Stored parameters for model quantization
- `quantize.py` – Applies quantization techniques to optimize the model
- `train.py` – Trains and saves the machine learning model
- `unquant_params.joblib` – Parameters before quantization (baseline model state)

- ## 📘 Purpose of Each Component

- `README.md` – Contains project overview, usage instructions, and setup notes
- `Dockerfile` – Defines the containerized environment for reproducible builds and deployment
- `requirements.txt` – Lists Python dependencies required for running the application
- `src/` – Holds the core project code:
  - Training, quantization, and prediction scripts
  - Serialized models and parameter files
- `tests/` – Contains unit or integration tests to validate functionality and ensure code reliability *(contents not listed yet)*
- `venv/` – Local Python virtual environment (typically excluded from version control in production)

  ## Code Analysis
  
  ## Code

## Train.py
from sklearn.datasets import fetch_california_housing

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

import joblib


data = fetch_california_housing()
X, y = data.data, data.target

model = LinearRegression()
model.fit(X, y)

r2 = r2_score(y, model.predict(X))

mse = mean_squared_error(y, model.predict(X))

print(f"R2 Score: {r2}, MSE: {mse}")

joblib.dump(model, "model.joblib")

## 📊 Analysis Summary

- Loads the California Housing dataset using `fetch_california_housing`
- Trains a Linear Regression model on the dataset
- Evaluates model performance using:
  - R² Score
  - Mean Squared Error (MSE)
- Prints the evaluation metrics to console
- Saves the trained model as `model.joblib` using `joblib.dump`

  ## Expected Output Example:
  R2 Score: 0.606, MSE: 0.53

## Quantize.py
from sklearn.datasets import fetch_california_housing

import joblib

import numpy as np

model = joblib.load("model.joblib")

raw_params = {"coef": model.coef_, "intercept": model.intercept_}

joblib.dump(raw_params, "unquant_params.joblib")


coef_q = np.array(model.coef_ * 100, dtype=np.uint8)

intercept_q = np.array([model.intercept_ * 100], dtype=np.uint8)

quantized = {"coef": coef_q, "intercept": intercept_q}

joblib.dump(quantized, "quant_params.joblib")


# De-quantize

coef_deq = coef_q / 100

intercept_deq = intercept_q[0] / 100

prediction = np.dot(fetch_california_housing().data, coef_deq) + intercept_deq

print("Sample Prediction:", prediction[:5])

## 📊 Quantization & Inference Analysis

- Loads the trained model (`model.joblib`)
- Extracts and saves raw model parameters for reference
- Applies quantization:
  - Scales parameters
  - Converts to `uint8` format
- Saves quantized parameters (`quant_params.joblib`)
- Performs dequantization for inference
- Uses dequantized weights to make predictions on the full dataset
- Prints the first 5 predictions to console

  ## Expected Output Example:

  Sample Prediction: [4.66 3.61 3.84 3.34 3.51]

  ## Predict.py

import joblib

from sklearn.datasets import fetch_california_housing

from sklearn.metrics import mean_squared_error, r2_score


data = fetch_california_housing()

X, y = data.data, data.target


model = joblib.load("src/model.joblib")

preds = model.predict(X)


print(f"Sample Predictions: {preds[:5]}")

print(f"R2 Score: {r2_score(y, preds)}, MSE: {mean_squared_error(y, preds)}")


## 📊 Prediction Analysis

- Loads the California Housing dataset using `fetch_california_housing`
- Loads the previously trained model from `model.joblib`
- Generates predictions on the full dataset
- Prints the first 5 predictions to console
- Evaluates and displays performance metrics (e.g., R² Score, Mean Squared Error)

    ## Expected Output Example:
  
Sample Predictions: [4.67 3.61 3.86 3.36 3.52]

R2 Score: 0.606, MSE: 0.53

## 📝 Script Summary

- `train.py`
  - Trains a regression model on the California Housing dataset
  - Evaluates performance using R² Score and MSE
  - Saves the trained model as `model.joblib`

- `quantize.py`
  - Loads the trained model
  - Saves original (raw) model parameters for reference
  - Applies quantization by scaling and casting to `uint8`
  - Saves quantized parameters
  - Dequantizes parameters and makes predictions
  - Prints first 5 dequantized predictions

- `predict.py`
  - Loads the trained model
  - Makes predictions on the full dataset
  - Prints first 5 predictions
  - Displays evaluation metrics (R² Score and MSE)

- **Note**: Output values will vary based on training data and model configuration, but the script structure and usage remain consistent

  
