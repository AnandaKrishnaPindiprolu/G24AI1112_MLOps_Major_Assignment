import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from src.train import model

def test_model_instance():
    assert isinstance(model, LinearRegression)

def test_model_trained():
    assert hasattr(model, "coef_")

def test_r2_score():
    data = fetch_california_housing()
    X, y = data.data, data.target
    r2 = model.score(X, y)
    assert r2 > 0.5  # Example threshold