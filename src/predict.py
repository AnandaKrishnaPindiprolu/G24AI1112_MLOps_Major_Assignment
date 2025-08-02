import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score

data = fetch_california_housing()
X, y = data.data, data.target

model = joblib.load("src/model.joblib")
preds = model.predict(X)

print(f"Sample Predictions: {preds[:5]}")
print(f"R2 Score: {r2_score(y, preds)}, MSE: {mean_squared_error(y, preds)}")