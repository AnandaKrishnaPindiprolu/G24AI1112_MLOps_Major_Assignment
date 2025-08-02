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