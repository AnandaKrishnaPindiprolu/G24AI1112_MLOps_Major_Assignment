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