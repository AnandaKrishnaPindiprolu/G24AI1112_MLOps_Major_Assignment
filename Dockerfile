FROM python:3.10-slim
WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY src/ src/
#COPY src/model.joblib /model.joblib
COPY src/model.joblib /app/src/model.joblib
CMD ["python", "src/predict.py"]