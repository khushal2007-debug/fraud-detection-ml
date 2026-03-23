from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Fraud Detection API")

model = joblib.load('fraud_model.pkl')

class Transaction(BaseModel):
    features: list[float]  # 30 values — Time, V1-V28, Amount

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running!"}

@app.post("/predict")
def predict(transaction: Transaction):
    features = np.array(transaction.features).reshape(1, -1)
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    return {
        "prediction": int(prediction),
        "result": "FRAUD" if prediction == 1 else "SAFE",
        "fraud_probability": round(float(probability[1]) * 100, 2),
        "safe_probability": round(float(probability[0]) * 100, 2)
    }