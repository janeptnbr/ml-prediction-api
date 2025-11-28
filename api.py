# api.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(
    title="Modelo de Regressão - Exemplo",
    description="API REST para servir um modelo de ML",
    version="1.0.0"
)

# Carrega o modelo treinado
model = joblib.load("model.joblib")

# Define o formato dos dados de entrada
class InputData(BaseModel):
    # Aqui estou usando um vetor genérico; você pode dar nomes mais semânticos
    features: list[float]

@app.get("/")
def read_root():
    return {"message": "API de previsão está no ar."}

@app.post("/predict")
def predict(data: InputData):
    # Converte lista para array 2D (n amostras x n features)
    X = np.array(data.features).reshape(1, -1)
    prediction = model.predict(X)[0]
    return {
        "prediction": float(prediction)
    }
