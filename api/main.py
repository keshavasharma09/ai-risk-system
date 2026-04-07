from fastapi import FastAPI
from src.models.predict import predict
from src.services.decision_engine import decide
from api.schemas import Transaction


app = FastAPI()

@app.get("/")
def home():
    return {"message": "Fraud Detection API Running"}

@app.post("/predict")
def get_prediction(data: Transaction):
    result = predict(data.dict())   # or model_dump() if using Pydantic v2
    
    action = decide(result["prediction"], result["probability"])

    return {
        **result,
        "action": action
    }