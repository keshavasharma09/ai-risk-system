from src.models.load_model import load_model, load_scaler
import pandas as pd

model = load_model()
scaler = load_scaler()

def preprocess_inference(df, scaler):
    df["Amount"] = scaler.transform(df[["Amount"]])
    return df

def predict(data: dict):
    df = pd.DataFrame([data])

    df.pop("Class", None)

    df = preprocess_inference(df, scaler)

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }