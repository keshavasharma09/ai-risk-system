import pickle
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = ROOT_DIR/"models"/"fraud_model_v1.pkl"
SCALER_PATH = ROOT_DIR / "models" / "scaler.pkl"

def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

def load_scaler():
    with open(SCALER_PATH, "rb") as f:
        return pickle.load(f)