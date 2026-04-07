from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_PATH = BASE_DIR/"data"/ "raw"/ "transactions.csv"
PROCESSED_PATH = BASE_DIR / "data"/ "processsed"/ "cleaned.csv"
MODEL_PATH = BASE_DIR/"models"/"fraud_model.pkl"