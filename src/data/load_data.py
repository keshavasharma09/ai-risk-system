import pandas as pd
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)

# Define root once

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

def get_data_path(filename: str = "transactions.csv") -> Path:
    return ROOT_DIR / "data" / "raw" / filename


def load_data(filename: str = "transactions.csv") -> pd.DataFrame:
    data_path = get_data_path(filename)

    try:
        df = pd.read_csv(data_path)
        # print(f"Successfully loaded data: {df.shape}")
        logging.info(f"Loaded data from {data_path}")
        return df
    except Exception as e:
        raise RuntimeError(f"Error loading CSV at {data_path}: {e}")
    



