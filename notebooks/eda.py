import sys
from pathlib import Path

# Going up from notebooks folder to root folder
ROOT_DIR = Path().resolve().parent
sys.path.append(str(ROOT_DIR))


from src.data.load_data import load_data

df = load_data("transactions.csv")
prediction_1 = df.loc[0].to_dict()
print(prediction_1)
prediction_1.pop("Class", None)
print(prediction_1)

