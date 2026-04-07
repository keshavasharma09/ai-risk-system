from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, \
classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pickle

from src.data.load_data import load_data

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = ROOT_DIR / "models" / "fraud_model_v1.pkl"
SCALER_PATH = ROOT_DIR / "models" / "scaler.pkl"


def train():
    # 1. Load data
    df = load_data("transactions.csv")
    df.drop(columns = "Time", inplace = True)

    # 2. Split features & target
    X = df.iloc[:, :-1]
    y = df["Class"]

    # 3. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=2
    )

    # 4. Scaling
    scaler = StandardScaler()
    X_train["Amount"] = scaler.fit_transform(X_train[["Amount"]])
    X_test["Amount"] = scaler.transform(X_test[["Amount"]])  # FIXED

    # 5. Handle imbalance
    smote = SMOTE(random_state=2)
    X_smote, y_smote = smote.fit_resample(X_train, y_train)

    print(f"Original distribution: {Counter(y_train)}")
    print(f"SMOTE distribution: {Counter(y_smote)}")

    # 6. Train model
    model = RandomForestClassifier(criterion="entropy")
    model.fit(X_smote, y_smote)

    # 7. Evaluation
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # 8. Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    # 9. Save scaler
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print("Model and scaler saved successfully!")


if __name__ == "__main__":
    train()