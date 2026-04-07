import mlflow
import mlflow.sklearn
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

    # 1. Force the script to send data directly to the active MLflow UI
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    mlflow.set_experiment("Credit_Card_Fraud_Detection")

    with mlflow.start_run():
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

        # # 5. Handle imbalance
        # smote = SMOTE(random_state=2)
        # X_smote, y_smote = smote.fit_resample(X_train, y_train)

        # print(f"Original distribution: {Counter(y_train)}")
        # print(f"SMOTE distribution: {Counter(y_smote)}")

        # 6. Train model
        criterion = "gini"
        n_estimators = 50
        max_depth = 4
        model = RandomForestClassifier(criterion=criterion, n_estimators=n_estimators, max_depth=max_depth)
        print("Modelling started")
        # model.fit(X_smote, y_smote)
        model.fit(X_train, y_train)

        # 7. Evaluation
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Recall:", recall_score(y_test, y_pred))
        print("Precision:", precision_score(y_test, y_pred))
        print("F1:", f1_score(y_test, y_pred))
        print("Report:\n", classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

        # 3. Log your parameters (inputs) in ML Flow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("criterion", criterion)
        
        # Loggig the metrics in ML Flow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("f1_score", f1)

        signature = mlflow.models.signature.infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(model, "fraud_model", signature=signature)
        
        print(f"Run complete! Accuracy: {accuracy}")
        print(f"Run complete! Recall: {recall}")
        print(f"Run complete! Precision: {precision}")
        print(f"Run complete! F1 Score: {f1}")

        # # 8. Save model
        # with open(MODEL_PATH, "wb") as f:
        #     pickle.dump(model, f)

        # 9. Save scaler
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)
        print("Scaler pickle file saved.")

        # print("Model and scaler saved successfully!")


if __name__ == "__main__":
    train()