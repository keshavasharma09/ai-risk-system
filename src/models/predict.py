from src.models.load_model import load_model, load_scaler
import pandas as pd

# model = load_model()
# scaler = load_scaler()

# def preprocess_inference(df, scaler):
#     df["Amount"] = scaler.transform(df[["Amount"]])
#     return df

import mlflow.sklearn  # Changed from pyfunc to sklearn!
import pandas as pd
import pickle
import os
# 1. Check if we are running in a CI/CD test environment
IS_TESTING = os.getenv("TESTING") == "True"

if not IS_TESTING:
    # We are in production/local! Load the real models.
    mlflow.set_tracking_uri("http://host.docker.internal:5000")
    RUN_ID = "a85bbb411e0048abb1e842ec4e309fe7"
    print(f"Loading MLflow model from Run ID: {RUN_ID}...")
    model = mlflow.sklearn.load_model(f"runs:/{RUN_ID}/fraud_model")
    
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
else:
    # We are in GitHub Actions! Mock the models so the app doesn't crash.
    print("Running in Test Mode. Skipping MLflow connection.")
    model = None
    scaler = None
# # 1. Point the API to your local MLflow tracking server
# mlflow.set_tracking_uri("http://host.docker.internal:5000")

# # 2. Your Run ID
# RUN_ID = "a85bbb411e0048abb1e842ec4e309fe7"

# print(f"Loading MLflow model from Run ID: {RUN_ID}...")

# # 3. Load the model dynamically using the SKLEARN flavor.
# # This gives us access to .predict_proba() and .feature_names_in_
# model = mlflow.sklearn.load_model(f"runs:/{RUN_ID}/fraud_model")

# # 4. Load your scaler to transform the incoming API data
# with open("models/scaler.pkl", "rb") as f:
#     scaler = pickle.load(f)


def decide(prediction, probability):
    """Business logic for routing transactions based on risk"""
    if prediction == 1 and probability > 0.8:
        return "Block Transaction"
    elif probability > 0.5:
        return "Manual Review"
    return "Allow"


def predict(data: dict):
    # 1. Convert incoming API JSON to DataFrame
    df = pd.DataFrame([data])

    # 2. Scale the 'Amount' column
    # (Double brackets [["Amount"]] ensures it remains a 2D array for the scaler)
    df["Amount"] = scaler.transform(df[["Amount"]])

    # 3. Force the DataFrame to use the model's exact expected column order
    expected_columns = model.feature_names_in_
    df = df[expected_columns]

    # 4. Make predictions
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]  # Get probability of class 1 (Fraud)

    # 5. Apply business logic
    action = decide(prediction, probability)

    # 6. Return the final payload to the FastAPI endpoint
    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "action": action
    }


# import mlflow.pyfunc
# import pandas as pd
# import pickle

# # 1. Point the API to your local MLflow tracking server
# mlflow.set_tracking_uri("http://127.0.0.1:5000")

# # 2. Paste the Run ID you copied from the MLflow UI
# # Example: RUN_ID = "7570b333c3cb472fafe95969949d0d0e"
# RUN_ID = "a85bbb411e0048abb1e842ec4e309fe7"

# print(f"Loading MLflow model from Run ID: {RUN_ID}...")
# # 3. Load the model dynamically. 
# # "fraud_model" is the artifact name you defined in your training script.
# model = mlflow.pyfunc.load_model(f"runs:/{RUN_ID}/fraud_model")

# # 4. Load your scaler to transform the incoming API data
# # (Assuming you uncommented the scaler save code in your training script!)
# with open("models/scaler.pkl", "rb") as f:
#     scaler = pickle.load(f)

# def predict(data: dict):
#     # Convert incoming API JSON to DataFrame
#     df = pd.DataFrame([data])

#     # df.pop("Class", None)  # optional safety
#     # 2. Force the DataFrame to use the model's exact expected column order
#     # This prevents the ValueError you are seeing!
#     expected_columns = model.feature_names_in_
#     df = df[expected_columns]

#     # df = preprocess_inference(df, scaler)

#     prediction = model.predict(df)[0]
#     probability = model.predict_proba(df)[0][1]



#     return {
#         "prediction": int(prediction),
#         "probability": float(probability)
#     }
# prediction, probability = predict()

# def decide(prediction, probability):
#     if prediction == 1 and probability > 0.8:
#         return "Block Transaction"
#     elif probability > 0.5:
#         return "Manual Review"
#     return "Allow"

# decide(prediction, probability)