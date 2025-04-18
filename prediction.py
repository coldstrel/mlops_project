import mlflow
import pandas as pd
import json

# Set the tracking URI to your DagsHub MLflow instance
mlflow.set_tracking_uri("https://dagshub.com/coldstrel/mlops_project.mlflow")  # URL to track the experiment

# Specify the model name
reports_path = "reports/run_info.json"
with open(reports_path, 'r') as file:
    run_info = json.load(file)
    
run_id = run_info['run_id'] # Fetch run id from the JSON file
model_name = run_info['model_name']  # Fetch model name from the JSON file

try:
    # Create an MlflowClient to interact with the MLflow server
    client = mlflow.tracking.MlflowClient()

    # Construct the logged_model string
    logged_model = f'runs:/{run_id}/{model_name}'
    print("Logged Model:", logged_model)

    # Load the model using the logged_model variable
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    print(f"Model loaded from {logged_model}")

    # Input data for prediction
    data = pd.DataFrame({
    'Age': [18, 20, 19, 23, 20, 33],
    'BMI': [26.5, 16.5, 23.1, 22.3, 14.9, 23.9],
    'Menstrual_Irregularity': [0, 0, 0, 0, 0, 1],
    'Testosterone_Level(ng/dL)': [40.2, 43.3, 31.9, 47.8, 50.8, 101.9],
    'Antral_Follicle_Count': [10, 5, 10, 4, 6, 36]
    # 'PCOS_Diagnosis': [0, 0, 0, 0, 0, 1]
    
    })

    # Make prediction
    prediction = loaded_model.predict(data)
    print("Prediction:", prediction)


except Exception as e:
    print(f"Error fetching model: {e}")