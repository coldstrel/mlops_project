import mlflow
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize DagsHub and set up MLflow experiment tracking
dagshub.init(repo_owner='coldstrel', repo_name='mlops_project', mlflow=True)
mlflow.set_experiment("Experiment2") # Name of the experiment in MLflow
mlflow.set_tracking_uri("https://dagshub.com/coldstrel/mlops_project.mlflow")  # URL to track the experiment

## Load data
data = pd.read_csv("./pcos_data.csv")

# Split data into features and target
X = data.drop(columns=['PCOS_Diagnosis'], axis=1)
y = data['PCOS_Diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Apply SMOTE to balance minority class
smote = SMOTE(random_state = 42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Scale the features 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

MAX_DEPTH = 3
N_ESTIMATORS = 100

## Start a new MLFlow run
with mlflow.start_run(run_name='Random Forest exp1') as parent:
    rf_model = RandomForestClassifier(max_depth=MAX_DEPTH, n_estimators=N_ESTIMATORS, random_state = 42)
    rf_model.fit(X_train, y_train)
    # Save the trained model as a pickle file
    pickle.dump(rf_model, open('random_forest.pkl', 'wb'))
    # Load saved model for predictions
    model = pickle.load(open('random_forest.pkl', 'rb'))
    y_pred = model.predict(X_test)
    # Calculate the metrics
    acc = accuracy_score(y_test, y_pred)  # Accuracy
    precision = precision_score(y_test, y_pred)  # Precision
    recall = recall_score(y_test, y_pred)  # Recall
    f1 = f1_score(y_test, y_pred)  # F1-score
    
    # Log the model and metrics to MLflow
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    
    # Log parameter
    mlflow.log_param("max_depth", MAX_DEPTH)
    mlflow.log_param("n_estimators", N_ESTIMATORS)
    
    # Create a confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("./reports/figures/experiments/confusion_matrix.png")
    # Log confusion matrix
    mlflow.log_artifact("./reports/figures/experiments/confusion_matrix.png")
    # Log model to mlflow
    mlflow.sklearn.log_model(rf_model, "random_forest_model")
    # Log the source code
    mlflow.log_artifact(__file__)
    
    # Set tags
    mlflow.set_tag("model_type", "Random Forest")
    mlflow.set_tag("dataset", "PCOS Dataset")
    mlflow.set_tag("experiment", "Experiment2")
    
    
    