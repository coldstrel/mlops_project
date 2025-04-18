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
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from tqdm import tqdm


# Initialize DagsHub and set up MLflow experiment tracking
dagshub.init(repo_owner='coldstrel', repo_name='mlops_project', mlflow=True)
mlflow.set_experiment("Expriment3") # Name of the experiment in MLflow
mlflow.set_tracking_uri("https://dagshub.com/coldstrel/mlops_project.mlflow")  # URL to track the experiment

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

# Define multiple baseline models to compare performance
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Classifier": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "XG Boost": XGBClassifier()
}

# Start a new MLFlow run
with mlflow.start_run(run_name="Multiple models"):
    for model_name, model in tqdm(models.items(), total=len(models), desc="Training models", unit="model"):
        with mlflow.start_run(run_name=model_name, nested=True):
            model.fit(X_train, y_train)
            # Save the trained model 
            model_filename = f"{model_name.replace(' ', '_').lower()}.pkl"
            pickle.dump(model, open(model_filename, 'wb'))
            # Make predictions
            y_pred = model.predict(X_test)
            # Calculate the metrics
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            # Log the model metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            # Generate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f"Confusion Matrix for {model_name}")
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(f"./reports/figures/experiments/{model_name.replace(' ', '_').lower()}_confusion_matrix.png")
            # Log the confusion matrix
            mlflow.log_artifact(f"./reports/figures/experiments/{model_name.replace(' ', '_').lower()}_confusion_matrix.png")
            # log the model to mlflow
            mlflow.sklearn.log_model(model, model_name.replace(' ', '_').lower())
            # Log the source code
            mlflow.log_artifact(__file__)
            # Set tags
            mlflow.set_tag("AldoF", model_name)
        
    print("Finished")
        