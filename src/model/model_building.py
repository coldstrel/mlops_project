import pandas as pd
import yaml
import pickle
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

from imblearn.pipeline import Pipeline


def load_params(filepath: str) ->int:
    """Load parameter from a YAML file.

    Args:
        filepath (str): Path to YAML file.

    Returns:
        int: Parameters.
    """
    try:
        with open(filepath, 'r') as file:
            params = yaml.safe_load(file)
            return params["model_building"]["n_estimators"]
    except Exception as e:
        raise Exception(f"Error loading parameters: {filepath}, {e}")
    
def load_data(filepath: str) -> pd.DataFrame:
    """Loads a CSV file and returns a DataFrame.

    Args:
        filepath (str): Path for the CSV file.

    Returns:
        pd.DataFrame: CSV file converted into a pandas DataFrame.
    """
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Unable to load CSV file {filepath}, {e}")
    

def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separates the data into features and target.

    Args:
        data (pd.DataFrame): pandas DataFrame

    Returns:
        tuple[pd.DataFrame, pd.Series]: Features and Target values.
    """
    try:
        X = data.drop(columns=['PCOS_Diagnosis'], axis=1)
        y = data["PCOS_Diagnosis"]
        print(X.shape, y.shape)
        return X, y
    except Exception as e:
        raise Exception(f"Unable to separate data {e}")
    
def train_model(X: pd.DataFrame, y: pd.Series, n_estimators: int) -> RandomForestClassifier:
    """Trains the RandomForestClassifier model.

    Args:
        X (pd.DataFrame): A DataFrame for features.
        y (pd.Series): A pandas Series containing the independent variable.
        n_estimators (int): Parameter for the RFC.

    Returns:
        RandomForestClassifier: A model.
    """
    try:
        pipe = Pipeline([
            ("smote", SMOTE(random_state = 42)),
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=n_estimators, random_state = 42))
        ])
        pipe.fit(X, y)
        return pipe
    except Exception as e:
        raise Exception(f"Not able to generate the model {e}")
    
def save_model(model: RandomForestClassifier, model_name: str) -> None:
    """Saves the model as a Pickle file.

    Args:
        model (RandomForestClassifier)
        model_name (str)
    """
    try: 
        with open(model_name, "wb") as file:
            pickle.dump(model, file)
    except Exception as e:
        raise Exception(f"Unable to save model {e}")

def main():
    
    
    try:
        params_path = "./params.yaml"
        data_path = "./data/raw/train.csv"
        model_name = "models/model.pkl"
        n_estimators = load_params(params_path)
        train_data = load_data(data_path)
        X_train, y_train = prepare_data(train_data)
        
        model = train_model(X_train, y_train, n_estimators)
        save_model(model, model_name)
        print("Model saved successfully!!")
        
    except Exception as e:
        raise Exception(f"Error {e}")
    
    
if __name__ == "__main__":
    main()
    