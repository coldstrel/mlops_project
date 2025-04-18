import pandas as pd
import os 
import numpy as np
from sklearn.model_selection import train_test_split
import yaml

def load_params(filepath: str) -> float:
    """ 
    Load parameters from a YAML file.
    Args:
        filepath (str): Path to the YAML file.
    Returns:
        float: Value of the parameter.
    """
    try:
        with open(filepath, 'r') as file:
            params = yaml.safe_load(file)
        return params["data_collection"]["test_size"]
    except Exception as e:
        raise Exception(f"Error loading parameters: {e}")

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads data from a CSV file
    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data: {filepath}, {e}")

def split_data(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits the data into training and testing sets.

    Args:
        df (pd.DataFrame): DataFrame containing data to split.
        test_size (float): Size of the test given as a fraction.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Tuple containing the training and testing DataFrames.
    """
    try:
        return train_test_split(df, test_size=test_size, random_state=42)
    except Exception as e:
        raise Exception(f"Error splitting data: {e}")


def save_data(df: pd.DataFrame, filepath: str) -> None:
    """Save the data to a desired location as a CSV.

    Args:
        df (pd.DataFrame): _description_
        filepath (str): _description_

    Raises:
        Exception: _description_
    """
    try:
        df.to_csv(filepath, index=False)
    except Exception as e:
        raise Exception(f"Error saving data: {filepath}, {e}")
    
def main():
    data_filepath = "./pcos_data.csv"
    params_filepath = "./params.yaml"
    raw_datapath = "./data/raw"
    
    try:
        data = load_data(data_filepath)
        test_size = load_params(params_filepath)
        train_data, test_data = split_data(data, test_size)
        raw_data_path = os.path.join("data","raw")
        os.makedirs(raw_data_path)
        save_data(train_data, os.path.join(raw_datapath, "train.csv"))
        save_data(test_data, os.path.join(raw_datapath, "test.csv"))
        
    except Exception as e:
        raise Exception(f"Error {e}")
    
if __name__ == "__main__":
    main()