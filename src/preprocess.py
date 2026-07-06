import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler

def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """ 
    Separate features and target.
    Parameters:
        df (pd.DataFrame): The input DataFrame to be preprocessed.
    Returns:
        X (pd.DataFrame): The feature matrix after preprocessing.
        y (pd.Series): The target variable.
    """

    X = df.drop("price", axis=1)
    # Transform price to log scale to reduce skewness
    y = df["price"].apply(lambda x: np.log10(x))  # log(1 + price) to handle zero prices
    
    return X, y

def build_preprocessor(X: pd.DataFrame, model_type: str) -> ColumnTransformer:
    """
    Build preprocessing pipeline based on model type.
    Parameters:
        df (pd.DataFrame): The input DataFrame to be preprocessed.
        model_type (str): The type of model to be trained. Options: "linear", "rf", "xgb".  
    Returns:
        preprocessor (ColumnTransformer): The preprocessing pipeline for features.
    """

    num_features = X.select_dtypes(include=["int64", "float64"]).columns
    cat_features = X.select_dtypes(include=["object", "category"]).columns

    # Numeric pipeline
    if model_type in ["linear", "rf", "xgb"]:

        numeric_transformer = Pipeline([
            ("scaler", StandardScaler()),
            # ("imputer", SimpleImputer(strategy="median"))
            ("imputer", KNNImputer(n_neighbors=5, weights="distance"))
        ])

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Categorical pipeline
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features)
    ])

    return preprocessor