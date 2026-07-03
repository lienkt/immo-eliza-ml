import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
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
    y = df["price"]

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
    if model_type == "linear":

        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

    elif model_type in ["rf", "xgb"]:

        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
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