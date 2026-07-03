import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def preprocess(df: pd.DataFrame, model_type: str | None) -> tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """ 
    Separate features/target and create preprocessing pipeline.
    Parameters:
        df (pd.DataFrame): The input DataFrame to be preprocessed.
        model_type (str): The type of model to be trained. Options: "linear", "rf", "xgb".  
    Returns:
        X (pd.DataFrame): The feature matrix after preprocessing.
        y (pd.Series): The target variable.
        preprocessor (ColumnTransformer): The preprocessing pipeline for features.
    """

    X = df.drop("price", axis=1)
    y = df["price"]

    num_features = X.select_dtypes(include=["int64", "float64"]).columns

    cat_features = X.select_dtypes(include=["object", "category"]).columns

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features)
    ])

    return X, y, preprocessor