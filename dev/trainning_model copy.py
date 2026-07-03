import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from xgboost import XGBRegressor

from preprocess import preprocess
from src.evaluate_model import evaluate_model

def _train_model(name, model, X_train, X_test, y_train, y_test, preprocessor, base_dir, filename):

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    metrics = evaluate_model(
        name,
        pipeline,
        X_train, X_test,
        y_train, y_test
    )

    # ----------------------
    # SAVE MODEL
    # ----------------------
    os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)

    joblib.dump(pipeline, os.path.join(base_dir, "models", filename))

    print(f"Model saved: {filename}")

    return pipeline, metrics

def train_linear(X_train, X_test, y_train, y_test, preprocessor, base_dir):

    # 1. Build pipeline (preprocessing + model)
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    # 2. Train model
    model.fit(X_train, y_train)

    # 3. Evaluate on TRAIN set
    train_score = model.score(X_train, y_train)
    print("Train R2:", train_score)

    # 4. Predict on TEST set
    y_pred = model.predict(X_test)

    print("\n[LINEAR REGRESSION]")
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)

    test_score = r2_score(y_test, y_pred)
    print("R2 :", test_score)

    # ----------------------
    # Over/Under fitting check
    # ----------------------
    if train_score < 0.5:
        print("⚠️ Underfitting (model too weak)")

    elif train_score - test_score > 0.1:
        print("⚠️ Overfitting (train >> test)")

    else:
        print("✅ Good fit")

    # 5. Save ONLY model (pipeline already includes scaler + encoder)
    model_path = os.path.join(base_dir, "models/linear_model.pkl")
    joblib.dump(model, model_path)

    print(f"Model saved at: {model_path}")

    return model

def train_random_forest(X_train, X_test, y_train, y_test, preprocessor, base_dir):
    
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    # ----------------------
    # TRAIN SCORE
    # ----------------------
    train_score = model.score(X_train, y_train)
    print("\n[RANDOM FOREST]")
    print("Train R2:", train_score)

    # ----------------------
    # TEST SCORE
    # ----------------------
    y_pred = model.predict(X_test)

    test_score = r2_score(y_test, y_pred)
    print("Test R2:", test_score)

    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)

    # ----------------------
    # OVER/UNDER FIT CHECK
    # ----------------------
    if train_score < 0.5:
        print("⚠️ Underfitting")

    elif train_score - test_score > 0.1:
        print("⚠️ Overfitting")

    else:
        print("✅ Good fit")

    # ----------------------
    # SAVE MODEL
    # ----------------------
    os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)

    joblib.dump(model, os.path.join(base_dir, "models", "rf_model.pkl"))

    return model

def train_xgboost(X_train, X_test, y_train, y_test, preprocessor, base_dir):

    from xgboost import XGBRegressor

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBRegressor(
            random_state=42,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6
        ))
    ])

    model.fit(X_train, y_train)

    # ----------------------
    # TRAIN SCORE
    # ----------------------
    train_score = model.score(X_train, y_train)
    print("\n[XGBOOST]")
    print("Train R2:", train_score)

    # ----------------------
    # TEST SCORE
    # ----------------------
    y_pred = model.predict(X_test)

    test_score = r2_score(y_test, y_pred)
    print("Test R2:", test_score)

    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)

    # ----------------------
    # OVER/UNDER FIT CHECK
    # ----------------------
    if train_score < 0.5:
        print("⚠️ Underfitting")

    elif train_score - test_score > 0.1:
        print("⚠️ Overfitting")

    else:
        print("✅ Good fit")

    # ----------------------
    # SAVE MODEL
    # ----------------------
    os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)

    joblib.dump(model, os.path.join(base_dir, "models", "xgb_model.pkl"))

    return model

def trainning_model(df: pd.DataFrame, base_dir: str, model_type: str):
    """
    Trains a machine learning model based on the specified model type.
    Parameters:
        df (pd.DataFrame): The input DataFrame containing features and target variable.
        base_dir (str): The base directory for saving the trained model.
        model_type (str): The type of model to train. Options: "linear", "rf", "xgb".   
    Returns:
        The trained model object.
    """

    # ---------------------------------------
    # preprocess
    # ---------------------------------------
    X, y, preprocessor = preprocess(df)


    # ---------------------------------------
    # split
    # ---------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # ---------------------------------------
    # route model
    # ---------------------------------------
    if model_type == "linear":
        return train_linear(X_train, X_test, y_train, y_test, preprocessor, base_dir)

    elif model_type == "rf":
        return train_random_forest(X_train, X_test, y_train, y_test, preprocessor, base_dir)

    elif model_type == "xgb":
        return train_xgboost(X_train, X_test, y_train, y_test, preprocessor, base_dir)

    else:
        raise ValueError("Unknown model type")