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
from src.model_comparison import compare_models

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

    return _train_model(
        "Linear Regression",
        LinearRegression(),
        X_train, X_test, y_train, y_test,
        preprocessor,
        base_dir,
        "linear_model.pkl"
    )

def train_random_forest(X_train, X_test, y_train, y_test, preprocessor, base_dir):

    return _train_model(
        "Random Forest",
        RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ),
        X_train, X_test, y_train, y_test,
        preprocessor,
        base_dir,
        "rf_model.pkl"
    )

def train_xgboost(X_train, X_test, y_train, y_test, preprocessor, base_dir):

    return _train_model(
        "XGBoost",
        XGBRegressor(
            random_state=42,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6
        ),
        X_train, X_test, y_train, y_test,
        preprocessor,
        base_dir,
        "xgb_model.pkl"
    )

def train_all_models(df, base_dir, preprocessor, X_train, X_test, y_train, y_test):

    all_metrics = []

    # =====================================================
    # LINEAR
    # =====================================================
    model1, m1 = _train_model(
        name="Linear Regression",
        model=LinearRegression(),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
        base_dir=base_dir,
        filename="linear_model.pkl"
    )

    all_metrics.append(m1)

    # =====================================================
    # RANDOM FOREST
    # =====================================================
    model2, m2 = _train_model(
        name="Random Forest",
        model=RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
        base_dir=base_dir,
        filename="rf_model.pkl"
    )

    all_metrics.append(m2)

    # =====================================================
    # XGBOOST
    # =====================================================
    model3, m3 = _train_model(
        name="XGBoost",
        model=XGBRegressor(
            random_state=42,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6
        ),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
        base_dir=base_dir,
        filename="xgb_model.pkl"
    )

    all_metrics.append(m3)

    # =====================================================
    # COMPARE ALL MODELS (OPTION NÀY)
    # =====================================================
    best = compare_models(all_metrics)

    return best

def trainning_model(df: pd.DataFrame, base_dir: str, model_type: str):

    # -----------------------
    # PREPROCESS
    # -----------------------
    X, y, preprocessor = preprocess(df)

    # -----------------------
    # SPLIT
    # -----------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # -----------------------
    # ROUTER
    # -----------------------
    if model_type == "linear":
        return train_linear(X_train, X_test, y_train, y_test, preprocessor, base_dir)

    elif model_type == "rf":
        return train_random_forest(X_train, X_test, y_train, y_test, preprocessor, base_dir)

    elif model_type == "xgb":
        return train_xgboost(X_train, X_test, y_train, y_test, preprocessor, base_dir)

    elif model_type == "all":
        return train_all_models(X_train, X_test, y_train, y_test, preprocessor, base_dir)
    
    else:
        raise ValueError("Unknown model type")
    
