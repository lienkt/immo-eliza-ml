import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.preprocess import preprocess
from src.evaluate_model import evaluate_model
from src.model_comparison import compare_models


# =====================================================
# MODEL CONFIG (EASY TO EXTEND)
# =====================================================
MODEL_CONFIGS = [
    ("Linear Regression", LinearRegression(), "linear_model.pkl"),
    ("Random Forest", RandomForestRegressor(n_estimators=200, random_state=42), "rf_model.pkl"),
    ("XGBoost", XGBRegressor(
        random_state=42,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6
    ), "xgb_model.pkl")
]


# =====================================================
# CORE TRAIN FUNCTION (REUSABLE ENGINE)
# =====================================================
def _train_model(name, model, X_train, X_test, y_train, y_test,
                 preprocessor, base_dir, filename):

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

    os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)

    joblib.dump(pipeline, os.path.join(base_dir, "models", filename))

    print(f"Model saved: {filename}")

    return pipeline, metrics


# =====================================================
# TRAIN SINGLE MODEL (ROUTER)
# =====================================================
def train_single_model(model_type, X_train, X_test, y_train, y_test,
                       preprocessor, base_dir):

    mapping = {
        "linear": MODEL_CONFIGS[0],
        "rf": MODEL_CONFIGS[1],
        "xgb": MODEL_CONFIGS[2]
    }

    name, model, filename = mapping[model_type]

    return _train_model(
        name, model,
        X_train, X_test,
        y_train, y_test,
        preprocessor,
        base_dir,
        filename
    )


# =====================================================
# TRAIN ALL MODELS (COMPARE MODE)
# =====================================================
def train_all_models(X_train, X_test, y_train, y_test,
                     preprocessor, base_dir):

    all_metrics = {}
    models = {}

    for name, model, filename in MODEL_CONFIGS:
        print(f"\nTraining {name}...")
        pipeline, metrics = _train_model(
            name=name,
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            preprocessor=preprocessor,
            base_dir=base_dir,
            filename=filename
        )

        all_metrics[name] = metrics
        models[name] = pipeline

    best = compare_models(list(all_metrics.values()))

    return models, best


# =====================================================
# MAIN TRAIN FUNCTION
# =====================================================
def trainning_model(df: pd.DataFrame, base_dir: str, model_type: str):

    # -----------------------
    # PREPROCESS
    # -----------------------
    X, y, preprocessor = preprocess(df, model_type)

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
    if model_type == "all":

        return train_all_models(
            X_train, X_test,
            y_train, y_test,
            preprocessor,
            base_dir
        )

    elif model_type in ["linear"]:

        return train_single_model(
            model_type,
            X_train, X_test,
            y_train, y_test,
            preprocessor,
            base_dir
        )
    
    elif model_type in ["rf", "xgb"]:

        return train_single_model(
            model_type,
            X_train, X_test,
            y_train, y_test,
            preprocessor,
            base_dir
        )

    else:
        raise ValueError("Unknown model type")