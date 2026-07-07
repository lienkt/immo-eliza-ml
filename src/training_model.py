
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline

from src.hyperparameter_tuning import tune_model
from src.preprocess import preprocess, build_preprocessor
from src.evaluate_model import evaluate_model
from src.model_comparison import compare_models


# =====================================================
# MODEL CONFIGURATION (DICTIONARY-BASED)
# =====================================================
# Purpose:
# Store all model-related metadata in a clean, extensible structure
#
# Key:
# - model_type: identifier used in routing ("linear", "rf", "xgb")
#
# Value:
# - name: human-readable model name
# - model: sklearn model instance
# - filename: file name used when saving trained pipeline
# =====================================================
MODEL_CONFIGS = {
    "linear": {
        "name": "Linear Regression",
        "model": LinearRegression(),
        "filename": "linear_model.pkl"
    },

    "rf": {
        "name": "Random Forest",
        "model": RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ),
        "filename": "rf_model.pkl"
    },

    "xgb": {
        "name": "XGBoost",
        "model": XGBRegressor(
            random_state=42,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6
        ),
        "filename": "xgb_model.pkl"
    }
}


# =====================================================
# INTERNAL TRAINING FUNCTION
# =====================================================
def _train_model(
    name: str,
    model,
    model_type: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    base_dir: str,
    filename: str
):
    """
    Train a single ML model using a full sklearn Pipeline.

    Parameters
    ----------
    name : str
        Human-readable model name used for logging/evaluation.

    model : sklearn estimator
        ML model instance (LinearRegression, RandomForest, XGBoost, etc.).

    model_type : str
        Type of model used to select preprocessing strategy.
        Options: "linear", "rf", "xgb"

    X_train : pd.DataFrame
        Training feature matrix.

    X_test : pd.DataFrame
        Test feature matrix.

    y_train : pd.Series
        Training target values.

    y_test : pd.Series
        Test target values.

    base_dir : str
        Root directory where trained models will be saved.

    filename : str
        File name for saving the trained pipeline (.pkl).

    Returns
    -------
    pipeline : sklearn Pipeline
        Fully trained pipeline (preprocessor + model).

    metrics : dict
        Evaluation metrics (R2, MAE, RMSE, etc.).
    """

    # -------------------------------------------------
    # Build model-specific preprocessing pipeline
    # -------------------------------------------------
    preprocessor = build_preprocessor(X_train, model_type)

    # -------------------------------------------------
    # Create full ML pipeline
    # -------------------------------------------------
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # -------------------------------------------------
    # Train pipeline
    # -------------------------------------------------
    # pipeline.fit(X_train, y_train)
    pipeline = tune_model(
        pipeline,
        model_type,
        X_train,
        y_train
    )

    # -------------------------------------------------
    # Evaluate model performance
    # -------------------------------------------------
    metrics = evaluate_model(
        name,
        pipeline,
        X_train,
        X_test,
        y_train,
        y_test
    )

    # -------------------------------------------------
    # Save trained pipeline (includes preprocessing!)
    # -------------------------------------------------
    os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)

    joblib.dump(
        pipeline,
        os.path.join(base_dir, "models", filename)
    )

    print(f"[INFO] Saved model: {filename}")

    return pipeline, metrics


# =====================================================
# TRAIN SINGLE MODEL (ROUTER)
# =====================================================
def train_single_model(
    model_type: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    base_dir: str
):
    """
    Train a single model based on model_type.

    Parameters
    ----------
    model_type : str
        Model identifier ("linear", "rf", "xgb")

    Returns
    -------
    pipeline : sklearn Pipeline
        Trained pipeline.

    metrics : dict
        Evaluation metrics for the model.
    """

    config = MODEL_CONFIGS[model_type]

    return _train_model(
        name=config["name"],
        model=config["model"],
        model_type=model_type,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        base_dir=base_dir,
        filename=config["filename"]
    )


# =====================================================
# TRAIN ALL MODELS (COMPARISON MODE)
# =====================================================
def train_all_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    base_dir: str
):
    """
    Train all models defined in MODEL_CONFIGS and compare performance.

    Returns
    -------
    models : dict
        Dictionary of trained pipelines.

    best_model : dict
        Best model based on evaluation comparison.
    """

    all_metrics = {}
    models = {}

    # Loop through all model configurations
    for model_type, config in MODEL_CONFIGS.items():

        print(f"[INFO] Training {config['name']}...")

        pipeline, metrics = _train_model(
            name=config["name"],
            model=config["model"],
            model_type=model_type,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            base_dir=base_dir,
            filename=config["filename"]
        )

        all_metrics[config["name"]] = metrics
        models[config["name"]] = pipeline

    # Compare all models and select best
    best = compare_models(list(all_metrics.values()))

    return models, best


# =====================================================
# MAIN TRAINING FUNCTION (ENTRY POINT)
# =====================================================
def training_model(
    df: pd.DataFrame,
    base_dir: str,
    model_type: str
):
    """
    Main entry point for training pipeline.

    Steps:
    1. Split raw dataframe into X, y
    2. Train/test split
    3. Route to single or multi-model training

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataset containing features + target column.

    base_dir : str
        Base directory for saving models.

    model_type : str
        - "linear" → train Linear Regression
        - "rf" → train Random Forest
        - "xgb" → train XGBoost
        - "all" → train and compare all models

    Returns
    -------
    Depends on mode:
    - single model → (pipeline, metrics)
    - all models → (dict of models, best model info)
    """

    # -------------------------------------------------
    # Step 1: Split features and target
    # -------------------------------------------------
    X, y = preprocess(df)

    # -------------------------------------------------
    # Step 2: Train-test split
    # -------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # -------------------------------------------------
    # Step 3: Routing logic
    # -------------------------------------------------
    if model_type == "all":
        return train_all_models(
            X_train, X_test,
            y_train, y_test,
            base_dir
        )

    elif model_type in ["linear", "rf", "xgb"]:
        return train_single_model(
            model_type,
            X_train, X_test,
            y_train, y_test,
            base_dir
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}")