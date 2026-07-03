import os
import joblib
import pandas as pd


# =====================================================
# SAMPLE INPUT (FOR TESTING ONLY)
# =====================================================
def create_dummy_data():
    """
    Create a sample input row for testing prediction.

    IMPORTANT:
    - Must match training feature columns exactly
    - No missing columns allowed
    """

    return pd.DataFrame([{
        "property_type": "HOUSE",
        "subproperty_type": "DETACHED",
        "province": "Brussels",
        "locality": "Brussels",
        "zip_code": 1000,
        "bedrooms": 3,
        "bathrooms": 2,
        "living_area": 150,
        "surface_of_the_plot": 350,
        "garden": 1,
        "garden_area": 100,
        "terrace": 1,
        "terrace_area": 20,
        "facades": 4,
        "build_year": 2015,
        "state": "GOOD",
        "equipped_kitchen": "INSTALLED",
        "heating_type": "GAS",
        "furnished": 0,
        "swimming_pool": 0
    }])


# =====================================================
# LOAD MODEL
# =====================================================
def load_model(base_dir: str, model_type: str):
    """
    Load trained sklearn Pipeline.

    Parameters
    ----------
    base_dir : str
        Project root directory

    model_type : str
        Model to load:
        - "linear"
        - "rf"
        - "xgb"

    Returns
    -------
    sklearn Pipeline
        Full pipeline (preprocessor + model)
    """

    model_path = os.path.join(
        base_dir,
        "models",
        f"{model_type}_model.pkl"
    )
    return joblib.load(model_path)


# =====================================================
# ALIGN FEATURES 
# =====================================================
def align_features(df: pd.DataFrame, model):
    """
    Ensure input dataframe matches training feature columns.
    Prevents 'missing columns' error.
    """

    try:
        expected_features = model.named_steps["preprocessor"].feature_names_in_
    except Exception:
        # fallback if sklearn version doesn't expose it
        return df

    # Reindex columns to match training
    df_aligned = df.reindex(columns=expected_features, fill_value=0)

    return df_aligned


# =====================================================
# SINGLE PREDICTION
# =====================================================
def predict(df: pd.DataFrame, base_dir: str, model_type: str):
    """
    Make predictions using a trained pipeline.

    IMPORTANT PRINCIPLE:
    --------------------
    The model already contains preprocessing inside pipeline:
        preprocess → model

    So NO manual preprocessing is needed here.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input features (same format as training data)

    base_dir : str
        Project root directory

    model_type : str
        Model type to use:
        - "linear"
        - "rf"
        - "xgb"

    Returns
    -------
    np.ndarray
        Predicted prices
    """
    model = load_model(base_dir, model_type)

    # align schema before prediction
    df = align_features(df, model)

    preds = model.predict(df)

    return preds


# =====================================================
# PREDICT ALL MODELS
# =====================================================
def predict_all(df: pd.DataFrame, base_dir: str):
    """
    Run predictions using all trained models.

    NOTE:
    - Not required for production
    - Useful for comparison or debugging

    Returns
    -------
    dict
        {
            "linear": preds,
            "rf": preds,
            "xgb": preds
        }
    """

    results = {}

    for model_type in ["linear", "rf", "xgb"]:
        model = load_model(base_dir, model_type)

        df_aligned = align_features(df, model)

        results[model_type] = model.predict(df_aligned)

    return results


# =====================================================
# BEST MODEL
# =====================================================
def predict_best(df: pd.DataFrame, base_dir: str):
    """
    Production-style prediction using ONLY the best model.

    REQUIREMENT:
    - You must have saved best_model.pkl during training

    Returns
    -------
    np.ndarray
        Final predicted prices
    """

    model_path = os.path.join(base_dir, "models", "best_model.pkl")
    model = joblib.load(model_path)

    df = align_features(df, model)

    return model.predict(df)