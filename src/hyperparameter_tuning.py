from sklearn.model_selection import RandomizedSearchCV

# =====================================================
# PARAMETER GRIDS
# =====================================================
PARAM_GRIDS = {
    "rf": {
        "model__n_estimators": [50],
        "model__max_depth": [10],
        "model__min_samples_split": [2],
        "model__min_samples_leaf": [1]
    },

    "xgb": {
        "model__n_estimators": [100],
        "model__learning_rate": [0.1],
        "model__max_depth": [3],
        "model__subsample": [0.8],
        "model__colsample_bytree": [0.8]
    }
}

# =====================================================
# HYPERPARAMETER TUNING
# =====================================================

def tune_model(
    pipeline,
    model_type,
    X_train,
    y_train,
    cv=2,
    scoring="r2",
    n_iter=1,
):
    """
    Tune model hyperparameters using RandomizedSearchCV.

    Parameters
    ----------
    pipeline : sklearn Pipeline
        Pipeline(preprocessor + model)

    model_type : str
        "rf" or "xgb"

    X_train : pd.DataFrame

    y_train : pd.Series

    Returns
    -------
    best_pipeline : sklearn Pipeline
        Best pipeline after tuning.
    """

    if model_type not in PARAM_GRIDS:
        pipeline.fit(X_train, y_train)
        return pipeline

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=PARAM_GRIDS[model_type],
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        random_state=42
    )

    search.fit(X_train, y_train)

    print("\n===== BEST PARAMETERS =====")
    print(search.best_params_)

    print(f"Best CV R²: {search.best_score_:.4f}")

    return search.best_estimator_