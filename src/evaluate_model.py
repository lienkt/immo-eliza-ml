
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """
    Evaluate regression model performance.

    Parameters
    ----------
    name : str
        Model name for logging

    model : sklearn Pipeline
        Trained pipeline (preprocess + model)

    X_train, X_test : pd.DataFrame
        Feature sets

    y_train, y_test : pd.Series
        Target values

    Returns
    -------
    dict
        Model performance metrics
    """

    # -----------------------
    # Predictions
    # -----------------------
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # -----------------------
    # Metrics
    # -----------------------
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    mae = mean_absolute_error(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # -----------------------
    # Overfitting check
    # -----------------------
    gap = train_r2 - test_r2

    if train_r2 < 0.5:
        status = "UNDERFITTING ⚠️"
    elif gap > 0.1:
        status = "OVERFITTING ⚠️"
    else:
        status = "GOOD FIT ✅"

    # -----------------------
    # Print results
    # -----------------------
    print(f"\n===== {name} =====")
    print(f"Train R2 : {round(train_r2, 4)}")
    print(f"Test R2  : {round(test_r2, 4)}")
    print(f"MAE      : {round(mae, 2)}")
    print(f"RMSE     : {round(rmse, 2)}")
    print(f"Status   : {status}")

    # -----------------------
    # Return metrics
    # -----------------------
    return {
        "model": name,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "mae": mae,
        "rmse": rmse,
        "status": status
    }