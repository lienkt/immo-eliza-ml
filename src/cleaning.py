import pandas as pd

def cleaning_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the input DataFrame by handling missing values and performing feature engineering.
    Parameters:
        df (pd.DataFrame): The input DataFrame to be cleaned.
    Returns:
        pd.DataFrame: The cleaned DataFrame.
      """
    
    df = df.copy()

    # 1. Drop useless / leakage columns
    cols_to_drop = ["property_id", "address", "price_per_m2"]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # 2. Feature engineering
    if "build_year" in df.columns:
        df["house_age"] = 2026 - df["build_year"]

    # 3. Missing value flags (optional but useful)
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns

    for col in num_cols:
        df[col + "_missing"] = df[col].isna().astype(int)

    # 4. Debug check
    print(f"Missing values remaining: {df.isnull().sum().sum()}")

    return df