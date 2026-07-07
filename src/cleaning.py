import numpy as np
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
    cols_to_drop = ["property_id", "address", "price_per_m2", "postcode"]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # If total_surface is missing, fill with livable_surface
    if "total_surface" in df.columns and "livable_surface" in df.columns:
        df["total_surface"] = df["total_surface"].fillna(df["livable_surface"])

    # If total_surface is missing and property_type is apartment, fill with livable_surface
    if "total_surface" in df.columns and "livable_surface" in df.columns and "property_type" in df.columns:
        df.loc[(df["total_surface"].isna()) & (df["property_type"] == "apartment"), "total_surface"] = df["livable_surface"]    

    # If garage is missing, fill with 0
    if "garage" in df.columns:
        df["garage"] = df["garage"].fillna(0)
    
    # If terrace is missing, fill with 0
    if "terrace" in df.columns:
        df["terrace"] = df["terrace"].fillna(0)
    
    # If swimming_pool is missing, fill with 0
    if "swimming_pool" in df.columns:
        df["swimming_pool"] = df["swimming_pool"].fillna(0)

    # 2. Feature engineering
    # if "build_year" in df.columns:
    #     df["house_age"] = 2026 - df["build_year"]

    # # 3. Missing value flags (optional but useful)
    # num_cols = df.select_dtypes(include=["int64", "float64"]).columns

    # # Fill missing values with NaN for numeric columns to create missing flags
    # df = df.fillna({col: np.nan for col in num_cols})

    # for col in num_cols:
    #     df[col + "_missing"] = df[col].isna().astype(int)

    # 4. Debug check
    print(f"Missing values remaining: {df.isnull().sum().sum()}")

    return df