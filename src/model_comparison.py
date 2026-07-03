
import pandas as pd


def compare_models(all_metrics):
    """
    Compare multiple regression models and return ranking.

    Parameters
    ----------
    all_metrics : list[dict]
        List of model metrics dictionaries.

    Returns
    -------
    dict
        Best performing model metrics.
    """

    # -----------------------------
    # CREATE DATAFRAME SAFELY
    # -----------------------------
    df = pd.DataFrame(list(all_metrics))

    # -----------------------------
    # SORT MODELS
    # -----------------------------
    df = df.sort_values(
        by="test_r2",
        ascending=False,
        kind="mergesort"
    ).reset_index(drop=True)

    df["rank"] = df.index + 1

    # -----------------------------
    # HEADER
    # -----------------------------
    print("\n" + "=" * 60)
    print("🏆 MODEL PERFORMANCE LEADERBOARD")
    print("=" * 60)

    # -----------------------------
    # DETAILS
    # -----------------------------
    for _, row in df.iterrows():

        print(f"\n#{row['rank']} - {row.get('model', 'Unknown')}")
        print("-" * 40)
        print(f"Test R2   : {round(row.get('test_r2', 0), 4)}")
        print(f"Train R2  : {round(row.get('train_r2', 0), 4)}")
        print(f"MAE       : {round(row.get('mae', 0), 2)}")
        print(f"RMSE      : {round(row.get('rmse', 0), 2)}")
        print(f"Status    : {row.get('status', 'N/A')}")

    # -----------------------------
    # BEST MODEL
    # -----------------------------
    best = df.iloc[0]

    print("\n" + "=" * 60)
    print(f"🥇 BEST MODEL: {best.get('model', 'Unknown')}")
    print(f"👉 Test R2: {round(best.get('test_r2', 0), 4)}")
    print("=" * 60)

    # -----------------------------
    # INSIGHT
    # -----------------------------
    print("\n📊 INSIGHT SUMMARY")

    performance_gap = best["test_r2"] - df.iloc[-1]["test_r2"]

    if best["test_r2"] >= 0.80:
        print("✅ Excellent predictive performance.")
    elif best["test_r2"] >= 0.60:
        print("👍 Good predictive performance.")
    elif best["test_r2"] >= 0.40:
        print("⚠️ Moderate predictive performance.")
    else:
        print("❌ Poor predictive performance.")

    if performance_gap < 0.02:
        print("• Models perform similarly.")
    elif performance_gap < 0.10:
        print("• Moderate performance difference.")
    else:
        print("• Large performance gap.")

    # -----------------------------
    # FAILED MODELS
    # -----------------------------
    failed_models = df[df["test_r2"] < 0]

    if not failed_models.empty:
        print("\n🚨 WARNING: Failed models detected")

        for model in failed_models.get("model", []):
            print(f" - {model}")

    return best.to_dict()