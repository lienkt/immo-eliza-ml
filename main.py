
from src import cleaning_data, training_model, predict, create_dummy_data
import pandas as pd
import os


def main():
    """
    Main entry point for ML project.

    Features:
    - Train single or all models
    - Predict using selected model
    """

    # ---------------------------------------
    # Load dataset
    # ---------------------------------------
    base_dir = os.path.dirname(__file__)

    file_path = os.path.join(
        base_dir,
        "data/cleaned/clean_dataframe.csv"
    )

    df = pd.read_csv(file_path)

    print(f"Dataset shape: {df.shape}")

    df = cleaning_data(df)
    df.info()

    # ---------------------------------------
    # MENU LOOP
    # ---------------------------------------
    while True:

        print("\n=== ML PIPELINE MENU ===")
        print("1 : Train Linear Regression")
        print("2 : Train Random Forest")
        print("3 : Train XGBoost")
        print("4 : Train ALL Models")
        print("5 : Predict (choose model)")
        print("6 : Show features of a model")
        # print("6 : Predict BEST model")
        print("7 : Exit")

        option = input("Choose option: ")

        # -----------------------------------
        # TRAINING
        # -----------------------------------
        if option == "1":
            print("\nTraining Linear Regression...")
            training_model(df, base_dir, "linear")

        elif option == "2":
            print("\nTraining Random Forest...")
            training_model(df, base_dir, "rf")

        elif option == "3":
            print("\nTraining XGBoost...")
            training_model(df, base_dir, "xgb")

        elif option == "4":
            print("\nTraining ALL models...")
            training_model(df, base_dir, "all")

        # -----------------------------------
        # PREDICTION (SINGLE MODEL)
        # -----------------------------------
        elif option == "5":
            print("\nChoose model for prediction:")
            print("linear / rf / xgb")

            model_type = input("Model: ").strip()

            # ============================
            # TAKE SAMPLE (KEEP PRICE FOR COMPARISON)
            # ============================
            sample = df.sample(1)

            price_actual = sample["price"].values[0]
            input_df = sample.drop("price", axis=1)

            # ============================
            # PRINT INPUT DATA (CLEAN VERSION)
            # ============================
            print("\n📊 INPUT DATA FOR PREDICTION")
            print("=" * 60)

            cols_to_show = [
                "property_type",
                "city",
                "latitude",
                "longitude",
                "bedroom_count",
                "livable_surface",
                "total_surface",
                "house_age"
            ]

            print(input_df[cols_to_show].to_string(index=False))

            print("=" * 60)

            # ============================
            # PREDICT
            # ============================
            result = predict(
                input_df,
                base_dir,
                model_type=model_type
            )

            price_pred = result[0]

            # ============================
            # COMPARE RESULT
            # ============================
            print("\n💰 PRICE COMPARISON")
            print("=" * 60)
            print(f"Model        : {model_type}")
            # Convert log price back to original scale for comparison
            price_pred_orig = 10. ** price_pred
            print(f"Price actual : €{price_actual:,.2f}")
            print(f"Price pred   : €{price_pred_orig:,.2f}")
            print(f"Error        : €{abs(price_actual - price_pred_orig):,.2f}")
            print("=" * 60)



        # -----------------------------------
        # PREDICTION (BEST MODEL)
        # -----------------------------------
        # elif option == "6":
        #     print("\nPredicting with BEST model...")

        #     input_df = create_dummy_data()

        #     result = predict(
        #         input_df,
        #         base_dir,
        #         model_type="best"
        #     )

        #     print(f"\nBEST Prediction: €{result[0]:,.2f}")

        elif option == "6":
            print("\n📊 FEATURE COUNT AFTER PREPROCESSING")
            print("=" * 60)

            # load a trained model (any one is fine)
            model_type = "xgb"  # or "linear", "rf"

            import joblib
            model_path = os.path.join(base_dir, "models", f"{model_type}_model.pkl")
            model = joblib.load(model_path)

            # get feature names after preprocessing
            feature_names = model.named_steps["preprocessor"].get_feature_names_out()

            print(f"Total features after preprocessing: {len(feature_names)}")

            print("\nSample features:")
            print(feature_names[:20])  # show first 20 only

            print("=" * 60)

        # -----------------------------------
        # EXIT
        # -----------------------------------
        elif option == "7":
            print("Exiting...")
            break

        else:
            print("Invalid option. Try again.")


if __name__ == "__main__":
    main()