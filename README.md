# 🏠 Immo Eliza - Machine Learning Regression Pipeline

A machine learning system to predict real estate prices in Belgium using multiple regression models.

This project implements a full end-to-end ML pipeline:

- Data preprocessing
- Feature engineering
- Model training
- Evaluation & comparison
- Model persistence (full pipeline saving)
- Prediction system

---

# 🎯 Project Goals

- Predict house and apartment prices in Belgium
- Build reusable ML pipelines (train + predict)
- Compare multiple regression models
- Evaluate performance using regression metrics
- Detect overfitting / underfitting
- Automatically select the best model

---

# 🤖 Machine Learning Models

We compare 3 regression models:

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

---

# ⚙️ Key Design Principle

> Each model is saved as a FULL sklearn PIPELINE

✔ Preprocessing included  
✔ No manual feature engineering during prediction  
✔ No train/predict mismatch risk

---

# 📌 Features

The model uses cleaned and engineered property features to predict house prices.

## Property Information

- `property_type`
- `city`
- `province`
- `property_state`
- `build_year`
- `house_age`

## Location Features

- `latitude`
- `longitude`
- `nearest_city`
- `nearest_city_distance_km`

## Property Characteristics

- `bedroom_count`
- `livable_surface`
- `total_surface`
- `garage`
- `terrace`
- `swimming_pool`

## Energy & Accessibility Features

- `energy_consumption_kWh/m2/year`
- `preschool_distance_m`
- `train_station_distance_m`
- `supermarket_distance_m`

  | Feature | Data Type | Description |
  |---------|-----------|-------------|
  | `property_type` | `str` | Main property type (e.g. apartment, house). |
  | `city` | `str` | City where the property is located. |
  | `province` | `str` | Belgian province of the property. |
  | `latitude` | `float64` | Geographic latitude coordinate. |
  | `longitude` | `float64` | Geographic longitude coordinate. |
  | `property_state` | `str` | Current condition of the property. |
  | `build_year` | `float64` | Year the property was built. |
  | `house_age` | `float64` | Property age, calculated as `2026 - build_year`. |
  | `bedroom_count` | `float64` | Number of bedrooms. |
  | `livable_surface` | `float64` | Livable area in square meters (m²). |
  | `total_surface` | `float64` | Total surface area in square meters (m²). |
  | `garage` | `int64` | Garage availability (`0` = No, `1` = Yes). |
  | `terrace` | `float64` | Terrace availability (`0` = No, `1` = Yes). |
  | `energy_consumption` | `float64` | Energy consumption in kWh/m²/year. |
  | `swimming_pool` | `int64` | Swimming pool availability (`0` = No, `1` = Yes). |
  | `preschool_distance` | `float64` | Distance to the nearest preschool (meters). |
  | `train_station_distance` | `float64` | Distance to the nearest train station (meters). |
  | `supermarket_distance` | `float64` | Distance to the nearest supermarket (meters). |
  | `nearest_city` | `str` | Name of the nearest city. |
  | `nearest_city_distance` | `float64` | Distance to the nearest city (kilometers). |

## Removed Features

The following columns are removed before training:

- `property_id` → identifier, not useful for prediction
- `address` → high-cardinality text feature
- `postcode` → removed to avoid location overfitting
- `price_per_m2` → removed to prevent target leakage

## Target Variable

- `price`

The target is transformed using:

```text
y = log10(price)
```

to reduce skewness in the price distribution.

---

# 📦 Model Architecture (Saved Pipelines)

```text
linear_model.pkl
├── Preprocessor
│   ├── Numeric features
│   │   ├── StandardScaler
│   │   └── KNNImputer
│   └── Categorical features
│       ├── SimpleImputer (most_frequent)
│       └── OneHotEncoder
└── LinearRegression

rf_model.pkl
├── Preprocessor
│   ├── Numeric features
│   │   ├── StandardScaler
│   │   └── KNNImputer
│   └── Categorical features
│       ├── SimpleImputer (most_frequent)
│       └── OneHotEncoder
└── RandomForestRegressor

xgb_model.pkl
├── Preprocessor
│   ├── Numeric features
│   │   ├── StandardScaler
│   │   └── KNNImputer
│   └── Categorical features
│       ├── SimpleImputer (most_frequent)
│       └── OneHotEncoder
└── XGBRegressor
```

---

# 🧠 ML Pipeline Flow

```text
Raw Dataset
    │
    ▼
cleaning_data(df)
    │
    ▼
preprocess(df)
    ├── X (features)
    └── y = log10(price)
    │
    ▼
train_test_split()
    │
    ▼
build_preprocessor()
    ├── Numeric
    │     ├── KNNImputer
    │     └── StandardScaler
    └── Categorical
          ├── SimpleImputer
          └── OneHotEncoder
    │
    ▼
Pipeline
(preprocessor + ML model)
    ↓
Hyperparameter Tuning
    │
    ▼
pipeline.fit()
    │
    ▼
evaluate_model()
    │
    ▼
joblib.dump()
(save trained pipeline)
    │
    ▼
joblib.load()
    │
    ▼
pipeline.predict()
    │
    ▼
10^prediction
(convert log price back to €)
```

---

# 🧪 Preprocessing Strategy

All models use the same preprocessing pipeline:

## Numeric Features

- StandardScaler  
  → Standardize numerical features

- KNNImputer  
  → Handle missing numerical values using nearest-neighbor estimation

## Categorical Features

- SimpleImputer(strategy="most_frequent")  
  → Fill missing categorical values

- OneHotEncoder(handle_unknown="ignore")  
  → Convert categorical variables into numerical features

## Model Compatibility

The preprocessing pipeline is applied to:

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

The complete preprocessing and model are combined into a single sklearn Pipeline and saved as a `.pkl` file.

---

# 📊 Evaluation Metrics

- R² Score
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

Also includes:

- Overfitting detection
- Underfitting detection

---

# 🏆 Model Selection

Best model = highest **Test R² Score**

---

# 📁 Project Structure

```text
immo-eliza-ml/
│
├── dev/
├── data/
├── models/
│   ├── linear_model.pkl
│   ├── rf_model.pkl
│   └── xgb_model.pkl
│
├── src/
│   ├── __init__.py
│   ├── cleaning.py
│   ├── preprocess.py
│   ├── hyperparameter_tuning.py
│   ├── training_model.py
│   ├── evaluate_model.py
│   ├── model_comparison.py
│   ├── predict.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

# 🚀 How to Run

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Run the Application

```bash
python main.py
```

## 3. Available Menu Options

The application provides an interactive CLI menu:

- **Train Linear Regression**
  - Train a Linear Regression model and save the pipeline

- **Train Random Forest**
  - Train a Random Forest model and save the pipeline

- **Train XGBoost**
  - Train an XGBoost model and save the pipeline

- **Train ALL Models**
  - Train all models
  - Evaluate performance
  - Compare model results

- **Predict**
  - Select a trained model (`linear`, `rf`, `xgb`)
  - Generate price prediction
  - Compare predicted price with actual price

- **Show Features**
  - Display the number of features after preprocessing
  - Preview generated feature names after encoding

- **Exit**
  - Close the application

---

# 🏆 Best Model - XGBoost

Highest Test R² Score wins:

```text
#1 - XGBoost
----------------------------------------
Test R2   : 0.7947
Train R2  : 0.7954
MAE       : 0.1
RMSE      : 0.14
Status    : GOOD FIT ✅

```

---

# 📈 Example Output

```text
============================================================
🏆 MODEL PERFORMANCE LEADERBOARD
============================================================

#1 - XGBoost
----------------------------------------
Test R2   : 0.7947
Train R2  : 0.7954
MAE       : 0.1
RMSE      : 0.14
Status    : GOOD FIT ✅

#2 - Random Forest
----------------------------------------
Test R2   : 0.7863
Train R2  : 0.8676
MAE       : 0.1
RMSE      : 0.14
Status    : GOOD FIT ✅

#3 - Linear Regression
----------------------------------------
Test R2   : -0.3777
Train R2  : 0.7289
MAE       : 0.13
RMSE      : 0.35
Status    : OVERFITTING ⚠️

============================================================
🥇 BEST MODEL: XGBoost
👉 Test R2: 0.7947
============================================================
```

---

🔮 Prediction Usage

The saved model already contains preprocessing.

Prediction flow:

```text
Raw Property Data
|
v
Feature Engineering
(create house_age)
|
v
Saved sklearn Pipeline
|
|
+-- Missing value handling
+-- Scaling
+-- Encoding
|
v
Regression Model
|
v
Predicted log(price)
|
v
Convert to €

```

Example data:

```text
{
  "property_type": "apartment",
  "city": "Mortsel",
  "province": "antwerp",
  "latitude": 51.17245,
  "longitude": 4.446859,
  "price": 149000,
  "property_state": "Normal",
  "build_year": 1969,
  "bedroom_count": 1,
  "livable_surface": 60,
  "total_surface": 100,
  "garage": 0,
  "terrace": 0,
  "energy_consumption": 505,
  "swimming_pool": 0,
  "preschool_distance": 796,
  "train_station_distance": 1000,
  "supermarket_distance": 181,
  "nearest_city": "Antwerp",
  "nearest_city_distance": 6.07
}

```

Advantages:

✔ No manual preprocessing
✔ Same pipeline during training and prediction
✔ Ready for FastAPI deployment

---

# 🚀 Future Improvements

- GridSearchCV optimization
- Cross-validation
- SHAP explainability
- FastAPI deployment
- MLflow model tracking

---

# 👨‍💻 Author

Lienkt
