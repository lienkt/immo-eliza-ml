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

- Predict house prices in Belgium
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

# 📦 Model Architecture (Saved Pipelines)

```text
linear_model.pkl
├── StandardScaler
├── OneHotEncoder
└── LinearRegression

rf_model.pkl
├── SimpleImputer
├── OneHotEncoder
└── RandomForestRegressor

xgb_model.pkl
├── SimpleImputer
├── OneHotEncoder
└── XGBoostRegressor
```

---

# 🧠 ML Pipeline Flow

```text
Raw Dataset
    ↓
preprocess(df)
    ↓
X / y split
    ↓
train_test_split
    ↓
ColumnTransformer
    ↓
Pipeline(preprocessor + model)
    ↓
model.fit()
    ↓
joblib.dump()
    ↓
model.predict()
```

---

# 🔄 Training Process

```text
training_model()
    ├── preprocess(df)
    │     └── X, y
    │
    ├── train_test_split
    │
    ├── train_single_model()
    │     └── Pipeline(preprocessor + model)
    │
    └── evaluate_model()
          ├── R² Score
          ├── MAE
          ├── RMSE
          └── Overfitting check
```

---

# 🧪 Preprocessing Strategy

## Linear Regression

- SimpleImputer
- StandardScaler
- OneHotEncoder

👉 Linear models are sensitive to feature scaling

---

## Tree-based models (Random Forest / XGBoost)

- SimpleImputer
- OneHotEncoder

👉 Trees do NOT need scaling

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

# 🔮 Prediction Usage

```python
pipeline = joblib.load("rf_model.pkl")
prediction = pipeline.predict(new_data)
```

✔ No preprocessing needed  
✔ Same input format as training

---

# 📁 Project Structure

```text
immo-eliza-ml/
│
├── data/
├── models/
│   ├── linear_model.pkl
│   ├── rf_model.pkl
│   └── xgb_model.pkl
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate_model.py
│   ├── model_comparison.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

# 🚀 How to Run

```bash
pip install -r requirements.txt
python main.py
```

Menu:

- Train models
- Train all models
- Compare models
- Predict using selected model

---

# 🏆 Best Model - XGBoost

Highest Test R² Score wins:

```text
#1 - XGBoost
----------------------------------------
Test R2   : 0.8292
Train R2  : 0.8947
MAE       : 0.09
RMSE      : 0.12
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
Test R2   : 0.8292
Train R2  : 0.8947
MAE       : 0.09
RMSE      : 0.12
Status    : GOOD FIT ✅

#2 - Random Forest
----------------------------------------
Test R2   : 0.8168
Train R2  : 0.9721
MAE       : 0.09
RMSE      : 0.13
Status    : OVERFITTING ⚠️

#3 - Linear Regression
----------------------------------------
Test R2   : -1.2549
Train R2  : 0.7362
MAE       : 0.13
RMSE      : 0.45
Status    : OVERFITTING ⚠️
```

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
