# 🏠 Immo Eliza - Machine Learning Regression Pipeline

This project builds a machine learning system to predict real estate prices in Belgium using multiple regression models.

We implement a full ML pipeline including:

- Data preprocessing
- Model training
- Evaluation
- Model comparison
- Model saving as reusable pipelines

---

# 📌 Project Goals

- Predict property prices in Belgium
- Build a reusable ML pipeline (train + predict)
- Compare multiple regression models
- Evaluate performance using regression metrics
- Detect overfitting / underfitting
- Select the best performing model automatically

---

# 🧠 Machine Learning Models Used

We compare 3 models:

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

---

# 💾 Model Architecture (Saved Pipelines)

Each model is saved as a FULL PIPELINE:

linear_model.pkl
├── scaler + encoder
└── LinearRegression

rf_model.pkl
├── imputer + encoder
└── RandomForest

xgb_model.pkl
├── imputer + encoder
└── XGBoost

👉 Each file contains BOTH preprocessing + model  
👉 No manual preprocessing needed during prediction

---

# 🚀 Training Flow (CODE STRUCTURE)

training_model()
│
├── preprocess(df)
│ └── X, y
│
├── train_test_split
│
├── train_single_model()
│ │
│ ├── build_preprocessor("linear/rf/xgb")
│ │
│ └── Pipeline(preprocessor + model)
│
└── joblib.dump(pipeline)

---

# 🧠 FULL MACHINE LEARNING PIPELINE ARCHITECTURE

                ┌────────────────────┐
                │      DATASET       │
                │     (raw df)       │
                └─────────┬──────────┘
                          │
                          ▼
            ┌────────────────────────────────┐
            │        preprocess(df)          │
            │  - split X / y                 │
            │  - clean basic data            │
            └─────────┬──────────────────────┘
                        │
                        ▼
            ┌────────────────────────────────┐
            │ build_preprocessor(model_type) │
            │                                │
            │ linear → scaler + onehot       │
            │ tree   → imputer + onehot      │
            └─────────┬──────────────────────┘
                        │
                        ▼
        ┌────────────────────────────────┐
        │        sklearn Pipeline        │
        │                                │
        │  preprocessor → model          │
        └─────────┬──────────────────────┘
                  │
        ┌─────────┴─────────┐
        ▼                   ▼

.fit() .predict()
(TRAINING) (INFERENCE)
│ │
▼ ▼
┌────────────────┐ ┌────────────────┐
│ joblib dump │ │ load pipeline │
│ model.pkl │ │ + predict │
└────────────────┘ └────────────────┘

---

# 🔄 ML PIPELINE STEPS

## 1. Data Preprocessing

- Handle missing values (imputation)
- Encode categorical variables (OneHotEncoding)
- Scale numeric features (only for Linear Regression)

## 2. Model Training

preprocessor → model

## 3. Evaluation

- R² Score
- MAE
- RMSE
- Overfitting check
- Underfitting detection

## 4. Model Comparison

Best model = highest Test R² Score

---

# 🏗 Preprocessing Strategy

## Linear Regression

- SimpleImputer
- StandardScaler
- OneHotEncoder

Why?
Linear models are sensitive to feature scale.

---

## Tree-based models (RF / XGBoost)

- SimpleImputer
- OneHotEncoder

Why?
Tree models do NOT need scaling.

---

# 📁 Project Structure

immo-eliza-ml/
│
├── data/
├── models/
│ ├── linear_model.pkl
│ ├── rf_model.pkl
│ └── xgb_model.pkl
│
├── src/
│ ├── preprocess.py
│ │ ├── preprocess()
│ │ └── build_preprocessor()
│ │
│ ├── train.py
│ ├── evaluate_model.py
│ ├── model_comparison.py
│
├── main.py
├── requirements.txt
└── README.md

---

# 🧪 How to Run

pip install -r requirements.txt

python main.py

Options:
1 → Linear Regression
2 → Random Forest
3 → XGBoost
4 → Train ALL models + compare

---

# 💾 Model Saving

models/
├── linear_model.pkl
├── rf_model.pkl
├── xgb_model.pkl

Each model contains:
preprocessor + model

---

# 📌 Key Idea (VERY IMPORTANT)

Everything used in training is stored inside the pipeline.

pipeline = joblib.load("rf_model.pkl")
pred = pipeline.predict(new_data)

✔ No preprocessing needed  
✔ No scaler/encoder rework  
✔ No data mismatch risk

---

# 🏆 Key Design Decisions

✔ Reusable pipeline  
✔ Modular preprocessing  
✔ No data leakage  
✔ Easy model extension  
✔ Consistent evaluation

---

# 📊 Metrics

- R² Score
- MAE
- RMSE

---

# 🏆 Best Model

Highest Test R² Score wins.

---

# 📈 Example Output

===== Random Forest =====
Train R2: 0.92
Test R2: 0.85
MAE: 35000
RMSE: 52000
Status: GOOD FIT

🏆 BEST MODEL: Random Forest

---

# 🚀 Future Improvements

- GridSearchCV
- Cross Validation
- SHAP explainability
- FastAPI deployment

---

# 👨‍💻 Author

Lienkt
