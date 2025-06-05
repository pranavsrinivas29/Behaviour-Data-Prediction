# Behaviour-Data-Prediction

# Personality Prediction from Behavioral Features 🧠

This project focuses on predicting **personality types** (specifically, whether someone is an introvert) based on behavioral survey features. The goal is to compare multiple machine learning classifiers and evaluate their performance using common metrics and visualizations.

---

## 📌 Project Objective

To classify individuals as **Introverts** (`1`) or **Extroverts** (`0`) based on behavioral traits such as:

- Time spent alone
- Social event attendance
- Stage fear
- Social media usage
- Friends circle size
- Drained after socializing

---

## 📁 Dataset Overview

- **Features**: Ordinal, nominal, and binary (booleans)
- **Target**: `Personality` (binary)
- **Preprocessing**:
  - Missing value imputation (group-wise, skew-aware)
  - Label encoding and one-hot encoding
  - Boolean-to-binary conversion
  - Feature scaling using `StandardScaler`

---



## 📈 Sample Output (Interactive Plot)

![Model Comparison Chart](path/to/image_or_screenshot.png)

---

## ⚙️ Model Workflow

### 1. **Train-Test Split**
- 70% training / 30% testing
- Stratified split to preserve class distribution

### 2. **Feature Scaling**
- Applied `StandardScaler` to numeric features

### 3. **Models Trained**
| Model              | Notes                                  |
|-------------------|----------------------------------------|
| Logistic Regression | Tuned with `GridSearchCV` |
| K-Nearest Neighbors | Tuned on `n_neighbors`, `weights`, and `metric` |
| Support Vector Classifier | Tuned kernel (`linear`, `rbf`, `poly`) and `C` |
| Random Forest      | Tuned depth, estimators, sample split criteria |

### 4. **Hyperparameter Tuning**
- Used `GridSearchCV` with 5-fold cross-validation
- Scored primarily on **F1-score**

---

## 📊 Evaluation Metrics

Each model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score

### 🔍 Visual Comparison
- Bar chart comparing all metrics across models (via `plotly.express`)


---

## ✅ Key Findings

All four models performed similarly with an overall **accuracy of ~92%** and identical **macro F1-scores**, indicating balanced performance across both classes. Here are the details:

---

### 🔹 Logistic Regression
- **Best Parameters**: `{'C': 0.01, 'class_weight': None, 'penalty': 'l2', 'solver': 'liblinear'}`
- **Accuracy**: `91.95%`
- **Precision/Recall (Class 0)**: `0.94 / 0.90`
- **Precision/Recall (Class 1)**: `0.90 / 0.94`
- **Macro F1-score**: `0.92`

---

### 🔹 K-Nearest Neighbors (KNN)
- **Best Parameters**: `{'metric': 'manhattan', 'n_neighbors': 9, 'weights': 'uniform'}`
- **Accuracy**: `91.84%`
- **Precision/Recall (Class 0)**: `0.94 / 0.90`
- **Precision/Recall (Class 1)**: `0.90 / 0.94`
- **Macro F1-score**: `0.92`

---

### 🔹 Support Vector Classifier (SVC)
- **Best Parameters**: `{'C': 0.1, 'class_weight': None, 'gamma': 'scale', 'kernel': 'linear'}`
- **Accuracy**: `91.95%`
- **Precision/Recall (Class 0)**: `0.94 / 0.90`
- **Precision/Recall (Class 1)**: `0.90 / 0.94`
- **Macro F1-score**: `0.92`

---

### 🔹 Random Forest
- **Best Parameters**: `{'class_weight': None, 'max_depth': None, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 100}`
- **Accuracy**: `91.72%`
- **Precision/Recall (Class 0)**: `0.94 / 0.90`
- **Precision/Recall (Class 1)**: `0.90 / 0.93`
- **Macro F1-score**: `0.92`

---

### 🧠 Summary
- All models performed **equally well** on macro metrics (F1 = 0.92), indicating **balanced performance** for both introvert and non-introvert classifications.
- **Logistic Regression and SVC** slightly edge out in accuracy.
- **Random Forest** offers robustness and feature importance but had marginally lower recall for Class 1.

---


## 📦 Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly
