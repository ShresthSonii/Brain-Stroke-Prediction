# 🧠 Brain Stroke Risk Prediction System

An end-to-end Machine Learning system and interactive Flask web application that predicts patient stroke risk based on clinical health parameters.

---

## 📌 Project Overview

Stroke is one of the leading causes of disability and mortality worldwide. This project uses supervised machine learning to evaluate patient demographic and medical data—such as age, hypertension, heart disease history, average glucose levels, and BMI—to classify stroke risk early.

- **Class Imbalance Handling:** Uses SMOTE (Synthetic Minority Over-sampling Technique) to balance clinical data.
- **Web Interface:** Built-in Flask application for entering patient details and receiving instant risk evaluations.
- **Model Evaluation:** Evaluated using ROC-AUC curves, Precision-Recall metrics, and Confusion Matrices.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Web Framework:** Flask
- **Machine Learning & Analytics:** Scikit-Learn, Imbalanced-Learn (SMOTE), Joblib, Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn

---

## 📁 Repository Structure

├── static/                         # CSS/JS assets for Flask frontend
├── templates/                      # HTML views for web application
├── app.py                          # Main Flask web application server
├── train.py                        # Machine learning training pipeline
├── evaluate_advanced.py            # Model evaluation script
├── model.joblib                    # Saved trained model pipeline
├── Stroke Prediction Using Python.ipynb # Data analysis & model notebook
├── confusion_matrix.png            # Model evaluation plot
├── pr_curve.png                    # Precision-Recall curve
└── roc_curve.png                   # ROC curve metric plot


---

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/ShresthSonii/Brain-Stroke-Prediction.git](https://github.com/ShresthSonii/Brain-Stroke-Prediction.git)
   cd Brain-Stroke-Prediction
