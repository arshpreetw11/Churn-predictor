# Customer Churn Prediction System

An end-to-end customer churn prediction project built using a production-style machine learning pipeline. The system focuses on identifying customers at high risk of churn, with an emphasis on recall for the positive (churn) class, which is critical for real-world retention use cases.

---

## Problem Statement

Customer churn directly impacts revenue. The objective of this project is to build a predictive system that can accurately identify customers who are likely to churn, enabling proactive intervention strategies. Since missing a churner is more costly than incorrectly flagging a non-churner, the project prioritizes recall and probability-based decision making over raw accuracy.

---

## Dataset

Telco Customer Churn Dataset

Target Variable:

* Churn (Yes / No → 1 / 0)

Feature Categories:

* Customer demographics (gender, senior citizen, dependents)
* Subscription services (internet, streaming, security, support)
* Contract and payment details
* Tenure and billing information

---

## System Architecture


Raw Data
   ↓
Train / Test Split
   ↓
Feature Engineering
   ↓
Skewness Correction
   ↓
Scaling and Encoding
   ↓
Random Forest Classifier
   ↓
Probability Output
   ↓
Threshold-Based Risk Classification


All preprocessing steps are encapsulated inside a single sklearn Pipeline to prevent data leakage and ensure consistent behavior across training, validation, and inference.

---

## Feature Engineering

Custom feature engineering is implemented using a sklearn-compatible transformer:

* Tenure grouping (TenureGroup)
* Average monthly spend (AvgMonthlySpend)
* Number of services subscribed (NumServices)
* Binary encoding for Yes/No features
* Nominal categorical variables handled via One-Hot Encoding

Skewed numerical features are corrected using log or Yeo–Johnson transformations before scaling.

---

## Model Development

Models evaluated:

* Random Forest Classifier (final model)
* XGBoost Classifier (baseline comparison)

Model selection was based on business-relevant metrics rather than accuracy alone. Random Forest was chosen due to its higher churn recall and more stable threshold behavior on the test set.

Hyperparameter tuning was performed using Optuna with cross-validation. The optimization objective was ROC-AUC to ensure robust probability ranking.

---

## Threshold Tuning

Instead of using the default probability threshold of 0.5, a data-driven threshold was selected using ROC curve analysis.

Youden’s J statistic was used to determine the optimal threshold:

python
FINAL_THRESHOLD = best_threshold_j


This allows explicit control over the precision–recall trade-off and aligns the model’s behavior with real business requirements.

---

## Model Performance (Test Set)

* Churn Recall (Class 1): approximately 74%
* Precision (Class 1): approximately 54%
* ROC-AUC: optimized via cross-validation

The results reflect a practical churn prediction system that prioritizes identifying at-risk customers rather than maximizing overall accuracy.

---

## Deployment

The trained model is deployed directly from the notebook using Gradio:

* No retraining during inference
* Uses the trained sklearn Pipeline object
* Accepts raw customer inputs
* Returns probability-based churn risk categories

Risk categories:

* Low Churn Risk
* Moderate Churn Risk
* High Churn Risk
* Very High Churn Risk

The deployment includes basic validation and is designed for demonstration and evaluation purposes.

---

## Tech Stack

* Python
* pandas, numpy
* scikit-learn
* Optuna
* XGBoost (evaluation)
* Gradio

---

## Key Learnings

* Accuracy is insufficient for imbalanced classification problems
* Threshold tuning is critical for business-aligned decision making
* Encapsulating preprocessing inside pipelines prevents data leakage
* Defensive input handling is important for ML deployment

---

## Notes

This project was developed with an industry-oriented mindset, focusing on correctness, reproducibility, and practical deployment. It is suitable as a strong intern or early junior-level machine learning project.
