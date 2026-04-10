# Credit Card Fraud Detection 💳

## 📌 Project Overview

This project detects fraudulent credit card transactions using machine learning classification techniques.

## 📊 Dataset

* Contains transaction details with anonymized features (V1 to V28)
* Additional features:

  * Time
  * Amount
* Target variable: **Class (0 = Normal, 1 = Fraud)**

## ⚙️ Steps Performed

* Data Preprocessing and Feature Scaling using StandardScaler
* Handling Class Imbalance using class_weight='balanced'
* Train-Test Split
* Model Training using Logistic Regression
* Model Evaluation using Precision, Recall, F1-score, and Confusion Matrix
  
## 🤖 Model Used

* Logistic Regression with class balancing

## 📈 Result

* Model achieves high recall in detecting fraudulent transactions, ensuring most fraud cases are correctly identified despite class imbalance.

## 📊 Key Insights

* Dataset is highly imbalanced (very few fraud cases)
* Accuracy is not a reliable metric for this problem
* Model focuses on maximizing recall to detect fraud cases
* Trade-off exists between precision and recall
  
## 🛠️ Technologies Used

* Python
* Pandas
* Scikit-learn

## 📂 Files

* fraud.py → Model code
## 📂 Dataset
Dataset is too large to upload. You can download it from Kaggle:
https://www.kaggle.com/mlg-ulb/creditcardfraud
