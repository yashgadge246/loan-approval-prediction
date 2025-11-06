# 🏦 Loan Approval Prediction App

This project predicts whether a **loan application will be approved or rejected** based on applicant details such as income, credit score, and asset values using **Logistic Regression** and **Streamlit**.

---

## 🚀 Project Overview

The **Loan Approval Prediction System** automates the process of predicting loan approval.  
It uses a **Machine Learning model (Logistic Regression)** trained on applicant data to classify whether a loan should be approved or not.

---

## 🧠 Features

- Predicts **Loan Approval Status** (Approved / Rejected)  
- Uses **Logistic Regression** for accurate and interpretable results  
- **Interactive Streamlit Web App** for user-friendly input  
- Displays **confidence percentage** for each prediction  
- Includes **data visualization** using Seaborn & Matplotlib  

---

## 🧾 Dataset Description

| Feature | Description |
|----------|--------------|
| `no_of_dependents` | Number of dependents |
| `education` | Graduate / Not Graduate |
| `self_employed` | Whether applicant is self-employed |
| `income_annum` | Annual income (₹) |
| `loan_amount` | Requested loan amount (₹) |
| `loan_term` | Duration of the loan (in months) |
| `cibil_score` | Applicant’s credit score |
| `residential_assets_value` | Value of residential assets (₹) |
| `commercial_assets_value` | Value of commercial assets (₹) |
| `luxury_assets_value` | Value of luxury assets (₹) |
| `bank_asset_value` | Total bank asset value (₹) |
| `loan_status` | Target variable (Approved / Rejected) |

---

## ⚙️ Tech Stack

- **Python**
- **Pandas, NumPy**
- **Matplotlib, Seaborn**
- **Scikit-learn**
- **Streamlit**
- **Pickle**

---

## 🧩 Model Building Process

1. **Data Cleaning** – Handling missing values using mode imputation  
2. **Encoding** – Converting categorical columns to numeric form  
3. **Feature Scaling** – Using `StandardScaler`  
4. **Model Training** – Logistic Regression  
5. **Evaluation** – Accuracy, Confusion Matrix, and Classification Report  
6. **Saving Model** – Exporting model and scaler using `pickle`

---

## 📊 Model Performance

- **Algorithm Used:** Logistic Regression  
- **Accuracy:** ~85–90% (depending on dataset)  
- **Metrics:** Accuracy, Precision, Recall, F1-Score  

---
