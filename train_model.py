# Loan Approval Prediction using Logistic Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

df = pd.read_csv(r"C:\Users\LENOVO\Downloads\loan_approval_dataset.csv")
df.columns = df.columns.str.strip()  

print("Dataset Loaded Successfully!")
print(df.head())

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())


df.fillna(df.mode().iloc[0], inplace=True)


if 'loan_id' in df.columns:
    df.drop(columns=['loan_id'], inplace=True)


le = LabelEncoder()
if 'education' in df.columns:
    df['education'] = le.fit_transform(df['education'])
if 'self_employed' in df.columns:
    df['self_employed'] = le.fit_transform(df['self_employed'])
if 'loan_status' in df.columns:
    df['loan_status'] = le.fit_transform(df['loan_status'])

print("\n Encoding Done:")
print(df.head())


print("\nColumns available:", df.columns.tolist())

X = df.drop('loan_status', axis=1)
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining Samples: {len(X_train)}, Testing Samples: {len(X_test)}")


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\n Model Accuracy: {acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Loan Approval Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()


sample = X_test.iloc[0:1]
sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)

print("\nSample Prediction Result:")
print(" Loan Approved" if prediction[0]==1 else " Loan Rejected")


with open("loan_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

print("\n Model and Scaler saved successfully as 'loan_model.pkl' and 'scaler.pkl'!")


with open("loan_model.pkl", "rb") as model_file:
    loaded_model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    loaded_scaler = pickle.load(scaler_file)


sample_scaled2 = loaded_scaler.transform(sample)
prediction2 = loaded_model.predict(sample_scaled2)

print("\n Loaded model works perfectly:")
print(" Loan Approved" if prediction2[0]==1 else " Loan Rejected")
