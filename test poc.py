import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

def preprocess_data(df):
    # Convert date columns to datetime
    df["Joining Date"] = pd.to_datetime(df["Joining Date"], errors="coerce")
    df["DOB"] = pd.to_datetime(df["DOB"], errors="coerce")
    
    # Create Age and Tenure columns
    df["Age"] = (pd.to_datetime("today") - df["DOB"]).dt.days / 365.25
    df["Tenure (Years)"] = (pd.to_datetime("today") - df["Joining Date"]).dt.days / 365.25
    
    # Drop unnecessary columns
    drop_columns = ["Employee UID (masked)", "Status", "Resignation Date", "LWD", "Last Promotion Date"]
    df.drop(columns=drop_columns, errors="ignore", inplace=True)
    
    # One-hot encode categorical variables
    categorical_cols = ["Gender", "Designation/Level", "Highest Degree", "Residential Status (Rented/Owned/With Parents)",
                        "Indicator (Y/N) whether shifting to or continue to stay at base location which is close to home",
                        "Manager feedback of pulse (attrition risk level)", "Overall Participation in Events (CSR/EE Events? L&D session)(L/M/H)",
                        "College Tier", "Industry Sector", "Previous Experience Type"]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Fill missing values
    df.fillna(df.median(), inplace=True)
    
    return df

# Load data
df = pd.read_csv("Test data Prediction model.csv")
df = preprocess_data(df)

# Define features and target
X = df.drop(columns=["Attrition"], errors="ignore")
y = df["Attrition"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred_prob = model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, (y_pred_prob > 0.5).astype(int))
roc_auc = roc_auc_score(y_test, y_pred_prob)

# Streamlit App
st.title("Employee Attrition Prediction")

st.write(f"Model Accuracy: {accuracy:.2%}")
st.write(f"ROC AUC Score: {roc_auc:.2%}")

# Function to predict attrition probability
def predict_attrition(employee_data):
    """Takes a single employee's data as input and returns attrition probability."""
    df_input = pd.DataFrame([employee_data])
    df_input = preprocess_data(df_input)
    df_input = df_input.reindex(columns=X.columns, fill_value=0)  # Ensure same feature set
    
    probability = model.predict_proba(df_input)[:, 1][0]
    return probability * 100

# Streamlit UI
employee_data = st.text_area("Enter employee details as JSON:")
if employee_data:
    import json
    employee_data = json.loads(employee_data)
    probability = predict_attrition(employee_data)
    st.write(f"Estimated Attrition Probability: {probability:.2f}%")
