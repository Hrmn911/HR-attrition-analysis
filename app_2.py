import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ----------------------------
# LOAD MODEL FILES
# ----------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="HR Attrition Predictor", layout="centered")

# ----------------------------
# TITLE
# ----------------------------
st.title("🏢 HR Employee Attrition Prediction System")
st.markdown("Predict whether an employee will stay or leave the company using Machine Learning.")

# ----------------------------
# MODEL METRICS (use real values)
# ----------------------------
accuracy = 0.7515
roc_auc = 0.798

col1, col2 = st.columns(2)
col1.metric("🎯 Accuracy", f"{accuracy*100:.2f}%")
col2.metric("📊 ROC-AUC", f"{roc_auc:.2f}")

st.divider()

st.subheader("📥 Employee Details (Please fill all fields)")

# ----------------------------
# EMPTY INPUTS using session_state
# ----------------------------

if "age" not in st.session_state:
    st.session_state.age = 0

if "income" not in st.session_state:
    st.session_state.income = 0

if "years" not in st.session_state:
    st.session_state.years = 0

if "distance" not in st.session_state:
    st.session_state.distance = 0


age = st.number_input("Age", min_value=0, max_value=60, key="age")

monthly_income = st.number_input("Monthly Income", min_value=0, max_value=20000000, key="income")

years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, key="years")

distance_from_home = st.number_input("Distance From Home (km)", min_value=0, max_value=100, key="distance")

overtime = st.selectbox("OverTime", ["Select", "No", "Yes"])

business_travel = st.selectbox(
    "Business Travel",
    ["Select", "Non-Travel", "Travel_Rarely", "Travel_Frequently"]
)

job_satisfaction = st.slider("Job Satisfaction (1-Low, 4-High)", 1, 4, 1)

work_life_balance = st.slider("Work Life Balance (1-Low, 4-High)", 1, 4, 1)

# ----------------------------
# VALIDATION
# ----------------------------
if st.button("🔮 Predict Attrition"):

    # create structure
    input_data = pd.DataFrame(np.zeros((1, len(features))), columns=features)

    # numeric features
    if "Age" in input_data.columns:
        input_data["Age"] = age

    if "MonthlyIncome" in input_data.columns:
        input_data["MonthlyIncome"] = monthly_income

    if "YearsAtCompany" in input_data.columns:
        input_data["YearsAtCompany"] = years_at_company

    if "DistanceFromHome" in input_data.columns:
        input_data["DistanceFromHome"] = distance_from_home

    if "JobSatisfaction" in input_data.columns:
        input_data["JobSatisfaction"] = job_satisfaction

    if "WorkLifeBalance" in input_data.columns:
        input_data["WorkLifeBalance"] = work_life_balance

    # categorical features
    if "OverTime_Yes" in input_data.columns:
        input_data["OverTime_Yes"] = 1 if overtime == "Yes" else 0

    if business_travel == "Travel_Rarely" and "BusinessTravel_Travel_Rarely" in input_data.columns:
        input_data["BusinessTravel_Travel_Rarely"] = 1

    if business_travel == "Travel_Frequently" and "BusinessTravel_Travel_Frequently" in input_data.columns:
        input_data["BusinessTravel_Travel_Frequently"] = 1


    # ----------------------------
    # SCALE INPUT
    # ----------------------------
    input_scaled = scaler.transform(input_data)

    # ----------------------------
    # PREDICTION
    # ----------------------------
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # ----------------------------
    # OUTPUT
    # ----------------------------
    st.divider()
    st.subheader("📊 Prediction Result")

    # MAIN DECISION BASED ON PROBABILITY
    if probability >= 0.7:
        st.error("⚠ Employee is likely to LEAVE the company")
    elif probability >= 0.4:
        st.warning("🟡 Employee is at MODERATE RISK")
    else:
        st.success("🟢 Employee is likely to STAY")

    # ----------------------------
    # PROBABILITY DISPLAY
    # ----------------------------
    st.metric("📈 Probability of Leaving", f"{probability:.2f}")

    # ----------------------------
    # RISK LEVEL
    # ----------------------------
    st.subheader("🧠 Risk Level")

    if probability >= 0.7:
        st.error("🔴 High Risk Employee")
    elif probability >= 0.4:
        st.warning("🟡 Medium Risk Employee")
    else:
        st.success("🟢 Low Risk Employee")