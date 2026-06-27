
import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Churn Prediction", page_icon="📊", layout="centered")

st.title("📊 Prédiction du churn client")
st.write("Cette application prédit si un client risque de quitter l'entreprise télécom.")

# Charger le modèle et les informations de préparation
with open("best_churn_package.pkl", "rb") as file:
    package = pickle.load(file)

model = package["model"]
columns = package["columns"]
num_cols = package["num_cols"]
scaler = package["scaler"]
uses_scaled = package["uses_scaled"]
model_name = package["model_name"]

st.sidebar.header("Informations client")
st.sidebar.write(f"Modèle utilisé : **{model_name}**")

# Variables numériques
tenure = st.number_input("Ancienneté du client (tenure)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("MonthlyCharges", min_value=0.0, max_value=200.0, value=70.0)
total_charges = st.number_input("TotalCharges", min_value=0.0, max_value=10000.0, value=900.0)
senior = st.selectbox("SeniorCitizen", [0, 1])

# Variables catégorielles principales
gender = st.selectbox("Gender", ["Female", "Male"])
partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])
phone_service = st.selectbox("PhoneService", ["No", "Yes"])
multiple_lines = st.selectbox("MultipleLines", ["No", "Yes", "No phone service"])
internet_service = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("OnlineSecurity", ["No", "Yes", "No internet service"])
online_backup = st.selectbox("OnlineBackup", ["No", "Yes", "No internet service"])
device_protection = st.selectbox("DeviceProtection", ["No", "Yes", "No internet service"])
tech_support = st.selectbox("TechSupport", ["No", "Yes", "No internet service"])
streaming_tv = st.selectbox("StreamingTV", ["No", "Yes", "No internet service"])
streaming_movies = st.selectbox("StreamingMovies", ["No", "Yes", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("PaperlessBilling", ["No", "Yes"])
payment_method = st.selectbox("PaymentMethod", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

# Construire une ligne avec toutes les colonnes attendues
input_data = pd.DataFrame([[0] * len(columns)], columns=columns)

# Remplir les variables numériques si elles existent
for col, val in {
    "SeniorCitizen": senior,
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}.items():
    if col in input_data.columns:
        input_data.loc[0, col] = val

# Fonction pratique pour activer une colonne encodée si elle existe
def set_dummy(column_name, condition=True):
    if condition and column_name in input_data.columns:
        input_data.loc[0, column_name] = 1

set_dummy("gender_Male", gender == "Male")
set_dummy("Partner_Yes", partner == "Yes")
set_dummy("Dependents_Yes", dependents == "Yes")
set_dummy("PhoneService_Yes", phone_service == "Yes")
set_dummy("MultipleLines_Yes", multiple_lines == "Yes")
set_dummy("MultipleLines_No phone service", multiple_lines == "No phone service")
set_dummy("InternetService_Fiber optic", internet_service == "Fiber optic")
set_dummy("InternetService_No", internet_service == "No")
set_dummy("OnlineSecurity_Yes", online_security == "Yes")
set_dummy("OnlineSecurity_No internet service", online_security == "No internet service")
set_dummy("OnlineBackup_Yes", online_backup == "Yes")
set_dummy("OnlineBackup_No internet service", online_backup == "No internet service")
set_dummy("DeviceProtection_Yes", device_protection == "Yes")
set_dummy("DeviceProtection_No internet service", device_protection == "No internet service")
set_dummy("TechSupport_Yes", tech_support == "Yes")
set_dummy("TechSupport_No internet service", tech_support == "No internet service")
set_dummy("StreamingTV_Yes", streaming_tv == "Yes")
set_dummy("StreamingTV_No internet service", streaming_tv == "No internet service")
set_dummy("StreamingMovies_Yes", streaming_movies == "Yes")
set_dummy("StreamingMovies_No internet service", streaming_movies == "No internet service")
set_dummy("Contract_One year", contract == "One year")
set_dummy("Contract_Two year", contract == "Two year")
set_dummy("PaperlessBilling_Yes", paperless_billing == "Yes")
set_dummy("PaymentMethod_Credit card (automatic)", payment_method == "Credit card (automatic)")
set_dummy("PaymentMethod_Electronic check", payment_method == "Electronic check")
set_dummy("PaymentMethod_Mailed check", payment_method == "Mailed check")

# Standardisation si le modèle final l'exige
input_for_prediction = input_data.copy()
if uses_scaled and scaler is not None:
    existing_num_cols = [col for col in num_cols if col in input_for_prediction.columns]
    input_for_prediction[existing_num_cols] = scaler.transform(input_for_prediction[existing_num_cols])

if st.button("Prédire"):
    prediction = model.predict(input_for_prediction)[0]

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_for_prediction)[0][1]

    if prediction == 1:
        st.error("Résultat : le client risque de quitter l'entreprise (Churn).")
    else:
        st.success("Résultat : le client ne semble pas risquer de quitter l'entreprise (No churn).")

    if proba is not None:
        st.write(f"Probabilité estimée de churn : **{proba:.2%}**")

st.caption("Projet Machine Learning - Telco Customer Churn")
