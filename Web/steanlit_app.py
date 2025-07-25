import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("model/churn_model.pkl")  # update path as needed
# If using a scaler or encoder, load those too:
# scaler = joblib.load("model/scaler.pkl")
# encoder = joblib.load("model/encoder.pkl")

# Streamlit app UI
st.title("Customer Churn Prediction")

# User inputs
creditscore = st.number_input('Credit Score', min_value=0, step=1)
geography = st.text_input('Geography')
gender = st.text_input("Gender")
age = st.number_input("Age", min_value=0, max_value=120, step=1)
tenure = st.number_input("Tenure", min_value=0, max_value=10, step=1)
balance = st.number_input("Balance", min_value=0.0, step=1.0)
numofproducts = st.number_input("Number of Products", min_value=0, max_value=10, step=1)
hascrcard = st.number_input("Has Credit Card", min_value=0, max_value=1, step=1)
isactivemember = st.number_input("Is Active Member", min_value=0, max_value=1, step=1)
estimatedsalary = st.number_input("Estimated Salary", min_value=0.0, step=1.0)

if st.button('Predict'):
    # Convert input to dataframe
    input_data = pd.DataFrame([{
        'creditscore': creditscore,
        'geography': geography,
        'gender': gender,
        'age': age,
        'tenure': tenure,
        'balance': balance,
        'numofproducts': numofproducts,
        'hascrcard': hascrcard,
        'isactivemember': isactivemember,
        'estimatedsalary': estimatedsalary
    }])

    # Preprocess if needed (example only)
    # input_data['gender'] = input_data['gender'].map({'Male': 0, 'Female': 1})
    # input_data = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success('Prediction: Customer is **not** likely to leave the company')
    else:
        st.warning('Prediction: Customer is **likely** to leave the company')
