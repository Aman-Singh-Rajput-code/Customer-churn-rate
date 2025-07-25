import streamlit as st
import requests

# Streamlit app UI
st.title("ML Model with Flask API")

# User inputs
creditscore = st.number_input('Credit Score',min_value=0,step=1)
geography = st.text_input('Geography')
gender = st.text_input("Gender")
age = st.number_input("Age",min_value=0,max_value=120,step=1)
tenure = st.number_input("Tenure",min_value=0,max_value=10,step=1)
balance = st.number_input("Balance",min_value=0,step=1)
numofproducts = st.number_input("Number of Productts",min_value=0,max_value=10,step=1)
hascrcard = st.number_input("Has Credit Card",min_value=0,max_value=1,step=1)
isactivemember = st.number_input("Is Active Member",min_value=0,max_value=1,step=1)
estimatedsalary = st.number_input("Estimated Salary",min_value=0,step=1)


if st.button('Predict'):
    # Data to send to Flask API
    data = {'creditscore': creditscore, 
            'geography': geography,
            'gender': gender,
            'age':age,
            'tenure':tenure,
            'balance':balance,
            'numofproducts':numofproducts,
            'hascrcard':hascrcard,
            'isactivemember':isactivemember,
            'estimatedsalary':estimatedsalary}
    

    # Send POST request to Flask API
    response = requests.post('http://localhost:5000/predict', json=data)

    if response.status_code == 200:
        result = response.json()
        prediction = int(result['prediction'])

        if prediction==1:
            st.write('Prediction: Customer is not likely to leave the company')
        else:
            st.write('Prediction: Customer is likely to leave the company')
    else:
        st.write('Error in prediction')
