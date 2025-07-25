from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load your trained ML model
model = joblib.load('new_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    

    # Get the data from the request
    data = request.json 
    creditscore = int(data['creditscore'])
    geography = data['geography']
    gender = data['gender']
    age = int(data['age']) 
    tenure = int(data['tenure'])
    balance = int(data['balance'])
    numofproducts = int(data['numofproducts'])
    hascrcard = int(data['hascrcard'])
    isactivemember = int(data['isactivemember'])
    estimatedsalary = int(data['estimatedsalary'])

    # List of columns and their corresponding values
    columns = ['creditscore', 'geography', 'gender', 'age', 'tenure', 'balance', 'numofproducts', 'hascrcard', 'isactivemember', 'estimatedsalary']
    values = [[creditscore, geography, gender, age, tenure, balance, numofproducts, hascrcard, isactivemember, estimatedsalary]]  # Wrap in a list to create a 2D array

    # Create the DataFrame with a single row of input data
    features = pd.DataFrame(values, columns=columns)
    print(features)


    # Make a prediction
    prediction = model.predict(features)
    return jsonify({'prediction': str(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
