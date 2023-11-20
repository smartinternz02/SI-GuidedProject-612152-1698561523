from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and scaler outside of the routes
with open('ada_boost_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('minmax.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST", "GET"])
def predict():
    # Reading the inputs given by the user
    input_feature = [float(x) for x in request.form.values()]
    print("length of input:", len(input_feature))

    features_values = [np.array(input_feature)]
    feature_names = ['Sex', 'Marital status', 'Age', 'Education', 'Income', 'Occupation', 'Settlement size']
    
    data = pd.DataFrame(features_values, columns=feature_names)
    scaled_data = scaler.transform(data)  # Use the scaler to transform the data

    # Predictions using the loaded model
    prediction = model.predict(scaled_data)
    print(prediction)

    if prediction == 0:
        prediction_text = "Not a potential customer"
    elif prediction == 1:
        prediction_text = "Potential customer"
    else:
        prediction_text = "Highly potential customer"

    return render_template("index.html", prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)