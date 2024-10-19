# Flask
from flask import Flask, render_template, request
# Data manipulation
import pandas as pd
# Matrices manipulation
import numpy as np
# Script logging
import logging
# ML model
import joblib
# JSON manipulation
import json
# Utilities
import sys
import os

# Current directory
current_dir = os.path.dirname(__file__)

# Flask app
app = Flask(__name__, static_folder='static', template_folder='template')

# Logging
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

# Function
def ValuePredictor(data=pd.DataFrame):
    # Model name
    model_name = 'model/SVM_model.pkl'
    # Directory where the model is stored
    model_dir = os.path.join(current_dir, model_name)
    # Load the model
    loaded_model = joblib.load(open(model_dir, 'rb'))
    # Predict the data
    result = loaded_model.predict(data)
    return result[0]

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction page
@app.route('/prediction', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the data from form
        battery_power = request.form['battery_power']
        blue = request.form['blue']
        clock_speed = request.form['clock_speed']
        dual_sim = request.form['dual_sim']
        fc = request.form['fc']
        four_g = request.form['four_g']
        int_memory = request.form['int_memory']
        m_dep = request.form['m_dep']
        mobile_wt = request.form['mobile_wt']
        n_cores = request.form['n_cores']
        pc = request.form['pc']
        px_height = request.form['px_height']
        px_width = request.form['px_width']
        ram = request.form['ram']
        sc_h = request.form['sc_h']
        sc_w = request.form['sc_w']
        talk_time = request.form['talk_time']
        three_g = request.form['three_g']
        touch_screen = request.form['touch_screen']
        wifi = request.form['wifi']

        # Load template of JSON file containing columns name
        # Schema name
        schema_name = 'data/columns_set.json'
        # Directory where the schema is stored
        schema_dir = os.path.join(current_dir, schema_name)
        with open(schema_dir, 'r') as f:
            cols = json.loads(f.read())
        schema_cols = cols['data_columns']

        # Parse the categorical columns
        # Column of Bluetooth
        schema_cols['blue'] = blue
        # Column of dual SIM
        schema_cols['dual_sim'] = dual_sim
        # Column of 4G support
        schema_cols['four_g'] = four_g
        # Column of 3G support
        schema_cols['three_g'] = three_g
        # Column of touch screen support
        schema_cols['touch_screen'] = touch_screen
        # Column of wifi support
        schema_cols['wifi'] = wifi

        # Parse the numerical columns
        schema_cols['battery_power'] = battery_power
        schema_cols['clock_speed'] = clock_speed
        schema_cols['fc'] = fc
        schema_cols['int_memory'] = int_memory
        schema_cols['m_dep'] = m_dep
        schema_cols['mobile_wt'] = mobile_wt
        schema_cols['n_cores'] = n_cores
        schema_cols['pc'] = pc
        schema_cols['px_height'] = px_height
        schema_cols['px_width'] = px_width
        schema_cols['ram'] = ram
        schema_cols['sc_h'] = sc_h
        schema_cols['sc_w'] = sc_w
        schema_cols['talk_time'] = talk_time

        # Convert the JSON into data frame
        df = pd.DataFrame(
            data={k: [v] for k, v in schema_cols.items()},
            dtype=float
        )

        # Create a prediction
        print(df.dtypes)
        result = ValuePredictor(data=df)

        # Determine the output
        if int(result) == 1:
            prediction = 'Dear your device is classified as medium cost!'
        elif int(result) == 2:
            prediction = 'Dear your device is classified as high cost!'
        elif int(result) == 3:
            prediction = 'Dear your device is classified as very high cost!'
        else:
            prediction = 'Dear your device is classified as low cost!'

        # Return the prediction
        return render_template('prediction.html', prediction=prediction)

    # Something error
    else:
        # Return error
        return render_template('error.html')

if __name__ == '__main__':
    app.run(debug=True)
