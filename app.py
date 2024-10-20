from flask import Flask, render_template, request, flash
import pandas as pd
import numpy as np
import logging
import joblib
import json
import sys
import os

current_dir = os.path.dirname(__file__)

app = Flask(__name__, static_folder='static', template_folder='template')
app.secret_key = 'supersecretkey'

app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

def ValuePredictor(data=pd.DataFrame):
    model_name = 'model/SVM_model.pkl'
    model_dir = os.path.join(current_dir, model_name)
    loaded_model = joblib.load(open(model_dir, 'rb'))
    result = loaded_model.predict(data)
    return result[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get the data from form
            battery_power = float(request.form['battery_power'])
            blue = bool(int(request.form['blue']))
            clock_speed = float(request.form['clock_speed'])
            dual_sim = bool(int(request.form['dual_sim']))
            fc = float(request.form['fc'])
            four_g = bool(int(request.form['four_g']))
            int_memory = float(request.form['int_memory'])
            m_dep = float(request.form['m_dep'])
            mobile_wt = float(request.form['mobile_wt'])
            n_cores = float(request.form['n_cores'])
            pc = float(request.form['pc'])
            px_height = float(request.form['px_height'])
            px_width = float(request.form['px_width'])
            ram = float(request.form['ram'])
            sc_h = float(request.form['sc_h'])
            sc_w = float(request.form['sc_w'])
            talk_time = float(request.form['talk_time'])
            three_g = bool(int(request.form['three_g']))
            touch_screen = bool(int(request.form['touch_screen']))
            wifi = bool(int(request.form['wifi']))

        except ValueError:
            # flash("enter error")
            return render_template('error.html')

        schema_name = 'data/columns_set.json'
        schema_dir = os.path.join(current_dir, schema_name)
        with open(schema_dir, 'r') as f:
            cols = json.loads(f.read())
        schema_cols = cols['data_columns']

        schema_cols['blue'] = blue
        schema_cols['dual_sim'] = dual_sim
        schema_cols['four_g'] = four_g
        schema_cols['three_g'] = three_g
        schema_cols['touch_screen'] = touch_screen
        schema_cols['wifi'] = wifi

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

        df = pd.DataFrame(
            data={k: [v] for k, v in schema_cols.items()},
            dtype=float
        )

        result = ValuePredictor(data=df)

        if int(result) == 1:
            prediction = 'Dear your device is classified as medium cost:1!'
        elif int(result) == 2:
            prediction = 'Dear your device is classified as high cost:2!'
        elif int(result) == 3:
            prediction = 'Dear your device is classified as very high cost:3!'
        else:
            prediction = 'Dear your device is classified as low cost!:0'

        return render_template('prediction.html', prediction=prediction)

    else:
        return render_template('error.html')

if __name__ == '__main__':
    app.run(debug=True)
