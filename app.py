from flask import Flask, request
from flask import render_template
import tensorflow as tf
import numpy as np
import pandas as pd
from tools.sensor import Sensor

# Cria a app
app = Flask(__name__)

# Página de entrada
@app.route("/")
def index():
    result = None
    return render_template("index.html", result = result)

# Página com resultado da previsão 
@app.route("/estimate", methods = ["POST"])


# Realiza o processamento dos dados e classifica as séries temporais
def estimate():
    
    sensor1_array = np.load( request.files['sensor1'])
    sensor2_array = np.load( request.files['sensor2'])
    sensor3_array = np.load( request.files['sensor3'])
    sensor4_array = np.load( request.files['sensor4'])
    sensor5_array = np.load( request.files['sensor5'])

    sensores = Sensor(sensor1_array, sensor2_array, sensor3_array, sensor4_array, sensor5_array)
    sensors_data_processed = sensores.prepare()
    prediction = sensores.predict(sensors_data_processed)

    # Encontra o índice da classe com valor 1 no vetor de saída
    predicted_class_index = tf.argmax(prediction, axis = 1)

    if predicted_class_index[0] == 0:
        predicted_class_name = 'Classe A'
    elif predicted_class_index[0] == 1:
        predicted_class_name = 'Classe B'
    elif predicted_class_index[0] == 2:
        predicted_class_name = 'Classe C'
    elif predicted_class_index[0] == 3:
        predicted_class_name = 'Classe D'
    elif predicted_class_index[0] == 4:
        predicted_class_name = 'Classe E'

    return render_template('index.html', result = predicted_class_name)

# Executa a app
if __name__ == "__main__":
    app.run(host = 'localhost', port = 3000, debug = True)
