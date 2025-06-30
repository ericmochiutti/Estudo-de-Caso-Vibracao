from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from app.tools.sensor import Sensor
import uvicorn
import os

# Cria a app
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas as origens
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos os métodos
    allow_headers=["*"],  # Permite todos os headers
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


# Página de entrada
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "result": None}
    )


# Página com resultado da previsão
@app.post("/estimate")
async def estimate(
    request: Request,
    sensor1: UploadFile = File(...),
    sensor2: UploadFile = File(...),
    sensor3: UploadFile = File(...),
    sensor4: UploadFile = File(...),
    sensor5: UploadFile = File(...),
):
    sensor1_array = np.load(sensor1.file)
    sensor2_array = np.load(sensor2.file)
    sensor3_array = np.load(sensor3.file)
    sensor4_array = np.load(sensor4.file)
    sensor5_array = np.load(sensor5.file)

    sensores = Sensor(
        sensor1_array, sensor2_array, sensor3_array, sensor4_array, sensor5_array
    )
    sensors_data_processed = sensores.prepare()
    prediction = sensores.predict(sensors_data_processed)

    # Encontra o índice da classe com valor 1 no vetor de saída
    predicted_class_index = tf.argmax(prediction, axis=1)

    if predicted_class_index[0] == 0:
        predicted_class_name = "Classe A"
    elif predicted_class_index[0] == 1:
        predicted_class_name = "Classe B"
    elif predicted_class_index[0] == 2:
        predicted_class_name = "Classe C"
    elif predicted_class_index[0] == 3:
        predicted_class_name = "Classe D"
    elif predicted_class_index[0] == 4:
        predicted_class_name = "Classe E"

    return templates.TemplateResponse(
        "index.html", {"request": request, "result": predicted_class_name}
    )


# Executa a app
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=3000)
