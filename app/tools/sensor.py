import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
import os


# Classe
class Sensor:
    # Método construtor
    def __init__(self, sensor1, sensor2, sensor3, sensor4, sensor5):
        self.sensor1 = sensor1
        self.sensor2 = sensor2
        self.sensor3 = sensor3
        self.sensor4 = sensor4
        self.sensor5 = sensor5

        base_dir = os.path.abspath(os.path.dirname(__file__))
        self.model_path = os.path.abspath(os.path.join(base_dir, "..", "..", "modelo"))
        self.scalers_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "modelo", "scalers"
        )

    # Método de preparação dos dados
    def prepare(self):
        # Garantir que todos os sensores tenham shape (1, 200)
        dados_s1 = self.sensor1[:, :200]
        dados_s2 = self.sensor2[:, :200]
        dados_s3 = self.sensor3[:, :200]

        # Preencher sensor 4 com 50. onde houver NaN
        dados_s4 = pd.DataFrame(self.sensor4[:, :200]).fillna(50.0).values

        # Interpolação linear e preenchimento com 0 no sensor 5
        dados_s5 = (
            pd.DataFrame(self.sensor5[:, :200])
            .interpolate(method="linear", axis=1)
            .fillna(0.0)
            .values
        )

        # Carregar scalers
        def carregar_scaler(nome):
            with open(os.path.join(self.scalers_path, nome), "rb") as f:
                return pickle.load(f)

        scaler1 = carregar_scaler("scaler_sensor_1.pkl")
        scaler2 = carregar_scaler("scaler_sensor_2.pkl")
        scaler3 = carregar_scaler("scaler_sensor_3.pkl")
        scaler4 = carregar_scaler("scaler_sensor_4.pkl")
        scaler5 = carregar_scaler("scaler_sensor_5.pkl")

        # Normalização com reshape para compatibilidade com o scaler
        dados_s1_N = scaler1.transform(dados_s1.reshape(-1, 1)).reshape(dados_s1.shape)
        dados_s2_N = scaler2.transform(dados_s2.reshape(-1, 1)).reshape(dados_s2.shape)
        dados_s3_N = scaler3.transform(dados_s3.reshape(-1, 1)).reshape(dados_s3.shape)
        dados_s4_N = scaler4.transform(dados_s4.reshape(-1, 1)).reshape(dados_s4.shape)
        dados_s5_N = scaler5.transform(dados_s5.reshape(-1, 1)).reshape(dados_s5.shape)

        # Reshape para (1, 200, 1)
        reshape_3d = lambda x: x.reshape((1, 200, 1))

        dados_s1_N = reshape_3d(dados_s1_N)
        dados_s2_N = reshape_3d(dados_s2_N)
        dados_s3_N = reshape_3d(dados_s3_N)
        dados_s4_N = reshape_3d(dados_s4_N)
        dados_s5_N = reshape_3d(dados_s5_N)

        # Stack final para shape (1, 200, 5)
        stacked_data = np.concatenate(
            [dados_s1_N, dados_s2_N, dados_s3_N, dados_s4_N, dados_s5_N], axis=-1
        )

        return stacked_data

    def predict(self, stacked_data):
        model_path = os.path.abspath(os.path.join(self.model_path, "best_model.keras"))

        print(f"🔍 Carregando modelo de: {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado em: {model_path}")

        model = load_model(model_path)
        prediction = model.predict(stacked_data)
        return prediction
