import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Classe
class Sensor:

    # Método construtor
    def __init__(self, sensor1, sensor2, sensor3, sensor4, sensor5):
        self.sensor1 = sensor1
        self.sensor2 = sensor2
        self.sensor3 = sensor3
        self.sensor4 = sensor4
        self.sensor5 = sensor5

    # Método de preparação dos dados ---> repetição dos processos de processamento dos dados para o treino (pipilene de tratamento dos dados da webapp)
    def prepare(self):
        
        #Remoção dos valores NaN dos ultimos instantes de tempo
        dados_s1_NaN_remove = self.sensor1[:,:200]
        dados_s2_NaN_remove = self.sensor2[:,:200]
        dados_s3_NaN_remove = self.sensor3[:,:200]

        #Conversão em pandas
        df_dados_s4 = pd.DataFrame(self.sensor4)
        df_dados_s5 = pd.DataFrame(self.sensor5)

        # Substituir NaN por 50. no sensor S4
        df_dados_s4_preenchidos = df_dados_s4.fillna(50.)

        # Interpolar para preencher os valores NaN no Sensor S5
        df_dados_s5_interpolados = df_dados_s5.interpolate(method='linear', axis=1)

        df_df_dados_s5_preenchidos = df_dados_s5_interpolados.fillna(0.)

        # DataFrame para um array numpy
        dados_s4_NaN_remove  = df_dados_s4_preenchidos.values
        dados_s5_NaN_remove  = df_df_dados_s5_preenchidos.values

        sensores_NaN_removidos = [dados_s1_NaN_remove , dados_s2_NaN_remove , dados_s3_NaN_remove , dados_s4_NaN_remove , dados_s5_NaN_remove]

        max_value_s1 = 5.0
        min_value_s1 = -0.500583548307494

        max_value_s2 = 5
        min_value_s2 = -1.239735330040627

        max_value_s3 = 5.0
        min_value_s3 = -1.557491418788932

        max_value_s4 = 50
        min_value_s4 = 0

        max_value_s5 = 139.99831052645226
        min_value_s5 = -19.964571047700915

        dados_s1_N = (sensores_NaN_removidos[0] - min_value_s1) / (max_value_s1 - min_value_s1)
        dados_s2_N = (sensores_NaN_removidos[1] - min_value_s2) / (max_value_s2 - min_value_s2)
        dados_s3_N = (sensores_NaN_removidos[2] - min_value_s3) / (max_value_s3 - min_value_s3)
        dados_s4_N = (sensores_NaN_removidos[3] - 0) / (max_value_s4 - 0)
        dados_s5_N = (sensores_NaN_removidos[4] - min_value_s5) / (max_value_s5 - min_value_s5)

        dados_s1_N_reshaped = dados_s1_N.reshape((1, 200, 1))
        dados_s2_N_reshaped = dados_s2_N.reshape((1, 200, 1))
        dados_s3_N_reshaped = dados_s3_N.reshape((1, 200, 1))
        dados_s4_N_reshaped = dados_s4_N.reshape((1, 200, 1))
        dados_s5_N_reshaped = dados_s5_N.reshape((1, 200, 1))

        stacked_data = np.concatenate([dados_s1_N_reshaped, dados_s2_N_reshaped, dados_s3_N_reshaped, dados_s4_N_reshaped, dados_s5_N_reshaped], axis=-1)

        return stacked_data

    # Método para as previsões
    def predict(self, stacked_data):
        model = load_model('modelo/best_model.keras')
        predicted_classification = model.predict(stacked_data)
        value = predicted_classification
        return value


