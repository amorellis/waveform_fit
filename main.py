import os
import pandas as pd

# Importar arquivos


# Converter DF para listas de cada sinal


# Aplica FFT


# Identifica os 50 harmonicos


# Salva em uma planilha


files = os.listdir("files")

for file_name in files:
    file = pd.read_csv("files/" + file_name, sep = ";", skiprows = 1)
    
    time = file[file.columns[0]].values.tolist()
    tensao_1 = file[file.columns[1]].values.tolist()
    tensao_2 = file[file.columns[2]].values.tolist()
    tensao_3 = file[file.columns[3]].values.tolist()

