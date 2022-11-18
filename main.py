import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importar arquivos

files = os.listdir("files")


for file in files:

    # Abre o arquivo e converte para dataframe
    
    data = pd.DataFrame()
    
    data = pd.read_csv("files/" + file, sep = ";", decimal = ",", skiprows = 1)
    
    
    # Converte os sinais para listas
    
    t = data[data.columns[0]].values.tolist()
    u = data[data.columns[1]].values.tolist()
    v = data[data.columns[2]].values.tolist()
    w = data[data.columns[3]].values.tolist()
    
    
    # Plota os sinais
    
    plt.plot(t[0:1000], u[0:1000], t[0:1000], v[0:1000], t[0:1000], w[0:1000])
    plt.show()
    
    
    # Faz a FFT dos sinais
    
    samples = len(t)
    
    
    freq = np.fft.rfftfreq(samples, t[1])
    
    n = len(freq)
    U = np.abs(np.fft.rfft(u))
    V = np.abs(np.fft.rfft(v))
    W = np.abs(np.fft.rfft(w))
    
    
    
    # Plota o espectro de frequencia dos sinais
    
    plt.plot(freq, U)
    plt.show()
    
    
    # Identificação dos harmonicos em cada FFT
    
    harm_u = []
    harm_v = []
    harm_w = []
    
    for i in range(1, len(freq)):
        if U[i -1] < 0.1 and U[i] > 0.1:
            harm_u.append(U[i])
                  
    for i in range(1, len(freq)):
        if V[i -1] < 0.1 and V[i] > 0.1:
            harm_v.append(V[i])
            
    for i in range(1, len(freq)):
        if W[i -1] < 0.1 and W[i] > 0.1:
            harm_w.append(W[i])
            
            
    harm_u = list(map(lambda x:((x**2/2)**0.5)/n, harm_u))
    harm_v = list(map(lambda x:((x**2/2)**0.5)/n, harm_v))
    harm_w = list(map(lambda x:((x**2/2)**0.5)/n, harm_w))
            
    # Salva em CSV o resultado
    
    resultado = pd.DataFrame()
    
    resultado['Index'] = range(1, 51)
    resultado['U'] = harm_u[0:50]
    resultado['V'] = harm_v[0:50]
    resultado['W'] = harm_w[0:50]
    
    resultado.to_csv("results/" + file, sep = ";", decimal = ",", index = False)





