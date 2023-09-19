import pandas as pd
import numpy as np
from numpy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
from  scipy.signal import detrend


filepath = '20230315-0002.csv'

# Desvio de onda

# def waveDeviation(filepath):
    
    

def generate_sine_wave(fundamental_freq, sampling_rate, num_harmonics, num_samples):
    time = np.arange(num_samples) / sampling_rate

    freqs = fundamental_freq * np.arange(1, num_harmonics+1)
    amplitudes = 1 / freqs
    phases = np.zeros(num_harmonics)

    signal = np.zeros(num_samples)
    for i in range(num_harmonics):
        signal += amplitudes[i] * np.sin(2 * np.pi * freqs[i] * time + phases[i])

    return signal, time
        


freq_fund = 60
sample_rate = 2e6
num_harmo = 50
samples = 400000

u, t = generate_sine_wave(freq_fund, sample_rate, num_harmo, samples)

# data = pd.DataFrame()
 
# data = pd.read_csv(filepath, sep = ";", decimal = ",", skiprows = [1])

# # retira o componente DC de cada sinal
# t = data[data.columns[0]].values.tolist()
# u = list(detrend(data[data.columns[1]].values.tolist()))
# # v = list(detrend(data[data.columns[2]].values.tolist()))


t = t[0:samples]
u = u[0:samples]
# v = v[0:samples]

plt.plot(t, u, label="U")
# plt.plot(t, v, label="V")
plt.legend()
plt.show()

# Coloca o inicio do array tempo para zero 0 e converte de ms para s
t = list(map(lambda x : ((x - t[0])), t))

dt = t[1]
sample_rate = 1/dt

def fft(signal, dt):
    fft = rfft(signal)*(2/len(signal))
    frequency = rfftfreq(samples, dt)
    
    amplitude = list(np.abs(fft))
    phase = list(np.angle(fft) + np.pi/2)
    
    
    return amplitude, phase, frequency

def CreateSinWave(signal, time):
    
    
    amplitude, phase, frequency = fft(signal, dt)
    
    # Frequencia de amostragem
    
    
    print('dt: {:.7f} s, freq: {:.0f} Hz, df: {:.2f} Hz'.format(dt, sample_rate, frequency[1]))
    

    idx = np.argmax(amplitude)
    dominant_frequency = frequency[idx]
    dominant_amplitude = amplitude[idx]
    dominant_phase = phase[idx]
    
    # Generate a sine wave with the same frequency and desired amplitude and phase
    sine_wave = list(map(lambda x:dominant_amplitude*np.sin((2 * np.pi * dominant_frequency * x) + dominant_phase), time))
    
    print('Index: {:.2f}, Frequencia: {:.2f} Hz, Amplitude: {:.2f} V, Fase: {:.2f} deg'.format(idx, dominant_frequency, dominant_amplitude, (dominant_phase*180/np.pi)))
    

    return sine_wave, dominant_amplitude, dominant_phase, dominant_frequency

def HalfCycle(sine_wave, signal):
    idx_inf = 0
    idx_sup = 0
    i = 0
    
    while ((idx_sup == 0) or (idx_inf == 0)):
        if ((sine_wave[i] <= 0) and (sine_wave[i + 1] > 0)):
            idx_inf = i
            
        
        if ((sine_wave[i] > 0) and (sine_wave[i + 1] <= 0)):
            idx_sup = i
            
            
        i += 1
        
    if (idx_sup < idx_inf):
        dif = idx_inf - idx_sup
        idx_sup = idx_inf + dif
            
    
    return idx_inf, idx_sup


def WaveDeviation(signal, sine_wave, idx_inf, idx_sup, EOM):
    
    dE = 0
    
    for i in range(idx_inf, idx_sup):
        dif = np.abs(signal[i] - sine_wave[i])
        if (dif > dE):
            dE = dif
            
            
    Fdev = dE/EOM
    
    return dE, Fdev


def FIT(signal, dt):
    
    TN = [0.5, 0, 30, 0, 225, 400, 650, 0, 1320, 0, 2260, 2760, 3360, 0, 4350, 0, 
          5100, 5400, 5630, 0 ,6050, 0, 6370, 6650, 6680, 0, 6970, 0 ,7320, 7570, 
          7820, 0, 8330, 0, 8330, 9080, 9330, 0, 9840, 0, 10340, 0, 10600, 0, 0, 0, 
          10210, 0, 9820, 9670]  
    
    amplitude, phase, frequency = fft(signal, dt)
    
    idx = np.argmax(amplitude)
    
    Erms = (2 ** 0.5) * amplitude[idx]
    
    E = []
    
    pace = round(60.0/frequency[1])

    
    for i in range(pace, len(amplitude), pace):
        
        E.append(amplitude[i])
        
    
    Etif = 0
    
    for i in range(0, 50):
        Etif += (E[i] * TN[i]) ** 2
        print("{:.0f} - {:.10f} V - {:.1f}".format((i + 1), E[i], TN[i]))
        
    Etif = (Etif) ** 0.5
    
    
    TIF = Etif / Erms
    
    print('EOM: {:.2f}, dE: {:.2f}, Fdev: {:.2f}, TIF: {:.2f}'.format(EOM, dE, Fdev, TIF))
    
    return TIF
        

    
    
sin, EOM, dominant_phase, dominant_frequency = CreateSinWave(u, t)

idx_inf, idx_sup = HalfCycle(sin, u)

plt.plot(t[idx_inf:idx_sup], sin[idx_inf:idx_sup], label="Sine")
plt.plot(t[idx_inf:idx_sup], u[idx_inf:idx_sup], label="Signal")
plt.legend()
plt.show()

dE, Fdev = WaveDeviation(u, sin, idx_inf, idx_sup, EOM)

TIF = FIT(u, dt)

amplitude, phase, frequency = fft(u, dt)


plt.subplot(2, 1, 1)
plt.plot(frequency, amplitude)
plt.ylabel('Amplitude [V]')
plt.xlim([0, 3100])
plt.subplot(2, 1, 2)
plt.plot(frequency, phase)
plt.ylabel('Phase [rad]')
plt.xlabel('Frequency [Hz]')
plt.xlim([0, 3100])
plt.show()


plt.plot(t, sin, label='Seno')
plt.plot(t, u, label='Sinal')
plt.xlabel("Time (s)")
plt.legend()
plt.show()











    
    
    
    
    
