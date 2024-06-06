# Given continuous signal x(t) = sin(2*pi*1000*t) + 0.5*sin(2*pi*2000*t + 3*pi/4);
import numpy as np
import matplotlib.pyplot as plt

def signal(t):
    return np.sin(2*np.pi*1000*t) + 0.5*np.sin(2*np.pi*2000*t + 3*np.pi/4)

t_star = 1/1000
t = np.linspace(0, t_star, 1000) # 1000 samples between 0 and 0.01
# t = np.arrenge(0, 0.01, 0.00001)
x = signal(t)
plt.plot(t, x)
plt.title('Original Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.show()

# Sampling the signal
fs = 8000 #8kHz
ts = 1/fs
n = np.arange(0, 8/fs, ts) # 8 samples between 0 and 0.001
x_sampled = signal(n)
plt.stem(n, x_sampled)
plt.plot(t, x, 'r')
plt.title('Sampled Signal')
plt.show()

# Compute the DFT
def dft(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for m in range(N):
        for n in range(N):
            X[m] += x[n]*np.exp(-2j * np.pi*m*n/N)
    return X

X = dft(x_sampled)
f = np.arange(0, fs, fs/len(X))

# Amplitude Spectrum
plt.stem(f, np.abs(X))
plt.title('DFT of the Signal')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.show()

# Phase Spectrum
plt.stem(f, np.angle(X))
plt.title('DFT of the Signal')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Phase')
plt.show()