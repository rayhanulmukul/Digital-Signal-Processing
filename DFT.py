import numpy as np
import matplotlib.pyplot as plt

def signal(t):
    sin1 = np.sin(2*np.pi*1000*t)
    sin2 = 0.5*np.sin(2*np.pi*2000*t + 3*np.pi/4)
    return sin1 + sin2

t_max = 1/1000
t = np.linspace(0, t_max, 1000)
# t = np.arange(0, t_max, 1/(10*2000))

sin = signal(t)
plt.plot(t, sin)
plt.show()

# Sampling
fs = 8000  # 8kHz
ts = np.arange(0, 8/fs, 1/fs)
sin_sampled = signal(ts)
plt.stem(ts, sin_sampled)
plt.plot(t, sin, 'r')
plt.show()

# DFT
def dft(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for m in range(N):
        for n in range(N):
            X[m] += x[n]*np.exp(-2j*np.pi*m*n/N)
    return X

X = dft(sin_sampled)
f = np.arange(0, fs, fs/len(X))

# Amplitude spectrum
plt.stem(f, np.abs(X))
plt.show()

# Phase spectrum
# plt.stem(f, np.angle(X))
# plt.show()