# FIR Low Pass filter using Convolution

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import firwin

def convolve(x, h ):
    len_x = len(x)
    len_h = len(h)
    len_y = len_x + len_h - 1
    y = [0] * len_y

    for k in range(len_y):
        for n in range(len_x):
            if(k - n >= 0 and k - n < len_h):
                y[k] += x[n] * h[k - n]
    return y

fs = 1000
cutoff_freq = 100

# Number of filter coefficient
num_taps = 15

# List of coefficients
fir_coefficients = firwin(num_taps, cutoff_freq, fs=fs, window="hamming")

t = np.arange(0, 1, 1/fs)
x = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 200 * t)

t = t[t < 0.5]
x = x[:len(t)]

plt.subplot(2, 1, 1)
plt.plot(t, x)
plt.title('x(t) = sin(2π.50t + 0.5sin(2π.200t))')
plt.xlabel('Time(s)')
plt.ylabel('Amplitude')

y = convolve(fir_coefficients, x)
t = np.linspace(0, 0.5, len(y))
plt.subplot(2, 1, 2)
plt.plot(t, y)
plt.title('y(n) = fir_filter(x(n))')
plt.xlabel('Time(s)')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()