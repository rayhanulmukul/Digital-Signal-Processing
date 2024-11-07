# 1. Write a python program for generating a composite signal (you could use sine or cosine waves). The parameters including the signal frequencies of 40 Hz, 80 Hz, 160 Hz with the amplitudes of 10, 20, and 40 respectively, and the signal length should be limited to 512 in samples.
# 2. Plot the generated signal.
# 3. Do standard sampling by following the Nyquist rate.
# 4. Perform under sampling and over sampling too. Use Subplot function to show the original, sampled, under sampled, and over sampled signal.
# 5. Then perform N=512 point DFT, show the magnitude and phase spectrum.

import matplotlib.pyplot as plt
import numpy as np
import cmath as cm
import math

N = 512
def DFT(x):
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * n * k / N)
    return X

def phase(X_n):
    x_phase = []
    for i in X_n:
        phase = cm.phase(round(i.real) + round(i.imag) * 1j)
        x_phase.append(math.degrees(phase))
    return x_phase

t = np.arange(0, 1, 1/N)
x = 10 * np.sin(2 * np.pi * 40 * t) + 20 * np.sin(2 * np.pi * 80 * t) + 40 * np.sin(2 * np.pi * 160 * t)
t = t[:13] # 512/40 = 13
x = x[:13]
plt.subplot(2, 2, 1)
plt.plot(t, x)
plt.title("Composite Signal")
plt.xlabel("Time(s)")
plt.ylabel("Amplitude")
plt.grid()

N = 512
fs = 320 # Nyquist Rate = 2 * Fmax
t = np.arange(0, N/fs, 1/fs)
x = 10 * np.sin(2 * np.pi * 40 * t) + 20 * np.sin(2 * np.pi * 80 * t) + 40 * np.sin(2 * np.pi * 160 * t)
t = t[:8] # 512/40 = 13
x = x[:8]
plt.subplot(2, 2, 2)
plt.plot(t, x)
plt.title("Sampled Signal (Nquist Rate)")
plt.xlabel("n")
plt.ylabel("Amplitude")


# Over Sampling
N = 512
fs = 640 # Nyquist Rate = 2 * Fmax = * 2
t = np.arange(0, N/fs, 1/fs)
x = 10 * np.sin(2 * np.pi * 40 * t) + 20 * np.sin(2 * np.pi * 80 * t) + 40 * np.sin(2 * np.pi * 160 * t)
t = t[:16] # 512/40 = 13
x = x[:16]
plt.subplot(2, 2, 3)
plt.plot(t, x)
plt.title("Over Signal")
plt.xlabel("n")
plt.ylabel("Amplitude")

# Under Sampling
N = 512
fs = 160 # Nyquist Rate = 2 * Fmax = * 2
t = np.arange(0, N/fs, 1/fs)
x = 10 * np.sin(2 * np.pi * 40 * t) + 20 * np.sin(2 * np.pi * 80 * t) + 40 * np.sin(2 * np.pi * 160 * t)
t = t[:5] # 512/40 = 13
x = x[:5]
plt.subplot(2, 2, 4)
plt.plot(t, x)
plt.title("Over Signal")
plt.xlabel("n")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()

N = 512
fs = 320 # Nyquist Rate = 2 * Fmax
t = np.arange(0, N/fs, 1/N)
x_n = 10 * np.sin(2 * np.pi * 40 * t) + 20 * np.sin(2 * np.pi * 80 * t) + 40 * np.sin(2 * np.pi * 160 * t)
X_n = DFT(x_n)
plt.subplot(2, 1, 1)
plt.stem(np.abs(X_n))
plt.title("Magnitude Spectrum")
plt.xlabel("n")
plt.ylabel("Amplitude")


x_phase = phase(X_n)
plt.subplot(2, 1, 2)
plt.stem(x_phase, basefmt="")

plt.show()