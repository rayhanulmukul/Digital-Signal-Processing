import matplotlib.pyplot as plt
import numpy as np
import cmath as cm
import math

# Complex Signal
t = np.arange(0, 0.1, 0.0001)
x = 10 * np.sin(2 * np.pi * 40 * t) + 20 * np.sin(2 * np.pi * 80 * t) + 40 * np.sin(2 * np.pi * 160 * t)
plt.subplot(3, 2, 1)
plt.plot(t, x)
plt.title("Complex Signal")
plt.xlabel("Time(s)")
plt.ylabel("Amplitude")

# Sampled Signal
fs = 512
t = np.arange(0, 1, 1/fs)
x = 10 * np.sin(2 * np.pi * 40 * t) + 20 * np.sin(2 * np.pi * 80 * t) + 40 * np.sin(2 * np.pi * 160 * t)
plt.subplot(3, 2, 2)
t = t[:13]
x = x[:13]
plt.plot(t, x)
plt.title("Sampled Signal")
plt.xlabel("Time(s)")
plt.ylabel("Amplitude")

# Nyquist Rate Signal
N = 512
Fs = 320
t = np.arange(0, N/Fs, 1/Fs)
x = 10 * np.sin(2 * np.pi * 40 * t) + 20 * np.sin(2 * np.pi * 80 * t) + 40 * np.sin(2 * np.pi * 160 * t)
plt.subplot(3, 2, 3)
t = t[:8]
x = x[:8]
plt.plot(t, x)
plt.title("Nyquist Rate Signal")
plt.xlabel("Time(s)")
plt.ylabel("Amplitude")

# Over Sampled Signal
N = 512
Fs = 640
t = np.arange(0, N/Fs, 1/Fs)
x = 10 * np.sin(2 * np.pi * 40 * t) + 20 * np.sin(2 * np.pi * 80 * t) + 40 * np.sin(2 * np.pi * 160 * t)
plt.subplot(3, 2, 4)
t = t[:16]
x = x[:16]
plt.plot(t, x)
plt.title("Over Sampled Signal")
plt.xlabel("Time(s)")
plt.ylabel("Amplitude")

# Under Sampled Signal
N = 512
Fs = 160
t = np.arange(0, N/Fs, 1/Fs)
x = 10 * np.sin(2 * np.pi * 40 * t) + 20 * np.sin(2 * np.pi * 80 * t) + 40 * np.sin(2 * np.pi * 160 * t)
plt.subplot(3, 2, 5)
t = t[:5]
x = x[:5]
plt.plot(t, x)
plt.title("Under Sampled Signal")
plt.xlabel("Time(s)")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()


def dft(x, N):
    y = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            y[k] += x[n] * np.exp(-2j * np.pi * n * k / N)
    return y

def phase(x_n):
    y = []
    for i in x_n:
        phase1 = cm.phase(round(i.real) + round(i.imag) * 1j)
        y.append(math.degrees(phase1))
    return y

def idft(x_n, N):
    y = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            y[k] += x_n[n] * np.exp(2j * np.pi * k * n / N)
        y[k] = y[k] / N
    return y

# Magnitude Spectrum
fs = 512
t = np.arange(0, 1, 1/fs)
x = 10 * np.sin(2 * np.pi * 40 * t) + 20 * np.sin(2 * np.pi * 80 * t) + 40 * np.sin(2 * np.pi * 160 * t)

x_n = dft(x, len(x))
# plt.subplot(2, 1, 1)
# t = t[:13]
# x_n = x_t[:13]
# plt.stem(t, x_n)
# plt.title("DFT")
# plt.xlabel("n")
# plt.ylabel("Amplitude")

plt.subplot(3, 1, 1)
plt.stem(np.abs(x_n))
plt.title("Magnitude Spectrum")
plt.xlabel("n")
plt.ylabel("Amplitude")

# Phase Spectrum
x_phase = phase(x_n)
plt.subplot(3, 1, 2)
plt.stem(x_phase)
plt.title("Phase Spectrum")
plt.xlabel("Angle (in Degrees)")
plt.ylabel("Amplitude")

# Reconstructed Signal(IDFT)
x_idft = idft(x_n, len(x_n))
plt.subplot(3, 1, 3)
plt.plot(x_idft.real)
plt.stem(x_idft.real)
plt.title("Reconstructed Signal(IDFT)")
plt.xlabel("n")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()

