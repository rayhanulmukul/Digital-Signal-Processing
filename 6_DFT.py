import matplotlib.pyplot as plt
import numpy as np
import cmath as cm
import math
plt.close('all')

def dft(sin_wave, N):
    out = np.zeros(N, dtype=np.complex128) 
    for m in range(N):
        for n in range(N):
            out[m] += sin_wave[n] * np.exp(-2j * np.pi * m * n / N)

    return out

def inverseDft(x_dft, N):
    out = np.zeros(N, dtype=np.complex128)
    for n in range(N):
        for m in range(N):
            out[n] += x_dft[m] * np.exp(2j * np.pi * n * m / N)
        out[n] = out[n] / N
    
    return out

def phase(x_dft, N):
    out = []
    for i in x_dft:
        phase = cm.phase(round(i.real) + round(i.imag) * 1j)
        out.append(math.degrees(phase))

    return out

# Input Signal
# x(t) = sin(2π⋅1000⋅t) + 0.5sin(2π⋅2000⋅t + 3π/4)
N = 8
t = np.arange(0, 0.001, 0.00001)
sin_wave = np.sin(2 * np.pi * 1000 * t) + 0.5 * np.sin(2 * np.pi * 2000 * t + (3 * np.pi)/4)
plt.subplot(3, 2, 1)
plt.plot(t, sin_wave)
plt.title("Input Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid()

# Sampled Signal
Fs = 8000
N = 8
t = np.arange(0, N/Fs, 1 / Fs)
sin_wave = np.sin(2 * np.pi * 1000 * t) + 0.5 * np.sin(2 * np.pi * 2000 * t + (3 * np.pi)/4)
plt.subplot(3, 2, 2)
plt.stem(sin_wave)
plt.plot(sin_wave)
plt.title("Sampled Signal")
plt.xlabel("n (Sample Index)")
plt.ylabel("Amplitude")
plt.grid()

# Amplitude Spectrum
x_dft = dft(sin_wave, N)
plt.subplot(3, 2, 3)
plt.stem(np.abs(x_dft))
plt.title("x_dftnitude Spectrum")
plt.xlabel("m (kHz)")
plt.ylabel("Amplitude")

# x_dftnitude Spectrum
x_phase = phase(x_dft, N)
plt.subplot(3, 2, 4)
plt.stem(x_phase)
plt.title("x_dftnitude Spectrum")
plt.xlabel("m (kHz)")
plt.ylabel("Angle (in Degrees)")

# Power Spectrum
plt.subplot(3, 2, 5)
plt.stem(np.abs(x_dft) ** 2)
plt.title("Power Spectrum")
plt.xlabel("m (kHz)")
plt.ylabel("Amplitude")

# Reconstructed Signal
inv_dft = inverseDft(x_dft, N)
plt.subplot(3, 2, 6)
plt.stem(inv_dft.real)
plt.plot(inv_dft.real)
plt.title("Reconstructed Signal")
plt.xlabel("n")
plt.ylabel("Amplitude")



plt.tight_layout()
plt.show()
