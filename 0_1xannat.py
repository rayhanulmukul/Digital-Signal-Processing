import numpy as np
import matplotlib.pyplot as plt
import cmath as cm
import math

N = 512
Fs = 320

def signal(t):
    # return np.sin(2 * np.pi * 1000 * t) + 0.5 * np.sin(2 * np.pi * 2000 * t + (3 * np.pi)/4)
    component1 = 10*np.sin(2*np.pi*40*t)
    component2 = 20*np.sin(2*np.pi*80*t)
    component3 = 40*np.sin(2*np.pi*160*t)
    return component1+component2+component3


def DFT():
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += signal(n/Fs)*np.exp(-2j*np.pi*n*k/N)
    return X


def phase(complex):
    # return cm.phase(round(complex.real) + round(complex.imag) * 1j)
    real = round(complex.real)
    imag = round(complex.imag)
    angle = cm.phase(real + imag * 1j)
    angle = np.degrees(angle)
    return angle


plt.subplot(2, 2, 1)
t = np.arange(0, 0.01, 0.001)
x = signal(t)
plt.plot(t, x)
plt.title("Original Signal")
plt.xlabel("Time(s)")
plt.ylabel("Amplitude")
plt.grid()

# Sampling
t = np.arange(0, 0.5, 1/N)
x = signal(t)
plt.subplot(2, 2, 2)
plt.stem(t, x, basefmt="")
plt.title("Sampled Signal(512)")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.grid()


t = np.arange(0, 0.1, 1/Fs)
x = signal(t)
plt.subplot(2, 2, 3)
plt.plot(t, x, '-r')
plt.stem(t, x, basefmt="")
plt.title("Sampled Signal(Nq)")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()
plt.show()

X = DFT()

plt.subplot(2, 1, 1)
plt.stem(np.abs(X)**2, basefmt="")
plt.title("Magintude Spectrum")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.grid()

Phase = [phase(num) for num in X]
plt.subplot(2, 1, 2)
plt.stem(Phase, basefmt="")
plt.title("Phase Spectrum")
plt.xlabel("n")
plt.ylabel("Degree")
plt.grid()


plt.tight_layout()
plt.show()
