import numpy as np
import matplotlib.pyplot as plt

# Clear all figures (equivalent to close all)
plt.close('all')

# Unit sample sequence
point = 21
unit_sample = np.zeros(point)
unit_sample[10] = 1  # index 10 corresponds to 0 in the -10 to 10 range
n = np.arange(-10, 11)
plt.subplot(3, 3, 1)
plt.stem(n, unit_sample, basefmt=" ")
plt.title("Unit sample sequence")

# Unit step sequence
unit_step = np.concatenate((np.zeros(10), np.ones(11)))
plt.subplot(3, 3, 2)
plt.stem(n, unit_step, basefmt=" ")
plt.title("Unit step sequence")

# Ramp sequence
ramp_sequence = np.zeros(point)
n = np.arange(-5, 16)  # Adjust n range to match MATLAB code
for i in range(5, point):
    ramp_sequence[i] = i - 5
plt.subplot(3, 3, 3)
plt.stem(n, ramp_sequence, basefmt=" ")
plt.title("Ramp sequence")

# Exponential sequence (growing)
point = 20
n = np.arange(0, point + 1)
expo_seq = 1.3 ** n
plt.subplot(3, 3, 4)
plt.stem(n, expo_seq, basefmt=" ")
plt.title("Exponential sequence (growing)")

# Exponential sequence (decaying)
expo_seq = 0.7 ** n
plt.subplot(3, 3, 5)
plt.stem(n, expo_seq, basefmt=" ")
plt.title("Exponential sequence (decaying)")

# Random sequence
point = 21
n = np.arange(-10, 11)
random_seq = np.random.rand(point)
plt.subplot(3, 3, 6)
plt.stem(n, random_seq, basefmt=" ")
plt.title("Random sequence")

# Analog sine wave
f = 5  # Frequency in Hz
a = 12  # Amplitude
t = np.arange(0, 1, 0.01)  # Continuous time
sin_seq = a * np.sin(2 * np.pi * f * t)
plt.subplot(3, 3, 7)
plt.plot(t, sin_seq)
plt.title("Sine wave (analog)")

# Digital sine wave (sampled sine wave)
fs = 50  # Sampling frequency
n = np.arange(0, 1, 1/fs)  # Samples
sampled_seq = a * np.sin(2 * np.pi * f * n)
plt.subplot(3, 3, 8)
plt.stem(n, sampled_seq, basefmt=" ")
plt.title("Sampled sine wave")

# Complex sine wave (combination of two sine waves)
t = np.arange(0, 1, 0.005)
complex_seq = 5 * np.sin(2 * np.pi * 3 * t) + 5 * np.sin(2 * np.pi * 7 * t)
plt.subplot(3, 3, 9)
plt.plot(t, complex_seq)
plt.title("Complex sine wave")

# Show all plots
plt.tight_layout()
plt.show()
