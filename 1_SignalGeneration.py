import matplotlib.pyplot as plt
import numpy as np
plt.close('all')

# Unit Sample Sequence
n = np.arange(-10, 11)
unit_step = np.where(n == 0, 1, 0)
plt.subplot(3, 3, 1)
plt.stem(n, unit_step, basefmt="")
plt.title("Unit Sample")
plt.grid()

# Unit Step Signal
n = np.arange(-10, 11)
unit_step = np.where(n >= 0, 1, 0)
plt.subplot(3, 3, 2)
plt.stem(n, unit_step, basefmt="")
plt.title("Unit Step")
plt.grid()

# Unit Ramp Signal
n = np.arange(-10, 11)
unit_step = np.where(n >= 0, n, 0)
plt.subplot(3, 3, 3)
plt.stem(n, unit_step, basefmt="")
plt.title("Unit Ramp")
plt.grid()

# Exponential Signal Growing
n = np.arange(0, 20)
base = 1.3
expo_seq = base ** n
plt.subplot(3, 3, 4)
plt.stem(n, expo_seq, basefmt="")
plt.title("Exponential Signal")
plt.grid()

# Exponential Signal Decaying
n = np.arange(0, 20)
base = 0.7
expo_seq = base ** n
plt.subplot(3, 3, 5)
plt.stem(n, expo_seq, basefmt="")
plt.title("Exponential Signal")
plt.grid()

# Random Signal
n = np.arange(-10, 11)
rand_seq = np.random.rand(len(n))
plt.subplot(3, 3, 6)
plt.stem(n, rand_seq, basefmt="")
plt.title("Random Signal")
plt.grid()

# Analog Sin Wave
freq = 5
ampl = 12
t = np.arange(0, 1, 0.01)
sin_wave = ampl * np.sin(2 * np.pi * freq * t)
plt.subplot(3, 3, 7)
plt.plot(t, sin_wave)
plt.title("Analog Sin Wave")
plt.grid()

# Digital Sin Wave (Sampled Sin Wave)
freq = 5
ampl = 12
t = np.arange(0, 1, 1/50)
sin_wave = ampl * np.sin(2 * np.pi * freq * t)
plt.subplot(3, 3, 8)
plt.stem(t, sin_wave, basefmt="")
plt.title("Sampled Sin Wave")
plt.grid()

# Complex Sin Wave
a = 5
t = np.arange(0, 1, 1/100)
comp_sin = a * np.sin(2 * np.pi * 3 * t) + a * np.sin(2 * np.pi * 7 * t)
plt.subplot(3, 3, 9)
plt.plot(t, comp_sin)
plt.title("Complex Sin Wave")
plt.grid()

plt.tight_layout()
plt.show()

