# Show that 50 Hz is an alias of the frequency 10 Hz, when sampling at 40 Hz. Assume the signals are sin(2π.10t), sin(2π.50t)

import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0, 1, 0.001)
x = np.sin(2 * np.pi * 10 * t)
plt.figure(figsize=(10, 4))
plt.subplot(2, 2, 1)
plt.plot(t, x)
plt.title("x(t) = sin(2π.10t)")
plt.xlabel("Time(s)")
plt.ylabel("Amplitude")
plt.grid()

t = np.arange(0, 1, 0.001)
x = np.sin(2 * np.pi * 50 * t)
plt.subplot(2, 2, 2)
plt.plot(t, x)
plt.title("x(t) = sin(2π.50t)")
plt.xlabel("Time(s)")
plt.ylabel("Amplitude")
plt.grid()

fs = 40
n = np.arange(0, 1, 1/fs)
x = np.sin(2 * np.pi * 10 * n)
plt.subplot(2, 2, 3)
plt.stem(n, x)
plt.plot(n, x)
plt.title("x(t) = sin(2π.(10/40)n)")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.grid()

fs = 80
n = np.arange(0, 1, 1/fs)
x = np.sin(2 * np.pi * 50 * n)
plt.subplot(2, 2, 4)
plt.stem(n, x)
plt.plot(n, x)
plt.title("x(t) = sin(2π.(50/40)n)")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()
plt.show()