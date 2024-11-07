# Upsampling and Downsampling of a discrete time sequence

import matplotlib.pyplot as plt
import numpy as np


t = np.arange(0, 0.008, 0.0001)
x = 5 * np.sin(2 * np.pi * 500 * t + np.radians(90))
plt.subplot(3, 2, 1)
plt.plot(t, x)
plt.title('x(n) = 5sin(2π.500t + π/2)')
plt.xlabel('Time(s)')
plt.ylabel('Amplitude')

fs = 2000
n = np.arange(0, 0.008, 1/fs)
x_n = 5 * np.sin(2 * np.pi * 500 * n + np.pi/2)
plt.subplot(3, 2, 2)
plt.stem(x_n)
plt.title('x(n) = 5cos(2π.(500/3750)n)')
plt.xlabel('n')
plt.ylabel('Amplitude')

# Upsampling by a factor of 2
l = 2
upsampled_x = np.zeros(l * len(x_n))
upsampled_x[::2] = x_n
plt.subplot(3, 2, 3)
plt.stem(upsampled_x)
plt.title('Upsampled Sequence(with zeros)')
plt.xlabel('n')
plt.ylabel('Amplitude')

# Downsampling by a factor of 2
m = 2
downsampled_x = np.zeros(l * len(x_n))
downsampled_x = x_n[::2]
plt.subplot(3, 2, 4)
plt.stem(downsampled_x)
plt.title('Downsampled Sequence(with zeros)')
plt.xlabel('n')
plt.ylabel('Amplitude')

# Smoothing the signal by taking average of two adjacents of zero
for i in range(1, len(upsampled_x) - 1, 2):
    upsampled_x[i] = upsampled_x[i - 1] + upsampled_x[i + 1] / 2

plt.subplot(3, 2, 5)
plt.stem(upsampled_x)
plt.title('Upsampled sequence (Smoothed)')
plt.xlabel('n')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()