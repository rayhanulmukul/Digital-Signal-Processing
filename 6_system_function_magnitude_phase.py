# Finding magnitude and phase response of system described by system function
# # $$H(z) = \\frac{0.1 + 0.2z^{-1} + 0.3z^{-2}}{1 - 0.5z^{-1} + 0.25z^{-2}}$$"

import matplotlib.pyplot as plt
import numpy as np

a = [0.1, 0.2, 0.3] # Numerator coefficients
b = [1, -0.5, 0.25] # Denominator coefficients

N = 8000
w = np.linspace(0, np.pi, N)
H = np.zeros(N, dtype=complex)

for k in range(N):
    z = np.exp(1j * w[k]) # z = e^(jw)
    numerator = 0 + 0j
    denominator = 0 + 0j
    for n in range(len(a)):
        numerator += a[n] * (z ** -n)
    for n in range(len(a)):
        denominator += b[n] * (z ** -n)
    H[k] = numerator / denominator



magnitude = np.abs(H)
phase = np.angle(H)
plt.subplot(2, 1, 1)
plt.plot(w, magnitude)
plt.title('Magnitude Response')
plt.xlabel('Frequency in (radians / sample)')
plt.ylabel('Magnitude of H')


plt.subplot(2, 1, 2)
plt.plot(w, phase)
plt.title('Phase Response')
plt.xlabel('Frequency in (radians / sample)')
plt.ylabel('Phase of H')


plt.tight_layout()
plt.show()