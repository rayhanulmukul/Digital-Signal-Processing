import matplotlib.pyplot as plt
import numpy as np

def corelation(x):
    sz = len(x)
    length = (sz * 2) - 1
    x_n = [0] * length
    for k in range(length):
        x_n[k] = sum(x[n] * x[n + k] for n in range(sz - k))
    return x_n

# Sequence
x = [1, 2, 3, 4]
x_n = corelation(x)
plt.stem(range(len(x_n)), x_n, use_line_collection=" ")
plt.title("Auto Correlation")
plt.xlabel('n')
plt.ylabel('Amplitude')

plt.show()