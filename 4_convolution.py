import matplotlib.pyplot as plt

def convolution(x, h):
    n = len(x)
    m = len(h)
    output = n + m - 1
    y = [0] * output
    for i in range(output):
        for j in range(n):
            if 0 <= i - j < m:
                y[i] += x[j] * h[i - j]
    return y

# Input Signals
x = [1, 2, 3]
h = [4, 5, 6]

result = convolution(x, h)

plt.stem(range(len(result)), result, use_line_collection=True)
plt.title("Convolution Result")
plt.xlabel("Index")
plt.ylabel("Value")
plt.grid
plt.show()