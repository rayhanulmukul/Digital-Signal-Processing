import numpy as np
import math
import matplotlib.pyplot as plt

print(math.pi)

def makeSignal(t,f,delta,am):
  return am*math.sin(2*math.pi*f*t + delta)

def rightShifting(x,ts):
    return x+ts;

def leftShifting(x,ts):
    return x-ts;

x = np.linspace(0,0.01,120)
am_t = input("Enter amplitude:")
am = float(am_t)
freq_t = input("Enter Frequency:")
freq = float(freq_t)
phase_t = input("Enter Phase in degree:")
phase = float(phase_t)
phase = phase*(math.pi/180.0)
print(phase)
y = []
for i in x:
  y.append(makeSignal(i,freq,phase,am))

leftShifted_x = []
for i in x:
  leftShifted_x.append(leftShifting(i,0.005))

rightShifted_x = []
for i in x:
  rightShifted_x.append(rightShifting(i,0.005))


plt.figure(figsize=(10,5))
plt.plot(x,y,color='blue',label="original-wave")
# plt.plot(x,upShifted_y,color='red',label="Up-Shifted Wave")
# plt.plot(x,downShifted_y,color='black',label="Down-Shifted Wave")
plt.plot(leftShifted_x,y,color='orange',label="left-Shifted Wave")
plt.plot(rightShifted_x,y,color='red',label="left-Shifted Wave")
plt.xlabel("Time (s)")
plt.ylabel("Amplitute")
plt.title("Operation on Sine Wave")
plt.grid()
plt.show()