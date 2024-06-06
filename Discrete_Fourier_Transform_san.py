import matplotlib.pyplot as plt
import numpy as np


def signal(n,ts):
    return np.sin(2*np.pi*1000*n*ts) + 0.5*np.sin(2*np.pi*2000*n*ts + 3*np.pi/4.0)

#using DFT formula of X(m)
def DFT(x,N):
    X = []
    for m in range(N):
        val = 0
        for n in range(N):
            val += x[n]*np.exp(-2j*np.pi*n*m/N)
        X.append(np.round(val,4))
    return X

# calculate Phase in degree
def calc_Phase(X):
    theta = []
    for i in X:
        #handle infinity case
        if i.real==0:
            if i.imag==0:
                theta.append(0)
            elif i.imag<0:
                theta.append(-90)
            else:
                theta.append(90)
        else:
            # Since 1 rad = 180/pi degree
            theta.append(np.arctan(i.imag/i.real)*180/np.pi)
    return theta

#calculate Magnitude
def calc_Mag(X):
    Mg = []
    for i in X:
        Mg.append(np.round(abs(i),4)) #this abs(a+bj) = sqrt(a^2 + b^2)
    return Mg

#calculating PowerSpectrum
def calc_Pow(Mg):
    Ps = []
    for i in Mg:
        Ps.append(np.round(i*i,4))
    return Ps

#calculating Inverse DFT using IDFT formula
def IDFT(X,N):
    y = []
    for n in range(N):
        val = 0
        for m in range(N):
            val += X[m]*np.exp(2j*np.pi*n*m/N)
        y.append(round(val.real/N,7))
    return y

#sampling point
N = 8
#sampling rate
fs = 8000
#sampling interval
ts = 1/fs
#sampling points in kHz
n = np.arange(N)
#sample value x(n)
x = signal(n,ts)

#result of DFT
X = DFT(x,N)
#calculate magnitude
Mg = calc_Mag(X)
#calculating Phase
theta = calc_Phase(X)
#calculating Pw
X_ps = calc_Pow(Mg)

#Inverse DFT to restore signal
y = IDFT(X,N)
##plot e original signal er x axis e time 
##er sathe milanor jonno Sample point diye vag dite hoiche. 
##keno karon jani na.
ty = []
for i in n:
    ty.append(i/N)

#original signal
t = np.linspace(0,1,1000)  ## To just plot smoothly we take 1000 point
sg = np.sin(2*np.pi*1000*t) + 0.5*np.sin(2*np.pi*2000*t + 3*np.pi/4.0)

##plotting Magnitude & Phase spectrum
plt.figure(figsize=(10,6))
plt.subplot(121)
plt.stem(n,Mg,label='Magnitude Spectrum')
plt.ylabel('Magnitude')
plt.xlabel('Frequency (kHz)')
plt.grid()
plt.subplot(122)
plt.stem(n,theta,label='Phase Spectrum')
plt.ylabel('Phase')
plt.xlabel('Frequency (kHz)')
plt.grid()
plt.show()


##plotting Power Spectrum
##This describe the distribution of power into each frequency component.
# plt.figure(figsize=(8,5))
# plt.stem(n,X_ps)
# plt.ylabel('Power')
# plt.xlabel('Frequency (kHz)')
# plt.grid()
# plt.show()


##Plotting the original & restored signal that we get after IDFT
# plt.figure(figsize=(10,6))
# plt.subplot(121)
# plt.plot(t,sg,label='Original Signal',color='Blue')
# plt.xlabel('Time(s)')
# plt.ylabel('Amplitude')
# plt.grid()
# plt.subplot(122)
# plt.plot(ty,y,label='Restored Signal',color='Red')
# plt.xlabel('Time(s)')
# plt.ylabel('Amplitude')
# plt.grid()
# plt.show()