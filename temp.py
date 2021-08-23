import plotexpv2 as pex
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as scps
import scipy
import pandas as pd

T = np.linspace(0,30,30*20000)
V = np.sin(T/30*2*np.pi)
# V = np.sin(T)

# fs = int(len(V)/(T[-1]-T[0]))
# sos = scps.butter(10, 1/300, 'highpass', fs=fs, output='sos')
# V_filt = scps.sosfilt(sos, V)

yf = scipy.fft.rfft(V)
xf = scipy.fft.rfftfreq(30*20000, 1 / 20000)
print(yf)

plt.scatter(xf,np.abs(yf))
plt.show()



# plt.plot(T,V)
# plt.plot(T,V_filt)
# plt.show()
