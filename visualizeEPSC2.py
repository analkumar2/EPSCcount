import plotexpv2 as pex
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as scps
import pandas as pd
import scipy
import brute_curvefit as bcf

Address = '../../Raw_data/Deepanjali_data/Organized_Anal/WT_SKproto/preApa/'
filename = '2018_04_05_3.abf'

T, V = pex.expdata(Address+filename, -100e-12) #1st plot
T = T[24000:]
V = V[24000:]
fs = int(len(V)/(T[-1]-T[0]))

V_std = np.std(V)
V_mean = np.mean(V)

V_op = np.copy(V)
V_op[V_op>V_mean-3*V_std] = np.nan

peaks, _ = scipy.signal.find_peaks(-V_op)

plt.plot(T,V)
plt.plot(T, V_op)
# plt.hlines(V_mean-3*V_std, 0,30, colors='red')
plt.plot(T[peaks], V_op[peaks], "x")


# plt.show()

for p in peaks:
	# plt.plot(T[peaks[8]-100:peaks[8]+250], V[peaks[8]-100:peaks[8]+250])
	plt.plot(T[p-100:p+250], V[p-100:p+250], color='green')
plt.show()

def alpha(t, A,B,C,tau):
	Re = C-A*(t-B)*np.exp(-(t-B)/tau)
	Re[Re>C] = C
	return Re

t = np.linspace(0,350/fs, 350)
paramfitted,error = bcf.brute_scifit(alpha, t, V[peaks[20]-100:peaks[20]+250],  [[1,0,min(V),0.0005], [100,350/fs,max(V),0.005]], ntol = 1000, returnnfactor = 0.01, maxfev = 1000, printerrors=True)
print(paramfitted,error)

plt.plot(T,V)
plt.plot(T[peaks[20]-100:peaks[20]+250], alpha(t,*paramfitted))
# plt.plot(T[peaks[20]-100:peaks[20]+250], alpha(t,40,0.004,0.085,0.002))
plt.xlim(2,2.05)
plt.show()