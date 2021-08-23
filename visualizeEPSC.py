import plotexpv2 as pex
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as scps
import pandas as pd
import scipy

Address = '../../Raw_data/Deepanjali_data/Organized_Anal/WT_SKproto/preApa/'
filename = '2018_04_05_3.abf'

T, V = pex.expdata(Address+filename, -100e-12) #1st plot
T = T[24000:]
V = V[24000:]
fs = int(len(V)/(T[-1]-T[0]))

yf = scipy.fft.rfft(V)
xf = scipy.fft.rfftfreq(len(V), 1/fs)
points_per_freq = len(xf) / (fs / 2)

for a in np.arange(50,5000,50):
	target_idx = int(points_per_freq * a)
	yf[target_idx - 10 : target_idx + 10] = 0

target_idx = int(points_per_freq * 100)
yf[:target_idx] = 0
target_idx = int(points_per_freq * 5000)
yf[target_idx:] = 0

plt.plot(xf, np.abs(yf))
plt.show()

new_sig = scipy.fft.irfft(yf)
plt.plot(T,V)
plt.plot(T,new_sig)
plt.show()

# V_std = np.std(V)
# V_mean = np.mean(V)

# V_smooth = scps.savgol_filter(V, 3999, 3)
# V_mvavg = np.convolve(V, np.ones(2000), 'same') / 2000
# Vpd = pd.DataFrame(V)
# V_mvavgpd = np.array(Vpd.rolling(20000).mean()).flatten()
# V_mvstdpd = np.array(Vpd.rolling(20000).std()).flatten()

# V_onlypeaks = V[V<V_mvavgpd-3*V_mvstdpd]


# sos = scps.butter(10, 300, 'highpass', fs=fs, output='sos')
# V_filt = scps.sosfilt(sos, V)

# fig,axs = plt.subplots(1,1)
# axs.plot(T,V)
# axs.plot(T, V_smooth)

# fig2,axs2 = plt.subplots(1,1)
# axs2.plot(T, V-V_smooth)

# fig3,axs3 = plt.subplots(1,1)
# axs3.plot(T,V)
# axs3.plot(T, V-V_mvavg)

# fig4,axs4 = plt.subplots(1,1)
# axs4.plot(T,V)
# axs4.plot(T[V<V_mvavgpd-3*V_mvstdpd], V_onlypeaks)
# axs4.plot(T,V-V_mvstdpd)

# fig5,axs5 = plt.subplots(1,1)
# axs5.plot(T,V)
# axs5.plot(T, V_filt)
# axs5.plot(T,V-V_mvstdpd)

# fig6,axs6 = plt.subplots(1,1)
# axs6.plot(T,V)
# axs6.hlines(V_mean-3*V_std, 0,30, colors='green')
# axs6.plot(T, V_filt)


plt.show()
