import plotexpv2 as pex
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as scps
import pandas as pd
import scipy
import brute_curvefit as bcf
import math
from sklearn.decomposition import PCA

Address = '../../Raw_data/Deepanjali_data/Organized_Anal/WT_SKproto/preApa/'
filename = '2018_04_05_3.abf'

T, V = pex.expdata(Address+filename, -100e-12) #1st plot
T = T[24000:] #Only using data where the voltage clamp is at -65mV
V = V[24000:]*1e-9 #The imported data is actually holding current and not V. Imported data is in nA
fs = int(len(V)/(T[-1]-T[0])) #sampling rate
actualEPSClist = [1.34,2.0158,3.2047,4.413,4.77,5.17,9.16,9.21,11.28,13.34,18.59,22.96,24.47,26.87,28.45]
actualEPSCidxlist = [2792, 16316, 40092,64275,71341,79408,159236,160223,201638,242730,347704,435096,465485,513322,544970]

V_std = np.std(V)
V_mean = np.mean(V)

V_op = np.copy(V)
V_op[V_op>V_mean-3*V_std] = np.nan #Only using data whichis below 3std 


######## Potential peaks #########################
peaks, _ = scipy.signal.find_peaks(-V_op, distance=int(0.0025*fs))

t = np.linspace(0,400/fs, 400)
potentialEPSCpeaks = []
for p in peaks:
	minVidx = np.argmin(V[p-100:p+300])
	if 0.0049<t[minVidx]<0.0051:
		potentialEPSCpeaks.append(p)


k=0
for p in potentialEPSCpeaks:
	# plt.plot(T[peaks[8]-100:peaks[8]+300], V[peaks[8]-100:peaks[8]+300])
	if p in actualEPSCidxlist:
		plt.plot(t, V[p-100:p+300], color='blue')
		k=k+1
	else:
		plt.plot(t, V[p-100:p+300], color='green', alpha=0.3)

print(k)
plt.show()
#############################################

########## Fitting to alpha function ##############
def alpha(t, A,B,C,tau):
	Re = C-A*(t-B)/tau*np.exp(-(t-B)/tau)
	Re[Re>C] = C
	return Re

A1_list = []
tau1_list = []
B1_list = []
paramfitted1_list = []
error1_list = []
V1_list = []
A0_list = []
tau0_list = []
B0_list = []
paramfitted0_list = []
error0_list = []
V0_list = []

paramfitted_list,error_list,V_list = [],[],[]
k=0
for p in potentialEPSCpeaks:
	# paramfitted,error = bcf.brute_scifit(alpha, t, V[peaks[2]-100:peaks[2]+300],  [[1,0,min(V),0.0005], [100,400/fs,max(V),0.005]], ntol = 1000, returnnfactor = 0.01, maxfev = 1000, printerrors=False) #actual fitting to the 20th peak
	paramfitted,error = bcf.brute_scifit(alpha, t, V[p-100:p+300],  [[1e-12,0,min(V),0.0005], [50e-12,400/fs,max(V),0.005]], ntol = 1000, returnnfactor = 0.01, maxfev = 1000, printerrors=False) #actual fitting to the 20th peak
	print(p,paramfitted,error)
	paramfitted_list.append(paramfitted)
	error_list.append(error)
	V_list.append(V[p-100:p+300])
	A,B,C,tau = paramfitted
	# plt.plot(T[p-100:p+300], alpha(t,*paramfitted))
	if (abs(p-actualEPSCidxlist)<20).any():
		print(T[p],'True')
		A1_list.append(A)
		B1_list.append(B)
		tau1_list.append(tau)
		paramfitted1_list.append(paramfitted)
		error1_list.append(error)
		V1_list.append(V[p-100:p+300])
		k=k+1
		# plt.plot(T[p], V[p], "o")
		plt.plot(t, alpha(t,A,0,0,tau), color='blue')
	else:
		print(T[p], 'False')
		A0_list.append(A)
		B0_list.append(B)
		tau0_list.append(tau)
		paramfitted0_list.append(paramfitted)
		error0_list.append(error)
		V0_list.append(V[p-100:p+300])
		# plt.plot(T[p], V[p], "x")
		plt.plot(t, alpha(t,A,0,0,tau), color='green')

potentialEPSCpeaks = np.array(potentialEPSCpeaks)[np.array(error_list)<7]
paramfitted_list = np.array(paramfitted_list)[np.array(error_list)<7]
V_list = np.array(V_list)[np.array(error_list)<7]
error_list = np.array(error_list)[np.array(error_list)<7]

# plt.plot(T[peaks[2]-100:peaks[2]+300], alpha(t,40,0.004,0.085,0.002))
# plt.xlim(2,2.05)
print(k)
plt.show()

#################################


####### k means on alpha fitted parameters #############
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, algorithm='elkan',n_init=1000, max_iter=10000).fit(paramfitted_list[:,[0,3]])
labels = kmeans.labels_
potentialEPSCpeaks2 = []
for i,label in enumerate(labels):
	if label==1:
		potentialEPSCpeaks2.append(potentialEPSCpeaks[i])
		plt.plot(t, V_list[i], c='blue')
	else:
		plt.plot(t, V_list[i], c='green')

plt.show()

plt.plot(T,V)
plt.plot(T[potentialEPSCpeaks2], V[potentialEPSCpeaks2], 'x', label='predicted EPSC')
plt.plot(T[actualEPSCidxlist], V[actualEPSCidxlist]+1e-12, 'o', label='Actual EPSCs')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Holding current (A)')
plt.title(Address+filename)
plt.show()


######### Spectral clustering ##################
from sklearn.cluster import SpectralClustering
clustering = SpectralClustering(n_clusters=2, assign_labels='discretize', affinity='nearest_neighbors', n_init=1000).fit(paramfitted_list[:,[0,2,3]])
labels = clustering.labels_
potentialEPSCpeaks2 = []
for i,label in enumerate(labels):
	if label==1:
		potentialEPSCpeaks2.append(potentialEPSCpeaks[i])
		plt.plot(t, V_list[i], c='blue')
	else:
		plt.plot(t, V_list[i], c='green')

plt.show()

plt.plot(T,V)
plt.plot(T[potentialEPSCpeaks2], V[potentialEPSCpeaks2], 'x', label='predicted EPSC')
plt.plot(T[actualEPSCidxlist], V[actualEPSCidxlist]+1e-12, 'o', label='Actual EPSCs')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Holding current (A)')
plt.title(Address+filename)
plt.show()


####### Support vector machine ###############
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(np.concatenate((np.array(paramfitted1_list)[:,[0,1,3]], np.array(paramfitted0_list)[:,[0,1,3]])), np.concatenate((np.ones(len(paramfitted1_list)), np.zeros(len(paramfitted0_list)))))
# clf.fit(np.concatenate((V1_list, V0_list)), np.concatenate((np.ones(len(V1_list)), np.zeros(len(V0_list)))))

# labels = np.concatenate(( clf.predict(np.array(paramfitted1_list)[:,[0,1,3]]), clf.predict(np.array(paramfitted0_list)[:,[0,1,3]]) ))
# labels = np.concatenate( clf.predict(V1_list), clf.predict(V0_list) )
labels = clf.predict(np.array(paramfitted_list)[:,[0,1,3]])

potentialEPSCpeaks2 = []
for i,label in enumerate(labels):
	if label==1:
		potentialEPSCpeaks2.append(potentialEPSCpeaks[i])
		plt.plot(t, V_list[i], c='blue')
	else:
		plt.plot(t, V_list[i], c='green')

plt.show()

plt.plot(T,V)
plt.plot(T[potentialEPSCpeaks2], V[potentialEPSCpeaks2], 'x', label='predicted EPSC')
plt.plot(T[actualEPSCidxlist], V[actualEPSCidxlist]+1e-12, 'o', label='Actual EPSCs')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Holding current (A)')
plt.title(Address+filename)
plt.show()