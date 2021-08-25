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
V = V[24000:]
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
		plt.plot(t, V[p-100:p+300], color='green')
		k=k+1
	else:
		plt.plot(t, V[p-100:p+300], color='blue', alpha=0.3)

print(k)
plt.show()
#############################################

########## Fitting to alpha function ##############
def alpha(t, A,B,C,tau):
	Re = C-A*(t-B)/tau*np.exp(-(t-B)/tau)
	Re[Re>C] = C
	return Re

def alpha2(t, A,B,C,tau1,tau2):
	Re = C-A*(t-B)*np.exp(-(t-B)/tau)
	Re[Re>C] = C
	return Re

A1_list = []
tau1_list = []
B1_list = []
error1_list = []
V1_list = []
A0_list = []
tau0_list = []
B0_list = []
error0_list = []
V0_list = []

paramfitted_list,error_list,V_list = [],[],[]
k=0
for p in potentialEPSCpeaks:
	# paramfitted,error = bcf.brute_scifit(alpha, t, V[peaks[2]-100:peaks[2]+300],  [[1,0,min(V),0.0005], [100,400/fs,max(V),0.005]], ntol = 1000, returnnfactor = 0.01, maxfev = 1000, printerrors=False) #actual fitting to the 20th peak
	paramfitted,error = bcf.brute_scifit(alpha, t, V[p-100:p+300],  [[0.001,0,min(V),0.0005], [0.050,400/fs,max(V),0.005]], ntol = 1000, returnnfactor = 0.01, maxfev = 1000, printerrors=False) #actual fitting to the 20th peak
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
		error0_list.append(error)
		V0_list.append(V[p-100:p+300])
		# plt.plot(T[p], V[p], "x")
		plt.plot(t, alpha(t,A,0,0,tau), color='green')

paramfitted_list = np.array(paramfitted_list)[np.array(error_list)<7]
V_list = np.array(V_list)[np.array(error_list)<7]
error_list = np.array(error_list)[np.array(error_list)<7]

# plt.plot(T[peaks[2]-100:peaks[2]+300], alpha(t,40,0.004,0.085,0.002))
# plt.xlim(2,2.05)
print(k)
plt.show()

fig1,axs1 = plt.subplots(1,1)
axs1.plot(A1_list, tau1_list, 'o', label='Actual EPSC')
axs1.plot(A0_list, tau0_list, 'x', label='Noise')
axs1.legend()
axs1.set_xlabel('EPSC amplitude parameter')
axs1.set_ylabel('EPSC tau')


fig2,axs2 = plt.subplots(1,1)
axs2.plot(A1_list, B1_list, 'o')
axs2.plot(A0_list, B0_list, 'x')

fig3,axs3 = plt.subplots(1,1)
axs3.plot(B1_list, tau1_list, 'o')
axs3.plot(B0_list, tau0_list, 'x')
# plt.show()

#######################################

from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.scatter(A1_list, tau1_list, B1_list, c='blue')
ax.scatter(A0_list, tau0_list, B0_list, c='green')
plt.show()

#################################


############## PCA on raw data #######################

pca = PCA(n_components=15)
pca.fit(np.concatenate((V1_list,V0_list)))
print(pca.explained_variance_ratio_)

V1_list_tr = pca.fit_transform(V1_list)
V0_list_tr = pca.fit_transform(V0_list)

plt.scatter(V1_list_tr[:,0], V1_list_tr[:,1], c='blue')
plt.scatter(V0_list_tr[:,0], V0_list_tr[:,1], c='green')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

#####################################################

############ PCA on alpha fitted parameters ###########
pca = PCA(n_components=2)
parameters1_list = np.transpose([A1_list,tau1_list])
parameters0_list = np.transpose([A0_list,tau0_list])

pca.fit(np.concatenate((parameters1_list,parameters0_list)))
print(pca.explained_variance_ratio_)

parameters1_list_tr = pca.fit_transform(parameters1_list)
parameters0_list_tr = pca.fit_transform(parameters0_list)

plt.scatter(parameters1_list_tr[:,0], parameters1_list_tr[:,1], c='blue')
plt.scatter(parameters0_list_tr[:,0], parameters0_list_tr[:,1], c='green')
plt.show()

########## Checking how good are the fits ###########
for p in actualEPSCidxlist:
	paramfitted,error = bcf.brute_scifit(alpha, t, V[p-100:p+300],  [[0.001,0,min(V),0.0005], [0.050,400/fs,max(V),0.005]], ntol = 1000, returnnfactor = 0.01, maxfev = 1000, printerrors=False) 
	print(paramfitted,error)
	A,B,C,tau = paramfitted
	plt.plot(t, V[p-100:p+300], color='black')
	plt.plot(t, alpha(t,A,B,C,tau), color='blue')
	plt.show()


####### k means on alpha fitted parameters #############
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(paramfitted_list)
labels = kmeans.labels_
for i,label in enumerate(labels):
	if label==1:
		plt.plot(t, V_list[i], c='blue')
	else:
		plt.plot(t, V_list[i], c='green')

plt.show()