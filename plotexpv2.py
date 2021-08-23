# exec(open('plotexpv1.py').read())

#v2: now also plots dat files


import numpy as np
import quantities as pq
import matplotlib.pyplot as plt
from neo.io import AxonIO
import os
import argparse

def expdata(Address, currentlevel):
    if os.path.splitext(Address)[1] == '.abf':
        CurrAmp = int((currentlevel - (-100e-12))/25e-12)
        reader = AxonIO(filename=Address)
        Samprate = reader.get_signal_sampling_rate()
        seg = reader.read_block(signal_group_mode='split-all').segments[CurrAmp]
        Tdur = np.array(seg.t_stop - seg.t_start)
        V = np.array(np.ravel(seg.analogsignals[0]))*1e-3
        return [np.linspace(0,Tdur+0,len(V)), V]
    elif os.path.splitext(Address)[1] == '.dat':
        CurrAmp = int((currentlevel - (-100e-12))/25e-12)
        a = np.loadtxt(Address)
        b = np.transpose(a)
        T = b[0]
        V = b[1:][CurrAmp]*1e-3
        return [T,V]

def plotexp(Address, currentlevel):
    T, V = expdata(Address, currentlevel)
    plt.plot(T,V, label=Address)
    plt.legend()
    plt.title(f'Current clamp at {currentlevel}A')
    plt.xlabel('Time (s)')
    plt.ylabel('Membrane potential (V)')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Saves all the plots of all the abf files in the folder recursively')
    parser.add_argument('Address')
    parser.add_argument('currentlevel', type=float)
    args = parser.parse_args()
    plotexp(args.Address, args.currentlevel)
