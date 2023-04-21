'''
Following script provideds helpful methods  for visualizing the data recorded
'''
import matplotlib.pyplot as plt
from scipy import signal
from preprocess import Preprocess
import numpy as np 

class Grapher:

    @staticmethod
    def sig_viewer(timeseries):
        '''
        Helps visualize the signal

        timeseries: Should be of the shape (samples x channels) 
        '''
        x = [i for i in range(timeseries.shape[0])]
        channels = timeseries.shape[1]

        fig,axes = plt.subplots(nrows=channels,ncols=1,sharex=True)

        for i in range(channels):
            axes[i].plot(x,timeseries[:,i])

        
        plt.show()
        plt.tight_layout()

    @staticmethod
    def fft_visualizer(timeseries,sf):

        f,psd = signal.welch(timeseries,sf,axis=0)
        fig,axes = plt.subplots(nrows=psd.shape[1],ncols=1,sharex=True)

        for i in range(psd.shape[1]):
            axes[i].plot(f,psd[:,i])

        
        plt.show()
        plt.tight_layout()
 




