'''
Following scripts contains methods for extracting various kind of features
'''
import numpy as np
import scipy.signal
from scipy.fft import rfft,rfftfreq,fft,fftfreq
from copy import deepcopy
import pywt
from sklearn.feature_selection import mutual_info_classif,SelectKBest
import pandas as pd
from scipy.stats import skew

class TimeDomain:
    '''
    Following class is for extracting time domain features from emg data

    All methods expect timeseries in the format: (segments,time,channels)
    '''

    def IEMG(timeseries):
        '''
        Integrated EMG(IEMG)

        '''
        return np.sum(np.abs(timeseries),axis=1)

    @staticmethod
    def MAV(timeseries):
        '''
        Mean Absolute Value(mAV)
        '''

        return np.mean(np.abs(timeseries),axis=1)
    
    @staticmethod
    def MAVS(timeseries):
        '''
        Mean Absolute Value Slope(MAVS)
        '''
        temp = TimeDomain.MAV(timeseries)
        array = np.empty((temp.shape[0]-1,temp.shape[-1]))
        for i in range(temp.shape[0]-1):
            array[i] = temp[i+1] - temp[i]

        return array
    
    @staticmethod
    def SSI(timeseries):
        '''
        Simple Square Integral(SSI)
        '''
        return np.sum(np.abs(timeseries)**2,axis=1)
    
    @staticmethod
    def VAR(timeseries):
        '''
        Variance of EMG(VAR)
        '''
        return np.sum(np.square(timeseries),axis=1)/(timeseries.shape[1] -1)


    @staticmethod
    def RMS(timeseries):
        '''
        Root Mean Square(RMS)
        '''
        return np.sqrt(np.mean(np.square(timeseries),axis=1)) 
    
    @staticmethod
    def WL(timeseries):
        '''
        Waveform Length(WL)
        '''
        array = np.empty((timeseries.shape[0],timeseries.shape[1]-1,timeseries.shape[2]))
        for i in range(array.shape[1] - 1):
            array[:,i,:] = timeseries[:,i+1,:] - timeseries[:,i,:]
        
        return np.sum(np.abs(array),axis=1)
    
    @staticmethod
    def SD(timeseries):
        '''
        Standard Deviation(SD)
        '''
        return np.std(timeseries,axis=1)
    
    @staticmethod
    def ZC(timeseries):
        '''
        Zero Crossings(ZC)
        '''
        zc_array = np.empty((timeseries.shape[0],timeseries.shape[2]))
        temp = np.diff(np.signbit(timeseries),axis=1)
        for channel in range(timeseries.shape[2]):
            for row in range(timeseries.shape[0]):
                zero_crossings = np.where(temp[row,:,channel])[0]
                zc_array[row,channel] = len(zero_crossings)
      
        return zc_array
    
    @staticmethod
    def NP(timeseries):
        '''
        Number of Peaks(NP)
        '''
        RMS = TimeDomain.RMS(timeseries)
        np_array = np.empty((timeseries.shape[0],timeseries.shape[2]))
        for channel in range(timeseries.shape[2]):
            for row in range(timeseries.shape[0]):
                row_temp = timeseries[row,:,channel]
                rms_value = RMS[row,channel]
                temp = np.where(row_temp>rms_value)[0]
                np_array[row,channel] = len(temp)

        return np_array
    
    @staticmethod
    def MPV(timeseries):
        '''
        Mean of Peak Values
        '''
        RMS = TimeDomain.RMS(timeseries)
        np_array = np.empty((timeseries.shape[0],timeseries.shape[2]))
        for channel in range(timeseries.shape[2]):
            for row in range(timeseries.shape[0]):
                row_temp = timeseries[row,:,channel]
                rms_value = RMS[row,channel]
                temp = np.where(row_temp>rms_value)[0]
                mean = np.mean(row_temp[temp])
                np_array[row,channel] = mean

        np_array = np.nan_to_num(np_array,nan = 0)

        return np_array
    
    @staticmethod
    def DAMV(timeseries):
        '''
        Difference absolute Mean Value(DAMV)
        '''
        t = np.diff(timeseries,axis=1)
        a =  np.mean(t,axis=1)
        
        return a
    
    @staticmethod
    def Skewness(timeseries):
        '''
        Skewness
        '''
        return skew(timeseries,axis=1)



    
    




class FrequencyDomain:
    '''
    Following class is for extracting Freq domain features from emg data

    All methods expect timeseries in the format: (segments,time,channels)
    '''

    @staticmethod
    def FMD(timeseries,sf):
        '''
        Frequency Median
        '''
        (f,PSD) = scipy.signal.periodogram(timeseries,axis=1,scaling='density')
        return 0.5 * np.sum(PSD,axis = 1)
    
    @staticmethod
    def MMDF(timerseries):
        '''
        Modified Median Frequency
        '''
        amp = np.abs(rfft(timerseries,axis=1))
        return 1/2 * np.sum(amp,axis=1)
    
    @staticmethod
    def MMNF(timerseries,sf):
        '''
        Modified  Frequency Mean
        '''
        amp = np.abs(rfft(timerseries,axis=1))
        temp = deepcopy(amp)
        freq = rfftfreq(timerseries.shape[1],1/sf)
        for i in range(timerseries.shape[2]):
            temp [:,:,i] = temp[:,:,i]*freq

        return np.sum(temp,axis=1)/np.sum(amp,axis=1)
    

class TimeFreqDomain:
    '''
    Extracts Time-Freq Domain features using wavelet Packet-Transform
    '''
    def wpt(timeseries,levels,method):
        '''
        Wavelet Packet Transform

        levels = tree level at which time resolution must be carried out
        '''
        feature = np.empty((timeseries.shape[0],timeseries.shape[2]))
        for channel in range(8):
            wp = pywt.WaveletPacket(data=timeseries[:,:,channel], wavelet='db2', mode='symmetric',maxlevel=levels+1,axis=1)
            Nodes = [node for node in wp.get_level(levels, 'freq')]
            for i,node in enumerate(Nodes):
                reconstructed_data = node.reconstruct(False)
                feature[:,channel] = method(reconstructed_data)
        
        return feature
    
    @staticmethod
    def construct_wpt_feature_matrix(timeseries,levels,methods):
        feature_matrix = np.empty((timeseries.shape[0],len(methods.keys())))
        for key,method in methods.items():
            feature = TimeFreqDomain.wpt(timeseries,5,method)
            print(feature.shape)
    

class FeatureConstructor:

    @staticmethod
    def construct_features(timseries,features_dict,sf,columns):
        '''
        Constructs a feature matrix

        features:  Dictionary mapping feature names to corresponding methods
        '''
        features = np.empty((timseries.shape[0],len(features_dict.keys()),timseries.shape[2]))
        i = 0
        for key,method in features_dict.items():
            if method == FrequencyDomain.FMD or method == FrequencyDomain.MMNF:
                temp = method(timseries,sf)
            else:
                temp = method(timseries)
            features[:,i,:] = temp
            i+=1

        #Flatten out array with channels data
        features = np.reshape(features,(timseries.shape[0],-1))
        feature_labels = []
        for column in columns:
            for key in features_dict.keys():
                feature_labels.append(column+'_'+key)
      

        return features,feature_labels



    @staticmethod
    def feature_extractor(X,y,feature_labels,k=5):
        '''
        Selects K best features from all the feature set based on mutual information
        '''
        mi_scores = mutual_info_classif(X,y)
        mi_scores = pd.Series(mi_scores, name="MI Scores", index=feature_labels)
        mi_scores = mi_scores.sort_values(ascending=False)
        print(mi_scores)
        best_5 = mi_scores.index[:k]

        X = SelectKBest(mutual_info_classif,k=k).fit_transform(X,y)

        return X,best_5
   


                
                









