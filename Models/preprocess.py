'''
Following class contains static methods which will be used for preprocessing the data
'''
import math
import numpy as np
from  scipy.fft import fft,rfftfreq
from scipy.signal import butter,lfilter,iirnotch,freqz,filtfilt
class Preprocess:


    @staticmethod 
    def segmentation(timeseries,sampling_freq,window_size):
        '''
        Extract windows of EMG data. Extra samples at the end are ignored 
        Parameters:
        time_series: time series of emg data (time x channels)
        sampling_freq: Sampling freq of dataset (samples/sec)
        window_size: duration in ms chosen for getting the data

        Returns a  tuple of numpy arrays of shapes (segments x timeseries x channels),(total_segments)
        '''

        no_of_samples_per_segment = math.ceil(window_size * sampling_freq/1000)
        #print('No of samples per segment is ',no_of_samples_per_segment)
        
        
        
        AU_DICT = {
        'au01': None, 
        'au02': None,
        'au04': None,
        'au05': None,
        'au06': None,
        'au07': None,
        'au09': None,
        'au10': None,
        'au12': None,
        'au14': None,
        'au15': None,
        'au17': None,
        'au18': None,
        'au20': None,
        'au23': None,
        'au24': None,
        'au25': None,
        'au26': None,
        'au43': None
        }
        ranges = {}
        total_rows = 0
        
        for AU,value in AU_DICT.items():
           
            
            array = timeseries[timeseries['Label']==AU][['Zygomaticus_Major',
                                                          'Levator_Labi',
                                                          'Orbicularis_Oculi', 
                                                          'Corrugator_Supercili']].to_numpy()
            no_of_segments = math.floor(array.shape[0]/no_of_samples_per_segment)
            #print(f'No of segments for {AU}:',no_of_segments)
            ranges[AU] = [total_rows,total_rows+no_of_segments]
            total_rows += no_of_segments
            AU_DICT[AU] = np.empty((no_of_segments,no_of_samples_per_segment,array.shape[1]))
            for segment in range(no_of_segments):
               AU_DICT[AU][segment] = array[segment*no_of_samples_per_segment:(segment+1)*no_of_samples_per_segment,:]

        X = np.empty((total_rows,no_of_samples_per_segment,array.shape[1]))
        Y = np.empty((total_rows))
        for AU,array in AU_DICT.items():
            X[ranges[AU][0]:ranges[AU][1]] = array
            Y[ranges[AU][0]:ranges[AU][1]] = int(AU.split('u')[1])
        
        
        return X,Y,no_of_samples_per_segment
        

    
    @staticmethod
    def fft(timeseries,sf):
        '''
        To calculate the fast fourier transform of each timer series
        '''
        channels = timeseries.shape[1]
        transform = fft(timeseries,axis = 0)
        transform = np.abs(transform)
        freq = np.linspace(0,sf,transform.shape[0])
        return transform,freq
    
    @staticmethod
    def band_pass_filter(timeseries, lowcut, highcut, fs, order=4):
        '''
        Implement a 4th order Butterworth filter for band pass filtering the signal
        '''
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        transform = lfilter(b, a, timeseries,axis=1)
        return transform
    
    @staticmethod
    def notch_filter(timeseries,fs):
        '''
        To remove the PLI freq of 50hz from our data
        '''
        notch_freq = 50
        quality_factor = 20
        # Design a notch filter using signal.iirnotch
        b_notch, a_notch = iirnotch(notch_freq, quality_factor, fs)

        # Compute magnitude response of the designed filter
        freq, h = freqz(b_notch, a_notch, fs=fs)

        # Apply notch filter to the noisy signal using signal.filtfilt
        outputSignal = filtfilt(b_notch, a_notch, timeseries,axis=1)

        return outputSignal


    
    @staticmethod
    def full_wave_rectifier(timeseries):
        '''
        Carries out full wave rectification of the signal i.e takes absolute value of every sample
        '''
        return np.abs(timeseries)
    
  
            
            