'''
Following script will create a pipelined process for preprocessing and training/evaluating the data
'''
from pathlib import Path
import pandas as pd
from preprocess import *
from visualization import *
from features import *
import pickle

class Dataset:

    def __init__(self,filepath) -> None:
        '''
        Method is for loading the dataset
        '''
        self.csv_data  = pd.read_csv(str(filepath))
        self.sample_rate = 2000 
        self.filename = filepath.parts[-1].split('.')[0]

class Pipeline:

    def __init__(self,filepath,window_length =50) -> None:
        dataset = Dataset(filepath)
        self.sample_rate = dataset.sample_rate
        self.filename = dataset.filename 
        self.X,self.Y,self.samples_per_segment = Preprocess.segmentation(dataset.csv_data,self.sample_rate,window_length)
        self.total_channels = self.X.shape[2]
        self.columns = dataset.csv_data.columns[:4]  
        self.features = {
            'Integrated EMG'            : TimeDomain.IEMG,
            'Mean Absolute Value'       : TimeDomain.MAV,
            'Simple Square Integral'    : TimeDomain.SSI,
            'Variance of EMG'           : TimeDomain.VAR,
            'Root Mean Square'          : TimeDomain.RMS,
            'Waveform Length'           : TimeDomain.WL,
            'Standard Deviation'        : TimeDomain.SD,
            'Zero Crossings'            : TimeDomain.ZC,
            'Number of Peaks'           : TimeDomain.NP,
            'Mean Peak Value'           : TimeDomain.MPV,
            'Skewness'                  : TimeDomain.Skewness,
            'Difference Absolute Mean Value': TimeDomain.DAMV,
            'Frequency Median'          : FrequencyDomain.FMD,
            'Modified Median Frequency' : FrequencyDomain.MMDF,
            'Modified Frequency Mean'   : FrequencyDomain.MMNF,
        }
        
    def prepare_data_matrix(self,window_length,save=False):

        #for raw emg data
        X_with_features_raw,feature_labels = FeatureConstructor.construct_features(self.X,self.features,self.sample_rate,self.columns)
   
        #for filtered emg signal
        X_bp_filtered = Preprocess.band_pass_filter(self.X,lowcut=15,highcut=500,fs=self.sample_rate) 
        X_notch_filtered = Preprocess.notch_filter(X_bp_filtered,self.sample_rate)
        X_rectified      = Preprocess.full_wave_rectifier(X_notch_filtered)
        X_with_features_filtered,_ = FeatureConstructor.construct_features(X_rectified,self.features,self.sample_rate,self.columns)
        #print(f'*************features shape- raw:{X_with_features_raw.shape},filtred:{X_with_features_filtered.shape}')


        
        df_raw_features = pd.DataFrame(X_with_features_raw,columns=feature_labels)
        df_filtered_features = pd.DataFrame(X_with_features_filtered,columns=feature_labels)
        df_labels   = pd.DataFrame(self.Y,columns=['Labels'])
        if save:
             df_raw_features.to_csv('Models/post_processed/'+str(window_length)+'./raw/'+self.filename+'_feature_matrix.csv',index=False)
             df_filtered_features.to_csv('Models/post_processed/'+str(window_length)+'./filtered/'+self.filename+'_feature_matrix.csv',index=False)
             np.save('Models/post_processed/'+str(window_length)+'./raw/'+self.filename+'_only_segmented.npy',self.X)
             np.save('Models/post_processed/'+str(window_length)+'./filtered/'+self.filename+'_only_segmented.npy',X_rectified)
             df_labels.to_csv('Models/post_processed/'+str(window_length)+'/'+self.filename+'_labels.csv',index=False)

        return X_with_features_raw,X_with_features_filtered,feature_labels
              




    
if __name__ == '__main__':
        window_lengths = [50,100,150,200,250,300,350,400]
        samples_per_segment_dict = {}
        
        directory = 'Experimental_Setup/dataset/pre-processed/'
        
        filepaths = list(Path(directory).glob("*.csv"))
        for filepath in filepaths:
            print('Processing:',filepath)
            for window_length in window_lengths:
                print('-------Generating Data Matrices for segment length:',window_length)
                p = Pipeline(filepath,window_length)
                _,_,feature_labels=p.prepare_data_matrix(window_length,save=True)
                samples_per_segment_dict[window_length] = p.samples_per_segment
        pickle.dump(feature_labels,open('Models/post_processed/feature_labels'+'.sav','wb'))
        

        #construct one unified dataset for specific window lengths
        for window_length in window_lengths:
            filepaths_raw = Path('Models/post_processed/'+str(window_length)+'/raw/').glob('*.csv')
            filepaths_filtered = Path('Models/post_processed/'+str(window_length)+'/filtered/').glob('*.csv')
            filepaths_raw_only_segmented = Path('Models/post_processed/'+str(window_length)+'/raw/').glob('*.npy')
            filepaths_filtered_only_segmented = Path('Models/post_processed/'+str(window_length)+'/filtered/').glob('*.npy')

            X = np.empty((0,len(p.features.keys())*p.total_channels))
            Y = np.empty((0,1))
            for filepath in filepaths_raw:
                df = pd.read_csv(str(filepath))
                df_array = df.to_numpy()
                df_labels = pd.read_csv('Models/post_processed/'+str(window_length)+'/'+filepath.parts[-1].split('_')[0]+'_labels.csv').to_numpy()
                X = np.concatenate((X,df_array),axis=0)
                Y = np.concatenate((Y,df_labels),axis=0)
            #print('final raw shape',X.shape,Y.shape)
            np.save('Models/post_processed/'+str(window_length)+'/X_raw.npy',X)
            np.save('Models/post_processed/'+str(window_length)+'/Y_raw.npy',Y)
            with open('Models/post_processed/'+str(window_length)+'/labels', "wb") as fp:
                  pickle.dump(df.columns,fp)
                 


            X = np.empty((0,len(p.features.keys())*p.total_channels))
            Y = np.empty((0,1))
            for filepath in filepaths_filtered:
                df = pd.read_csv(str(filepath))
                df_array = df.to_numpy()
                df_labels = pd.read_csv('Models/post_processed/'+str(window_length)+'/'+filepath.parts[-1].split('_')[0]+'_labels.csv').to_numpy()
                X = np.concatenate((X,df),axis=0)
                Y = np.concatenate((Y,df_labels),axis=0)
            #print('final filtered shape',X.shape,Y.shape)
            np.save('Models/post_processed/'+str(window_length)+'/X_filtered.npy',X)
            np.save('Models/post_processed/'+str(window_length)+'/Y_filtered.npy',Y)
            
            X = np.empty((0,samples_per_segment_dict[window_length],4))
            for filepath in filepaths_raw_only_segmented:
                 temp_X = np.load(filepath)
                 X = np.concatenate((X,temp_X),axis=0)
            X = np.reshape(X,(X.shape[0],4,-1))
            np.save('Models/post_processed/'+str(window_length)+'/X_raw_only_segmented.npy',X)

            X = np.empty((0,samples_per_segment_dict[window_length],4))
            for filepath in filepaths_filtered_only_segmented:
                 temp_X = np.load(filepath)
                 X = np.concatenate((X,temp_X),axis=0)
            X = np.reshape(X,(X.shape[0],4,-1))
            np.save('Models/post_processed/'+str(window_length)+'/X_filtered_only_segmented.npy',X)
                 

                 


            
        


            
                





  

   
   
   
    
