import glob
import pyxdf
import pandas as pd
import numpy as np
from statistics import mean
def get_xdf_file(directory='Experimental_setup/recordings/xdf_recordings/'):
    '''
    Method to select file for conversion

    Parameters:
    Directory - Folder containing xdf_recordings
    '''
    xdf_files = glob.glob(directory+'*.xdf')
    print('Choose your file for conversion')
    for i,xdf_file in enumerate(xdf_files):
        print(f'{i},{xdf_file}')
    index = int(input('Index:'))
    xdf_file_path = xdf_files[index]

    return xdf_file_path

def extract_data(xdf_file_path):
    '''
    Method to get data into dataframes
    '''
    data, header = pyxdf.load_xdf(xdf_file_path)
   
    if data[0]['info']['type'][0] == 'VIDEO':
        print('case1')
        df_frames = pd.DataFrame(data ={ 'FrameNo':[i[0] for i in data[0]['time_series']],
                                        'Type':['Baseline' if i[1]==0 else 'Recording' for i in data[0]['time_series']],
                                        'AU':[i[2] for i in data[0]['time_series']],
                                        'Timestamps':data[0]['time_stamps']})
        df_emg = pd.DataFrame(data={'Channel_Zygomaticus_Major':[i[1] for i in data[1]['time_series']],
                                    'Channel_Levator_Labi':[i[2] for i in data[1]['time_series']],
                                    'Channel_Orbicularis_Oculi':[i[3] for i in data[1]['time_series']],
                                    'Channel_Corrugator_Supercili':[i[4] for i in data[1]['time_series']],                    
                                    'Timestamps': data[1]['time_stamps']})
        
    else:
        print('case 2')
        df_frames = pd.DataFrame(data ={ 'FrameNo':[i[0] for i in data[1]['time_series']],
                                        'Type':['Baseline' if i[1]==0 else 'Recording' for i in data[1]['time_series']],
                                        'AU':[i[2] for i in data[1]['time_series']],
                                        'Timestamps':data[1]['time_stamps']})
        
        df_emg = pd.DataFrame(data={'Channel_Zygomaticus_Major':[i[1] for i in data[0]['time_series']],
                                    'Channel_Levator_Labi':[i[2] for i in data[0]['time_series']],
                                    'Channel_Orbicularis_Oculi':[i[3] for i in data[0]['time_series']],
                                    'Channel_Corrugator_Supercili':[i[4] for i in data[0]['time_series']],                     
                                    'Timestamps': data[0]['time_stamps']})
    

    df_baseline = df_frames[(df_frames['Type'] == 'Baseline') & (df_frames['FrameNo']>0)]
    df_recording    = df_frames[(df_frames['Type']=='Recording') & (df_frames['FrameNo']>0) ]
    df_recording.index = [i for i in range(len(df_recording))]
    df_baseline.index = [i for i in range(len(df_baseline))]

   
    return df_recording,df_baseline,df_emg

def extract_frame_mappings(df_recording,df_emg):
    '''
    Method to map emg data per frame
    '''
    AU_DICT = {
        'au01': 1, 
        'au02': 2,
        'au04': 4,
        'au05': 5,
        'au06': 6,
        'au07': 7,
        'au09': 9,
        'au10': 10,
        'au12': 12,
        'au14': 14,
        'au15': 15,
        'au17': 17,
        'au18': 18,
        'au20': 20,
        'au23': 23,
        'au24': 24,
        'au25': 25,
        'au26': 26,
        'au43': 43
    }
    dataframes_dict = {}
    start_baseline = df_baseline.head(1)['Timestamps'].values[0]
    end_baseline   = df_baseline.tail(1)['Timestamps'].values[0]

    #get average value of baseline
    baseline = df_emg[(df_emg['Timestamps'] >= start_baseline) & (df_emg['Timestamps']<=end_baseline)].drop(['Timestamps'],axis=1).mean(axis=0)

    
    for key,value in AU_DICT.items():
        df = df_recording[df_recording['AU']==value]
        start_time = df.head(1)['Timestamps'].values[0]
        end_time   =  df.tail(1)['Timestamps'].values[0]
        temp = df_emg[(df_emg['Timestamps'] >= start_time) & (df_emg['Timestamps']<=end_time)].drop(['Timestamps'],axis=1)
    
        #subtracting baseline value from emg using brodcasting
        temp = temp.subtract(baseline)

        #Do segmenting of data
    
        #add a column for label
        au_label = pd.Series([ key for i in range(len(temp))])
        au_label.index = temp.index
        temp.insert(4,'Label',au_label,True)
        dataframes_dict[key] = temp


    return dataframes_dict

def construct_dataset(dataframes_dict,filename):
    '''
    Create a dataframe of  the data and saves it
    '''
    dataset = pd.DataFrame()
    sample_count = []
    index_count = 0
    for df in dataframes_dict.values():
        sample_count.append(len(df))
        index_count+=1
        dataset = pd.concat([dataset,df],axis=0)

    print(f'Avg emg samples per AU {sum(sample_count)/index_count}')
    dataset.to_csv('Experimental_setup/dataset/pre-processed/'+filename+'.csv',index=False)

def query_for_resolution(df_recording,df_emg,dataframes_dict):
    res_rec = []
    res_emg = []
    sample_count = []
    for i in range(len(df_recording)-1):
        t1 = df_recording['Timestamps'][i]
        t2 = df_recording['Timestamps'][i+1]
        res_rec.append(t2-t1)

    for i in range(len(df_emg)-1):
        t1 = df_emg['Timestamps'][i]
        t2 = df_emg['Timestamps'][i+1]
        res_emg.append(t2-t1)

    for key,value in dataframes_dict.items():
        print(len(value))
        sample_count.append(len(value))

    print(f'sum {sum(sample_count)}, {len(df_recording)}')

    return res_rec,res_emg,sum(sample_count)/len(df_recording)
    

if __name__ == '__main__':

    xdf_file_path = get_xdf_file()
    df_recording,df_baseline,df_emg = extract_data(xdf_file_path)
    dataframes_dict = extract_frame_mappings(df_recording,df_emg)
    filename = xdf_file_path.split('\\')[-1].split('.')[0]
    construct_dataset(dataframes_dict,filename)
    #res_rec,res_emg,c = query_for_resolution(df_recording,df_emg,dataframes_dict)
    #a = mean(res_rec)
    #b = mean(res_emg)
    #print(f'avg frame resolution:{a} avg emg resolution:{b}, avg emg samples per frame sample:{c}' )
    #print(f'No of samples required for 1ms is {0.01/b}')
   
    
   