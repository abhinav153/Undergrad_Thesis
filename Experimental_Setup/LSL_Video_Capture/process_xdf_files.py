'''
Script to convert xdf recordings to a a preprocessed dataset
'''
import pyxdf
import pandas as pd
import glob
from pathlib import Path

def get_xdf_file(directory='Experimental_setup/LSL_Video_Capture/xdf_recordings/'):
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

    #read the Video stream & EMG stream
    if data[1]['info']['type'][0] == 'VIDEO':
        df_frames = pd.DataFrame(data ={ 'FrameNo:':[i[0] for i in data[1]['time_series']],'Timestamps':data[1]['time_stamps']})
        df_emg = pd.DataFrame(data = {'Channel_1': [i[0] for i in data[0]['time_series']],
                              'Channel_2': [i[1] for i in data[0]['time_series']],
                              'Channel_3': [i[2] for i in data[0]['time_series']],
                              'Channel_4': [i[3] for i in data[0]['time_series']],
                              'Channel_5': [i[4] for i in data[0]['time_series']],
                              'Channel_6': [i[5] for i in data[0]['time_series']],
                              'Channel_7': [i[6] for i in data[0]['time_series']],
                              'Channel_8': [i[7] for i in data[0]['time_series']],
                              'Timestamps': data[0]['time_stamps'] })

    else:
        df_frames = pd.DataFrame(data ={ 'FrameNo:':[i[0] for i in data[0]['time_series']],'Timestamps':data[0]['time_stamps']})
        df_emg = pd.DataFrame(data = {'Channel_1': [i[0] for i in data[1]['time_series']],
                              'Channel_2': [i[1] for i in data[1]['time_series']],
                              'Channel_3': [i[2] for i in data[1]['time_series']],
                              'Channel_4': [i[3] for i in data[1]['time_series']],
                              'Channel_5': [i[4] for i in data[1]['time_series']],
                              'Channel_6': [i[5] for i in data[1]['time_series']],
                              'Channel_7': [i[6] for i in data[1]['time_series']],
                              'Channel_8': [i[6] for i in data[1]['time_series']],
                              'Timestamps': data[1]['time_stamps'] })
    
    #extract frame data,[frames with -1 as input are irrelevant]
    df_frames = df_frames[df_frames['FrameNo:']>=0]
    df_frames.index = df_frames['FrameNo:']


    return df_frames,df_emg

def extract_frame_mappings(df_frames,df_emg):
    '''
    Method to map emg data per frame
    '''
    frame_mapping = {}
    for i in range(len(df_frames)-1):
        t1 = df_frames.loc[i]['Timestamps']
        t2 = df_frames.loc[i+1]['Timestamps']
        temp = df_emg[(df_emg['Timestamps']>= t1) & (df_emg['Timestamps']<=t2)]
        frame_mapping[i] = list(temp.index)

    return frame_mapping

def get_labels(labels_directory='Experimental_setup/Labels/'):
    '''
    Method to extract AU unit recordings per frame label
    '''
    label_files = glob.glob(labels_directory+'*.csv')
    print('Choose your file for conversion')
    for i,label_file in enumerate(label_files):
        print(f'{i},{label_file}')
    index = int(input('Index:'))
    label_file_path = label_files[index]
    df_labels = pd.read_csv(label_file_path)

    return df_labels,Path(label_file_path).parts[-1]

def construct_dataframe(frame_mapping,df_emg,df_labels):
    '''
    Match the emg data with labels and construct a dataframe out of it
    '''
    df_constructed = pd.DataFrame(columns=[ 'frame', 
                                'Channel_1',
                                'Channel_2',
                                'Channel_3',
                                'Channel_4',
                                'Channel_5',
                                'Channel_6',
                                'Channel_7',
                                'AU01_r', 
                                'AU02_r',
                                'AU04_r', 
                                'AU05_r', 
                                'AU06_r', 
                                'AU07_r',
                                'AU09_r', 
                                'AU10_r',
                                'AU12_r',
                                'AU14_r', 
                                'AU15_r',
                                'AU17_r', 
                                'AU20_r', 
                                'AU23_r',
                                'AU25_r', 
                                'AU26_r'])
    
    for frame,mapping in frame_mapping.items():
        labels = df_labels[df_labels['frame']==frame][['frame',
                                'AU01_r', 
                                'AU02_r',
                                'AU04_r', 
                                'AU05_r', 
                                'AU06_r', 
                                'AU07_r',
                                'AU09_r', 
                                'AU10_r',
                                'AU12_r',
                                'AU14_r', 
                                'AU15_r',
                                'AU17_r', 
                                'AU20_r', 
                                'AU23_r',
                                'AU25_r', 
                                'AU26_r']]
    
        emg_index_labels = mapping
      
        labels = labels.append([labels]*(len(emg_index_labels)-1))
        emg_data = df_emg.loc[emg_index_labels][[
                                'Channel_1',
                                'Channel_2',
                                'Channel_3',
                                'Channel_4',
                                'Channel_5',
                                'Channel_6',
                                'Channel_7']]

        labels = labels.set_index(emg_data.index)
        concat = pd.concat([emg_data,labels],axis=1)

        #reorder columns
        concat = concat[['frame',
                        'Channel_1',
                        'Channel_2',
                        'Channel_3',
                        'Channel_4',
                        'Channel_5',
                        'Channel_6',
                        'Channel_7',
                        'AU01_r', 
                        'AU02_r',
                        'AU04_r', 
                        'AU05_r', 
                        'AU06_r', 
                        'AU07_r',
                        'AU09_r', 
                        'AU10_r',
                        'AU12_r',
                        'AU14_r', 
                        'AU15_r',
                        'AU17_r', 
                        'AU20_r', 
                        'AU23_r',
                        'AU25_r', 
                        'AU26_r',]]
        

        df_constructed = df_constructed.append(concat)
       
    return df_constructed
       





if __name__ == '__main__':

    xdf_file_path = get_xdf_file()
    df_frames,df_emg = extract_data(xdf_file_path)
    frame_mappings = extract_frame_mappings(df_frames,df_emg)
    df_labels,file_name = get_labels()
    df_constructed = construct_dataframe(frame_mappings,df_emg,df_labels)
    df_constructed.to_csv('Experimental_Setup/Dataset_preprocessed/'+file_name,index = False)

    








