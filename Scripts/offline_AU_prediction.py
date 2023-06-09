import sys
sys.path.append('.\\')
import pyxdf
import pandas as pd
from Models.preprocess import Preprocess
from Models.features import FeatureConstructor,TimeDomain,FrequencyDomain
import pickle
from pathlib import Path
from copy import deepcopy
import numpy as np
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report
import matplotlib.pyplot as plt

xdf_file_path = 'Scripts\offline_recordings\AU1.xdf'
xdf_file_paths = Path( 'Scripts\\offline_recordings\\').glob('*.xdf')
#load your model
model = pickle.load(open('Models/saved_models/RF_400_raw.sav','rb'))

y_predicted = np.empty((0))
y_true      = np.empty((0))

for xdf_file_path in xdf_file_paths:
    print(f'.....processing:{xdf_file_path}')
    data, header = pyxdf.load_xdf(xdf_file_path)
    df_emg = pd.DataFrame(data={'Zygomaticus_Major':[i[4] for i in data[0]['time_series']],
                                        'Levator_Labi':[i[2] for i in data[0]['time_series']],
                                        'Orbicularis_Oculi':[i[1] for i in data[0]['time_series']],
                                        'Corrugator_Supercili':[i[3] for i in data[0]['time_series']],                     
                                        'Timestamps': data[0]['time_stamps'],
                                        'Push Button':[i[5] for i in data[0]['time_series']]})

    df_emg = df_emg[df_emg['Push Button']>0.9][['Zygomaticus_Major','Levator_Labi','Corrugator_Supercili','Orbicularis_Oculi']]
    emg_array = df_emg.to_numpy()


    features_dict = {
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

    segmented= Preprocess.segmentation_only_ts(emg_array,2000,400)
    features,labels = FeatureConstructor.construct_features(segmented,features_dict,2000,['Zygomaticus_Major','Levator_Labi','Corrugator_Supercili','Orbicularis_Oculi'])

    pred_label = model.predict(features)
    predictions = model.predict_proba(features)
    pred_dataframe = pd.DataFrame(data={'AU1':predictions[:,0],
                                        'AU2':predictions[:,1],
                                        'AU4':predictions[:,2],
                                        'AU5':predictions[:,3],
                                        'AU6':predictions[:,4],
                                        'AU7':predictions[:,5],
                                        'AU9':predictions[:,6],
                                        'AU10':predictions[:,7],
                                        'AU12':predictions[:,8],
                                        'AU14':predictions[:,9],
                                        'AU15':predictions[:,10],
                                        'AU17':predictions[:,11],
                                        'AU18':predictions[:,12],
                                        'AU20':predictions[:,13],
                                        'AU23':predictions[:,14],
                                        'AU24':predictions[:,15],
                                        'AU25':predictions[:,16],
                                        'AU26':predictions[:,17],
                                        'AU43':predictions[:,18],
                                        'Predicted Label':pred_label})
    #print(pred_dataframe[['Predicted Label',xdf_file_path.parts[-1].split('.')[0]]])
    actual_label = deepcopy(pred_label)
    actual_label[:] = float(xdf_file_path.parts[-1].split('.')[0].split('U')[-1])
    y_true = np.concatenate([y_true,actual_label],axis=0)
    y_predicted = np.concatenate([y_predicted,pred_label],axis=0)

cm = confusion_matrix(y_true,y_predicted,)
display=ConfusionMatrixDisplay(cm,display_labels=[int(i) for i in model.classes_])
display.plot()
plt.gcf().savefig('Resources/model_results/confusion_matrix.png')

plt.figure()
cr = classification_report(y_true,y_predicted,target_names=[int(i) for i in model.classes_],output_dict=True)
cr_dataframe = pd.DataFrame(cr).transpose()
print(cr_dataframe.to_latex())
cr_dataframe.to_csv('Resources/model_results/cr.csv',index=False)
with open('Resources/model_results/cr_latex.txt','w') as tf:
    tf.write(cr_dataframe.to_latex())




   