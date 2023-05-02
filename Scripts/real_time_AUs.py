from pylsl import StreamInlet,resolve_stream
import pickle
import numpy as np
import sys
sys.path.append('.\\')
sys.path.append(sys.path[0]+'\\modules')
from Models.features import FeatureConstructor
from Models.features import TimeDomain,FrequencyDomain
from modules.gui.controller import Controller

#load your model
model = pickle.load(open('Models/saved_models/RF_400_raw.sav','rb'))
features = {
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

#start instance of facsvatar publishing port
#adapted
controller = Controller(pub_ip="127.0.0.1", pub_port=5570, pub_key="gui.face_config", pub_bind=False,
                        deal_ip="127.0.0.1", deal_port=5580, deal_key="gui", deal_topic="multiplier",
                        deal2_ip="127.0.0.1", deal2_port=5581, deal2_key="gui", deal2_topic="dnn",
                        deal3_ip="127.0.0.1", deal3_port=5582, deal3_key="gui", deal3_topic="dnn"
                       )



print("looking for an EMG stream...")
streams = resolve_stream()

# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])
print(inlet.info().type())


# get a new sample (you can also omit the timestamp part if you're not
# interested in it)
buffer = []
while True:

    AUs={'AU01': 0, 
     'AU02': 0, 
     'AU04': 0, 
     'AU05': 0, 
     'AU06': 0, 
     'AU07': 0, 
     'AU09': 0, 
     'AU10': 0, 
     'AU12': 0, 
     'AU14': 0, 
     'AU15': 0, 
     'AU17': 0, 
     'AU20': 0, 
     'AU23': 0, 
     'AU25': 0, 
     'AU26': 0, 
     'pose_Rx': 0, 
     'pose_Ry': 0, 
     'pose_Rz': 0}
    

    sample, timestamp = inlet.pull_sample()
    if sample:
        if len(buffer)<400:
            buffer.append(sample[1:5])
        else:
            buffer.pop(0)
            buffer.append(sample[1:5])
            array = np.array(buffer)
            array = np.expand_dims(array,axis=0)
            x_sample,labels=FeatureConstructor.construct_features(array,features,2000,['a','b','c','d'])
            prediction_prob = model.predict_proba(x_sample)
            prediction_label = model.predict(x_sample)
            print(prediction_prob)
            print(prediction_label)
            print(model.classes_)
            if prediction_label <10:
                AUs[f'AU{int(prediction_label):02d}'] = 1 
            else:
                AUs[f'AU{int(prediction_label)}'] = 1 

            #print(AUs)
            controller.face_configuration(AUs)


        