import sys
sys.path.append('.\\')
sys.path.append(sys.path[0]+'\\modules')

import pickle
from matplotlib import style
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pylsl import StreamInlet,resolve_stream
import pandas as pd
from Models.features import FeatureConstructor
from Models.features import TimeDomain,FrequencyDomain
from modules.gui.controller import Controller

def showInMovedWindow(winname, img, x, y,text=None):
    '''
    Method for generating windows to display the data/video
    '''
    cv2.namedWindow(winname)        # Create a named window
    cv2.moveWindow(winname, x, y)   # Move it to (x,y)
    img = cv2.resize(img,(800,500))
    if text:
        text = 'Predicted AU:'+str(text)
    print(text,'.....')
    coordinates = (0,30)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (0,0,255)
    thickness = 1
    img = cv2.putText(img, text, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow(winname,img)

def update_graph(predictions,graph):
    for i,rect in enumerate(graph):
        rect.set_width(predictions[i])
    
    return graph
def update_image(fig):
    '''
    Display new frame  with new data after updating AU data
    '''
     # redraw the canvas
    fig.canvas.draw()

    # convert canvas to image
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
            sep='')
    img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # img is rgb, convert to opencv's default bgr
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

    return img

#load your model
segment_length=400
model = pickle.load(open('Models/saved_models/RF_'+str(segment_length)+'_raw.sav','rb'))
#store index for classes
class_index={}
for i,class_ in enumerate(model.classes_):
    class_index[int(class_)] = i

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
multiplier = 2



print("looking for an EMG stream...")
streams = resolve_stream()

# create a new inlet to read from the stream
inlet = StreamInlet(streams[0],max_chunklen=(2*segment_length))
print(inlet.info().type())

#capture = cv2.VideoCapture(0)
style.use('fivethirtyeight')
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
labels =[ 'Action Unit 01 - Inner Brow Raiser',
            'Action Unit 02 - Outer Brow Raiser', 
            'Action Unit 04 - Brow Lowerer',
            'Action Unit 05 - Upper Lid Raiser', 
            'Action Unit 06 - Cheek Raiser',
            'Action Unit 07 - Lid Tightener', 
            'Action Unit 09 - Nose Wrinkler',
            'Action Unit 10 - Upper Lip Raiser',
            'Action Unit 12 - Lip Corner Puller', 
            'Action Unit 14 - Dimpler',
            'Action Unit 15 - Lip Corner Depressor', 
            'Action Unit 17 - Chin Raiser',
            'Action Unit 18 - Lip Pucker',
            'Action Unit 20 - Lip Stretcher',
            'Action Unit 23 - Lip Tightener',
            'Action Unit 24 - Lip Pressor',
            'Action Unit 25 - Lips Part', 
            'Action Unit 26 - Jaw Drop',
            'Action Unit 43 - Eyes Closed']
data = [ 0 for i in range(len(labels)) ]
graph = ax.barh(labels,data)
ax.set_xbound(0,1)
fig.tight_layout()


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
    
    #_,frame = capture.read()

    prediction_label = -1
    key= cv2.waitKey(1)
    if key == ord('q'):
        break
    

    samples,timestamps = inlet.pull_chunk()
    if samples:
        
        array = np.array(samples)
        array = array[:,1:5]
        array = np.expand_dims(array,axis=0)
        
        x_sample,labels=FeatureConstructor.construct_features(array,features,2000,['a','b','c','d'])
        try:
            prediction_label = model.predict(x_sample)
            prediction_prob  = model.predict_proba(x_sample)
            series = pd.Series(prediction_prob[0],index=model.classes_)
            
            AUs['AU01'] = series.loc[1.0] * multiplier
            AUs['AU02'] = series.loc[2.0] * multiplier
            AUs['AU04'] = series.loc[4.0]* multiplier
            AUs['AU06'] = series.loc[6.0]* multiplier
            AUs['AU07'] = series.loc[7.0]* multiplier
            AUs['AU09'] = series.loc[9.0]* multiplier
            AUs['AU10'] = series.loc[10.0]* multiplier
            AUs['AU12'] = series.loc[12.0]* multiplier
            AUs['AU14'] = series.loc[14.0]* multiplier
            AUs['AU15'] = series.loc[15.0]* multiplier
            AUs['AU17'] = series.loc[17.0]* multiplier
            AUs['AU20'] = series.loc[20.0]* multiplier
            AUs['AU23'] = series.loc[23.0]* multiplier
            AUs['AU25'] = series.loc[25.0]* multiplier
            AUs['AU26'] = series.loc[26.0]* multiplier
    
            
            print

            controller.face_configuration(AUs)

            graph = update_graph(prediction_prob[0],graph)
            img = update_image(fig)
            showInMovedWindow('graph',img,1000,100,)
        except Exception as e:
            print('Some error occured')
            print(e)
            continue
           
            
    

cv2.destroyAllWindows()
#capture.release()

        