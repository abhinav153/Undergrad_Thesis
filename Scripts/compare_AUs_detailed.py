"""
Script for comparing AUs generated by different AU recognition software
"""

import cv2
from pathlib import Path
import pandas as pd
from matplotlib import style
import matplotlib.pyplot as plt
import numpy as np
from misc import validate_frames

def showInMovedWindow(winname, img, x, y,frameno=0):
    '''
    Method for generating windows to display the data/video
    '''
    cv2.namedWindow(winname)        # Create a named window
    cv2.moveWindow(winname, x, y)   # Move it to (x,y)
    img = cv2.resize(img,(600,600))
    text = 'Frame No: '+str(frameno) + '_'+winname
    coordinates = (0,30)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (0,0,255)
    thickness = 1
    img = cv2.putText(img, text, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow(winname,img)

def render_screen(graph1,graph2,graph3,csv1,csv2,csv3,fig1,fig2,fig3,frame_count):
    '''
    Render all new windows after the updates are done
    '''
    label1 = ['AU1','AU2','AU4','AU5','AU6','AU7','AU9','AU10','AU12','AU14','AU15','AU17','AU18','AU20','AU23','AU24','AU25','AU26','AU27','AU43']
    label2 = ['AU1','AU2','AU4','AU5','AU6','AU7','AU9','AU10','AU12','AU14','AU15','AU17','AU18','AU20','AU23','AU24','AU25','AU26','AU28','AU43']
    label3 = ['AU1','AU2','AU4','AU5','AU6','AU7','AU9','AU10','AU12','AU14','AU15','AU17','AU20','AU23','AU25','AU26','AU45']

    graph1 = update_data(csv1,frame_count,graph1,label1)
    graph2 = update_data(csv2,frame_count,graph2,label2)
    graph3 = update_data(csv3,frame_count,graph3,label3)

    img1   = update_image(fig1)
    img2   = update_image(fig2)
    img3   = update_image(fig3)


    showInMovedWindow(file,frame,0,0,frame_count)
    showInMovedWindow(csv_files[0].parts[-1].split('_')[0],img1,1280,0,frame_count)
    showInMovedWindow(csv_files[1].parts[-1].split('_')[0],img2,0,800,frame_count)
    showInMovedWindow(csv_files[2].parts[-1].split('_')[0],img3,1280,800,frame_count)

def update_data(csv_data,frame_count,graph,label):
    '''
    Method to extract the relevant AU data corresponding to a particular frame
    '''
    AU = csv_data[csv_data['frame']==frame_count][label]
    AU =  AU.values
    AU = AU.tolist()[0]
    
    for i,rect in enumerate(graph):
        rect.set_width(AU[i])

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

#read the video files you want to visualize
videofiles = list(Path('Demonstration/Videos').glob("*.mp4"))

print("Choose file for comparision")
for i,file in enumerate(videofiles):
    print(f"{i}:{file.parts[-1]}")

index = int(input("Index:"))

file = videofiles[index].parts[-1].split('.')[0]
print(f'Chosen file: {file}')
print(f'filepath:{str(videofiles[index])}')

#get csv filepaths for corresponding chosen video
csv_files = list(Path('Demonstration/Post-processed_non_facsvatar/').glob(f"*{file}.csv"))

#read csv files
csv1 = pd.read_csv(csv_files[0])
file1 = csv_files[0].parts[-1]

csv2 = pd.read_csv(csv_files[1])
file2 = csv_files[1].parts[-1]

csv3 = pd.read_csv(csv_files[2])
file3 = csv_files[2].parts[-1]

csv2 = validate_frames(csv1,csv2,csv3) #if Imotions has less frames, add some duplicate rows towards the bottom


#read template vide0
vid = cv2.VideoCapture(str(videofiles[index]))
total_frames = total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

frame_count = 0

style.use('fivethirtyeight')

fig1 = plt.figure()
fig2 = plt.figure()
fig3 = plt.figure()
ax1 = fig1.add_subplot(1,1,1)
ax2 = fig2.add_subplot(1,1,1)
ax3  = fig3.add_subplot(1,1,1)

#FaceReader labels
label1 = [ 'Action Unit 01 - Inner Brow Raiser',
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
            'Action Unit 27 - Mouth Stretch',
            'Action Unit 43 - Eyes Closed']

#Imotion labels
label2 = [ 'Action Unit 01 - Inner Brow Raiser',
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
            'Action Unit 28 - Lip Suck',
            'Action Unit 43 - Eyes Closed']

#Open Face Labels
label3 = [ 'Action Unit 01 - Inner Brow Raiser',
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
            'Action Unit 20 - Lip Stretcher',
            'Action Unit 23 - Lip Tightener',
            'Action Unit 25 - Lips Part', 
            'Action Unit 26 - Jaw Drop',
            'Action Unit 45 - Blink']

data1 = [ 0 for i in range(len(label1)) ]
data2 = [ 0 for i in range(len(label2)) ]
data3 = [ 0 for i in range(len(label3)) ]
graph1 = ax1.barh(label1,data1)
graph2 = ax2.barh(label2,data2)
graph3 = ax3.barh(label3,data3)
ax1.set_xbound(0,1)
ax2.set_xbound(0,1)
ax3.set_xbound(0,1)
fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()

pause = False


while True:

    if pause is not True:
        _,frame = vid.read()

        if _:
            render_screen(graph1,graph2,graph3,csv1,csv2,csv3,fig1,fig2,fig3,frame_count)
            frame_count+=1
        else:
            vid.set(cv2.CAP_PROP_POS_FRAMES,0)
            frame_count = 0

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    elif key == ord('p'):
        pause = not pause

    elif key == ord('n') and pause:
        print('Detected command for next key frame')
        frame_count+=1
        if frame_count > total_frames-1:
            frame_count=0
        vid.set(cv2.CAP_PROP_POS_FRAMES,frame_count)
        
        _,frame = vid.read()
       
        render_screen(graph1,graph2,graph3,csv1,csv2,csv3,fig1,fig2,fig3,frame_count)
       
    elif key == ord('b') and pause:
        print('Detected command for previous key frame')
        frame_count-=1
        if frame_count < 0:
            frame_count = total_frames-1 
        vid.set(cv2.CAP_PROP_POS_FRAMES,frame_count)
        
        _,frame = vid.read()

        render_screen(graph1,graph2,graph3,csv1,csv2,csv3,fig1,fig2,fig3,frame_count)
    else:
        continue

vid.release()