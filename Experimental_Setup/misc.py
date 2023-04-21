'''
Miscellaneous functions which help us in visualizing data
'''
import cv2
from pathlib import Path
import numpy as np
import time
from vidgear.gears import WriteGear
import os

Instructions = """  Please press Ok after you have read the Instructions

1. Once you click on OK, you will be shown a white screen for about 3 seconds. During this interval you are expected to keep a neutral expression.

2. After the white screen dissapears, the Experiment window will be shown \n The experiment window shows three things (i) Stimulus (ii)Webcam Feed 

3. You are expected to practice to do  the expression shown in the stimulus video with the help of the stimulus window. 

3. Once you feel confident with your expression please press  the record button. 
 
4. You will be  immediately be shown your  webcam feed. Please  mimic the expression.

7. Once you reach the peak expression intensity , click the middle mouse button, the recording will begin now.

8. Hold the expression until the recording window dissapears by itself.

6. If you feel that  you would like redo the recording again for the expression,You can click on retry button.

7. All stimulus videos will be shown in sequential order followed by their recording"""

training_instructions = '''
Please press Ok after you have read the Instructions

1. Once you click on Ok. You will be shown a sequence of windows. Each window contains three things (i) Stimulus (ii)Webcam Feed (iii) Feedback 

2. You are expected to practice to do  the expression shown in the stimulus video with the help of the feedback window.

3. Once you feel confident with your expression please press  the next button. 

4. Now you wil be shown the next stimuli video.

5. You can practice all expressions by going through all the stimulus videos.
'''

def showInMovedWindow(winname, img, x, y,outlet,sample,frame_count=0,text = None,facet_au_value=None,size = (1280,720)):
    '''
    Create a named window at a specific position on screen
    '''
    cv2.namedWindow(winname)        # Create a named window
    cv2.moveWindow(winname, x, y)   # Move it to (x,y)
    img = cv2.resize(img,size)
    coordinates = (100,100)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (0,0,255)
    thickness = 2
    if facet_au_value is not None:
        img = cv2.putText(img,'Trackbar value:'+str(round(facet_au_value,2)),(100,130),font,fontScale,(0,255,0),thickness,cv2.LINE_AA)
    if text:
        img = cv2.putText(img, 'Recording', coordinates, font, fontScale, color, thickness, cv2.LINE_AA)
       
    cv2.imshow(winname,img)
    outlet.push_sample(sample)
    if text:
        frame_count += 1
    
    return frame_count

def get_stimuli_files():
    '''
    Function to get all stimuli files to be shown to the subject
    '''
    stimuli_files = list(Path('Experimental_setup/stimuli/').glob('*'))

    return stimuli_files

def get_existing_subjects():
    '''
    Functions helps in knowing how many subjects data is  already present which help
    '''

    return len(list(Path('Experimental_setup/recordings/video_recordings/').glob('*')))

def update_data(AUs,graph,fig,AU):
    '''
    Method to update the data
    '''
    for i,rect in enumerate(graph):
        rect.set_width(AU)

    # redraw the canvas
    fig.canvas.draw()

    # convert canvas to image
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
            sep='')
    img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # img is rgb, convert to opencv's default bgr
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

    return img

def change_recording_status(event,x,y,flags,params):

    if event == cv2.EVENT_MBUTTONDOWN:
        params[0] = True


def record_baseline(webcam,outlet):
    #white screen
    white_screen = cv2.VideoCapture('Experimental_setup/baseline_screen.mp4')
    frame_count = 1

    ret_white, frame_white_screen = white_screen.read()
    ret_record,frame_webcam = webcam.read()
    white_screen.set(cv2.CAP_PROP_FPS, 30)
    webcam.set(cv2.CAP_PROP_FPS, 30)

    baseline_frames   = []
    while ret_white:
        # Display the resulting frame
        baseline_frames.append(frame_webcam)
        sample = [frame_count,0,-1]
        showInMovedWindow('Baseline',frame_white_screen,880,400,outlet,sample,facet_au_value=None)
        ret_white,frame_white_screen = white_screen.read()
        ret_record,frame_webcam = webcam.read()
        key = cv2.waitKey(1)
        frame_count+=1

    white_screen.release()
    cv2.destroyAllWindows()

    return baseline_frames

def recording_prompt(webcam,outlet,AU):

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
	
    
    
    
    ret_record,frame_webcam = webcam.read()
    webcam.set(cv2.CAP_PROP_FPS, 30)
    recording_status = [False]
    recording_frames = []
    frame_count=1
    
    while frame_count<=150:

        
        if recording_status[0]:
            recording_frames.append(frame_webcam)
            sample = [frame_count,1,AU_DICT[AU]]
        else:
            sample = [0,1,AU_DICT[AU]]

        frame_count = showInMovedWindow('Record',frame_webcam,880,400,outlet,sample,frame_count,recording_status[0])
        if not recording_status[0]:
            cv2.setMouseCallback('Record',change_recording_status,recording_status)
        ret_record,frame_webcam = webcam.read()
        key = cv2.waitKey(1)



  
    cv2.destroyAllWindows()

    return recording_frames

def save_baseline_frames(directory,baseline_frames):
    if not os.path.isdir(directory):
        os.mkdir(directory)

    # define suitable (Codec,CRF,preset) FFmpeg parameters for writer
    output_params = {"-vcodec":"libx264", "-crf": 0, "-preset": "fast",'-fps':30}
    writer_baseline  = WriteGear(output= directory  + '/'+'baseline.mp4', logging = True, compression_mode=False,**output_params)

    for frame in baseline_frames:
        writer_baseline.write(frame)
    writer_baseline.close()

def save_recording_frames(directory,filename,baseline_frames,recorded_frames):
    # define suitable (Codec,CRF,preset) FFmpeg parameters for writer
    output_params = {"-vcodec":"libx264", "-crf": 0, "-preset": "fast",'-fps':30}
   
    writer_recording = WriteGear(output = directory + '/'+ filename+'_recording.mp4', logging = True,compression_mode=False, **output_params)

    for frame in recorded_frames:
        writer_recording.write(frame)

    writer_recording.close()
   
      








