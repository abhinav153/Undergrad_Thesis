import tkinter as tk
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from Imotions_receive_data import FACET

from misc import *
import misc
import time
from pylsl import StreamInfo,StreamOutlet




def show_webcam_frame():

    _, frame = webcam.read()
    if _:
        pass
    else:
        webcam.set(cv2.CAP_PROP_POS_FRAMES,0)
        _,frame = webcam.read()
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img,master=root)
    label_webcam.imgtk = imgtk
    label_webcam.configure(image=imgtk)
    
    
    

def show_stimulus_frame(stimulus_playback):

    _, frame = stimulus.read()
    if _:
        pass
    else:
        stimulus.set(cv2.CAP_PROP_POS_FRAMES,0)
        _,frame = stimulus.read()
    if stimulus_playback:
        frame = cv2.flip(frame, 1)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img,master=root,)
        label_stimulus.imgtk = imgtk
        label_stimulus.configure(image=imgtk)
    else:
        pass

def show_facet_frame(facet_connection,AU):
    AUs = facet_connection.extract_AU_data()
    print(list(AUs.keys()))
    updated_img = update_data(AUs,graph,fig,AUs[AU.upper()])
    cv2image = cv2.cvtColor(updated_img, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img,master=root,)
    label_facet.imgtk = imgtk
    label_facet.configure(image=imgtk)

def change_stimulus_playback():
    global stimulus_playback
    stimulus_playback = not stimulus_playback

def start_recording():
    global recording_call
    recording_call = not recording_call


   
   

if __name__=='__main__':

    width, height = 800, 600
    
    #get a webcam feed
    webcam = cv2.VideoCapture(0)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    #open a facet connection
    #facet = FACET()
    #facet.open_connection()

    #setup the graph whose data will be updated
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    labels = ['Intensity']    
    Instructions = {'AU01':'Inner brow raiser', #1
        'AU02':'Outer brow raiser',     #2
        'AU04': 'Brow lowerer',          #3
        'AU05':'Upper lid raiser',      #4
        'AU06':'Cheek raiser',          #5
        'AU07':'Lid tightener',         #6
        'AU09':'Nose wrinkler',         #7
        'AU10':'Upper lip raiser',      #8
        'AU12':'Lip corner puller',     #9
        'AU14':'Dimpler',               #10
        'AU15':'Lip corner depressor',  #11
        'AU17':'Chin raiser',           #12
        'AU18':'Lip Puckerer',           #13
        'AU20':'Lip stretcher',         #14
        'AU23':'Lip tightener',         #15
        'AU24':'Lip Pressor',            #16
        'AU25':'Lips part',             #17
        'AU26':'Jaw drop',              #18
        'AU28':'Lip Suck',              #19
        'AU43':'Eyes Closed'}
    init_data = [ 0  ]
    graph = ax.barh(labels,init_data)
    ax.set_xbound(0,1)
    ax.tick_params(axis='x',labelsize = 20)
    ax.tick_params(axis='y',labelsize=20)
    fig.tight_layout()

    #set up lsl tools
    info_vid = StreamInfo(
                name = 'VideoStream',
                type = 'VIDEO',
                channel_format= 'int32',
                channel_count = 3,
                source_id = 'myuid342',
                nominal_srate=30
                )
    
    outlet = StreamOutlet(info_vid)

    #create a tkinter setup
    root = tk.Tk()
    root.geometry("+800+500") #size of window , x-axis, yaxis

    root.rowconfigure(0,weight=1)
    root.columnconfigure(0,weight=1)

    #create a frame for webcam video
    frame_webcam = tk.Frame(root)
    frame_webcam.grid(row=0 ,column=1,padx=10,pady=10)
    label_webcam = tk.Label(frame_webcam)
    label_webcam.grid()

    #create a frame for stimulus video
    frame_stimulus = tk.Frame(root)
    frame_stimulus.grid(row = 0,column=0,padx=10,pady=10)
    label_stimulus = tk.Label(frame_stimulus)
    label_stimulus.grid()

    #create a frame for showing FACET output
    frame_facet = tk.Frame(root)
    frame_facet.grid(row=1,column=1,padx=10,pady=10)
    label_facet = tk.Label(frame_facet)
    label_facet.grid()

    
    #create a frame for all buttons
    frame_buttons = tk.Frame(master=root)
    frame_buttons.grid(row=1,column=0,padx=10,pady=10)
    record_button = tk.Button(frame_buttons, text='Record',fg="red",command=start_recording)
    stimulus_playback = tk.Button(frame_buttons,text='Start/Stop stimulus',command = change_stimulus_playback)
    
    stimulus_playback.grid()
    record_button.grid()
  

    #boolean variables
    stimulus_playback = True
    recording_call    = False


    result = tk.messagebox.showinfo("Instructions",misc.Instructions)

    #get stimulus feed
    stimulus_files = get_stimuli_files()
    index = 0
    subject_no = str(get_existing_subjects() + 1)

    baseline_frames = record_baseline(webcam,outlet)
    save_baseline_frames('Experimental_setup/recordings/video_recordings/subject_'+subject_no,baseline_frames)
    while index < len(stimulus_files):
        stimulus_file = stimulus_files[index]
        stimulus = cv2.VideoCapture(str(stimulus_file)) 
        print(stimulus_file)
        AU = stimulus_file.parts[-1].split('_')[2]
        root.title('Stimulus '+AU)
        ax.set_ylabel(AU)
        while recording_call is not True:
            show_webcam_frame()
            show_stimulus_frame(stimulus_playback)
                    
            #show_facet_frame(facet,AU)
            root.update()
            time.sleep(1/30)

        root.withdraw()
        #show recording loop
        recording_frames = recording_prompt(webcam,outlet,AU)

        result = tk.messagebox.askretrycancel("Record", "Would you like to record again?")
        if result:
            continue
        else:
            save_recording_frames('Experimental_setup/recordings/video_recordings/subject_'+subject_no,AU,baseline_frames,recording_frames)
            index+=1

        root.deiconify()
        recording_call = not recording_call
    
    tk.messagebox.showinfo("Thank you","Thank you, the trial is over")


    #facet.close_connection()
   