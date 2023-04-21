import cv2
import argparse
from pathlib import Path

def showInMovedWindow(winname, img, x, y):
    cv2.namedWindow(winname)        # Create a named window
    cv2.moveWindow(winname, x, y)   # Move it to (x,y)
    img = cv2.resize(img,(900,480))
    cv2.imshow(winname,img)

videofiles = list(Path('Demonstration/Videos').glob("*.mp4"))

print("Choose file for comparision")
for i,file in enumerate(videofiles):
    print(f"{i}:{file.parts[-1]}")

index = int(input("Index:"))

file = videofiles[index].parts[-1].split('.')[0]
print(f'Chosen file: {file}')
print(f'filepath:{str(videofiles[index])}')

vis_software_dict = {0:'Blender',1:'Unity',2:'MakeHuman'}
vis_software_index = int(input('Choose your visualization software \n 0:Blender \n 1:Unity \n 2:MakeHuman \n'))
vis_software = vis_software_dict[vis_software_index]

render_files = list(Path('Demonstration/Rendered_Animations/'+vis_software).glob(f"*{file}.mkv"))

vid1 = cv2.VideoCapture(str(render_files[0]))
vid2 = cv2.VideoCapture(str(render_files[1]))
try:
    vid3 = cv2.VideoCapture(str(render_files[2]))
except:
    pass
vid4 = cv2.VideoCapture(str(videofiles[index]))

while True:

    f1,frame1 = vid1.read()
    f2,frame2 = vid2.read()
    try:
        f3,frame3 = vid3.read()
    except:
        pass

    f4,frame4 = vid4.read()

    if f1:
        showInMovedWindow(render_files[0].parts[-1].split('_')[0],frame1,0,0)
    else:
        vid1.set(cv2.CAP_PROP_POS_FRAMES,0)
        
    if f2:
        showInMovedWindow(render_files[1].parts[-1].split('_')[0],frame2,940,0)
    else:
        vid2.set(cv2.CAP_PROP_POS_FRAMES,0)
    
    try:
        if f3:
            showInMovedWindow(render_files[2].parts[-1].split('_')[0],frame3,0,500)
        else:
            vid3.set(cv2.CAP_PROP_POS_FRAMES,0)
    except:
        pass
    
    if f4:
        showInMovedWindow(file,frame4,940,500)
    else:
        vid4.set(cv2.CAP_PROP_POS_FRAMES,0)


    
    key = cv2.waitKey(1)

    if key == ord('q'):
        break


vid1.release()
vid2.release()
try:
    vid3.release()
except:
    pass
vid4.release()
cv2.destroyAllWindows()
























