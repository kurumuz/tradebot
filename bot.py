import cv2
import torch
from PIL import Image
import numpy as np
import mss
import mss.tools
from PIL import Image
import time
import sys
import numpy
import cv2
import keyboard
import win32api, win32com
import win32con

def click(x, y, sleep=0):
    win32api.SetCursorPos((x, y))
    time.sleep(sleep)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)
    time.sleep(0.1)

# Model
presentones = [0, 0, 0 , 0]
lockedone = -1
enabled = True
device = torch.device("cuda")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', classes=1)  # for file/URI/PIL/cv2/np inputs and NMS
model.load_state_dict(torch.load('centerbest.pt')['model'].state_dict())
model = model.fuse().autoshape()
model.to(device)
import os
imgs = []
url = 'ss'
starttime = time.time()
countertime = 0
first = True

with mss.mss() as sct:
    monitor = {"top": 184, "left":600 , "width": 700, "height": 500}
    per = 1
    while (1):
        counter = 0
        presentones = [0, 0, 0, 0]
        sct_img = sct.grab(monitor)
        result = model(sct_img, size=416)

        countertime += 1
        if (time.time() - starttime) > per:
            print("fps: {}".format(countertime / (time.time() - starttime)))
            countertime = 0
            starttime = time.time()

        if keyboard.is_pressed('q'):
            enabled = not enabled

        if keyboard.is_pressed('x'):
            sys.exit(0)

        if result.xyxy and enabled is True:
            counter = 0
            for match in result.xyxy[0]:
            #match = result.xyxy[0][0]
            #print(match)

                if float(match[4]) > 0.7:
                    print(f"{lockedone}")
                    print(presentones)
                    
                    #print(float(match[4]))
                    matchdata = match
                    xcenter, ycenter = matchdata[0] + (matchdata[2] - matchdata[0]) / 2, matchdata[1] + (matchdata[3] - matchdata[1]) / 2
                    xscreen, yscreen = 600+int(xcenter), 184+int(ycenter)
                    print(f"x: {xcenter} y: {ycenter}")
                    if xcenter > 0 and xcenter < 175:

                        if lockedone == -1:
                            lockedone = 0
                            first = True
                        if lockedone == 0:
                            if first:
                                click(xscreen, yscreen, sleep=0.3)
                                first = False
                            else:
                                click(xscreen, yscreen, sleep=0)
                            
                        presentones[0] += 1

                    else:
                        if lockedone == 0 and presentones[0] == 0 and counter == (len(result.xyxy)-1):
                            print("TEST:")
                            presentones[0] = 0
                            lockedone = -1
                    
                    if xcenter > 175 and xcenter < 350:

                        if lockedone == -1:
                            lockedone = 1
                            first = True
                        if lockedone == 1:
                            click(xscreen, yscreen)
                            if first:
                                click(xscreen, yscreen, sleep=0.3)
                                first = False
                            else:
                                click(xscreen, yscreen, sleep=0)
                        presentones[1] += 1

                    else:
                        if lockedone == 1 and presentones[1] == 0 and counter == (len(result.xyxy)-1):
                            print("TEST:")
                            presentones[1] = 0
                            lockedone = -1

                    if xcenter > 350 and xcenter < 525:

                        if lockedone == -1:
                            lockedone = 2
                            first = True
                        if lockedone == 2:
                            if first:
                                click(xscreen, yscreen, sleep=0.3)
                                first = False
                            else:
                                click(xscreen, yscreen, sleep=0)
                        presentones[2] += 1

                    else:
                        if lockedone == 2 and presentones[2] == 0 and counter == (len(result.xyxy)-1):
                            print("TEST:")
                            presentones[2] = 0
                            lockedone = -1
                    
                    if xcenter > 525 and xcenter < 700:

                        if lockedone == -1:
                            lockedone = 3
                            first = True
                        if lockedone == 3:
                            if first:
                                click(xscreen, yscreen, sleep=0.3)
                                first = False
                            else:
                                click(xscreen, yscreen, sleep=0)
                        presentones[3] += 1

                    else:
                        if lockedone == 3 and presentones[3] == 0 and counter == (len(result.xyxy)-1):
                            print("TEST:")
                            presentones[3] = 0
                            lockedone = -1
                    counter += 1

                    
                    #xscreen, yscreen = 600+int(xcenter), 184+int(ycenter)
                    #click(xscreen, yscreen, 0.1)
                    #click(xscreen, yscreen)

                #break



            


        #time.sleep(0.001)

'''
for img in os.listdir(url):
    #imgs.append(Image.open(url + "/" + img).convert('RGB'))
    img = Image.open(url + "/" + img).convert('RGB')
    result = model(img, size=416)
    img = np.array(img)
    img = img[:, :, ::-1].copy() 
    print(result.xyxy)
    if result.xyxy:
        for match in result.xyxy[0]:
            img2 = img.copy()
            matchdata = match
            xcenter, ycenter = matchdata[0] + (matchdata[2] - matchdata[0]) / 2, matchdata[1] + (matchdata[3] - matchdata[1]) / 2
            print(f"{xcenter}, {ycenter}")
            cv2.circle(img, (int(xcenter), int(ycenter)), 6, (0, 255, 255), -1)
            cv2.imshow("OpenCV/Numpy normal", img)
            cv2.waitKey(0)
            img = img2.copy()

'''

# Inference
'''
result = model(imgs[0], size=416)  # includes NMS
matchdata = result.xyxy[0][0]
xcenter, ycenter = matchdata[0] + (matchdata[2] - matchdata[0]) / 2, matchdata[1] + (matchdata[3] - matchdata[1]) / 2
print(f"{xcenter}, {ycenter}")
img = np.array(imgs[0])
img = img[:, :, ::-1].copy() 
cv2.circle(img, (int(xcenter), int(ycenter)), 6, (0, 255, 255), -1)
cv2.imshow("OpenCV/Numpy normal", img)
cv2.waitKey(0)
# Results
#result.print()  # print results to screen
#result.show()  # display results
#results.save()  # save as results1.jpg, results2.jpg... etc.
'''

# Data

#print(result.xyxy[0][0])  # print img1 predictions
#          x1 (pixels)  y1 (pixels)  x2 (pixels)  y2 (pixels)   confidence        class
# tensor([[7.47613e+02, 4.01168e+01, 1.14978e+03, 7.12016e+02, 8.71210e-01, 0.00000e+00],
#         [1.17464e+02, 1.96875e+02, 1.00145e+03, 7.11802e+02, 8.08795e-01, 0.00000e+00],
#         [4.23969e+02, 4.30401e+02, 5.16833e+02, 7.20000e+02, 7.77376e-01, 2.70000e+01],
#         [9.81310e+02, 3.10712e+02, 1.03111e+03, 4.19273e+02, 2.86850e-01, 2.70000e+01]])