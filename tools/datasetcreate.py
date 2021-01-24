import cv2
import torch
from PIL import Image
import time
import os

global sscount
files = os.listdir('char/')
if files:
    for z in range(0, len(files)):
        files[z] = int(files[z].strip('.png'))

    sscount = max(files) + 1

else:
    
    sscount = 0


imglist = os.listdir('ss/')
for imgpoint in range(2, len(imglist)):
    print(imglist[imgpoint])
    img = cv2.imread('ss/' + imglist[imgpoint])
    #grayscale
    img = cv2.resize(img ,None, fx = 5, fy = 5, interpolation = cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    cv2.waitKey(0)

    #binarize 
    ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
    cv2.waitKey(0)

    #find contours
    ctrs, hier = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, 
    cv2.CHAIN_APPROX_SIMPLE)

    #sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    start_time = time.time()
    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
        if h > 20:
            # Getting ROI
            if w < 18:
                x = int(x - w/1.2)
                if x < 0:
                    x = 0
                w = 27

            roi = gray[y:y+h, x:x+w]

            roi = cv2.resize(roi, (28, 28), interpolation = cv2.INTER_AREA)
            f = open("labels.txt", "a+")
            cv2.imshow("test", roi)
            cv2.waitKey(0)
            label = input(str(sscount) + ": ")
            f.write(label + "\n")
            cv2.imwrite("char/" + str(sscount) +".png", roi)
            sscount += 1
            
            cv2.rectangle(img,(x,y),( x + w, y + h ),(90,0,255),2)
            #print(str(w) + ' , ' +  str(h))
            cv2.waitKey(0)

    f.close()
    print(str(time.time() - start_time))
    cv2.imshow('marked areas',img)
    cv2.waitKey(0)