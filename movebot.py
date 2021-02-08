import cv2
import torch
from PIL import Image
import numpy as np
import mss
import mss.tools
from PIL import Image
import sys
import numpy
import cv2
import keyboard
import win32api, win32com
import win32con, win32clipboard
import os
import pyautogui
import pytesseract
from matplotlib import pyplot as plt
from pynput.keyboard import Listener, Key
import time
from threading import Event, Thread
import threading
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import math

def get_ss(x, y, w, h):
    monitor = {"top": y, "left": x, "width": w, "height": h}
    with mss.mss() as sct:
        ss = sct.grab(monitor)
        ss = np.array(ss)
        #ss = cv2.resize(ss ,None, fx = 5, fy = 5, interpolation = cv2.INTER_CUBIC)
    return ss

def click(x, y, sleep=0):
    win32api.SetCursorPos((x, y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)
    time.sleep(sleep)

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


img = cv2.imread("testmap2.png")
img = rotate_image(img, 35)
cv2.imshow("test", img)
#cv2.waitKey(0)
frame = img
#img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#lower_blue= np.array([63,116,166])
#upper_blue = np.array([138,255,255])
lower_blue = np.array([255,160,80])
upper_blue = np.array([255,200,110])


mask = cv2.inRange(img,lower_blue,upper_blue)
cv2.imshow("mask", mask)

cnts, hier = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
cv2.drawContours(frame,cnts,-1,(0,255,0),3)
sorted_ctrs = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0])

##### templatec
xm, ym = 0, 0
framegray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
tmp0 = cv2.imread("flag3.png", 0)
cv2.imshow("ta", tmp0)
cv2.waitKey(0)
w, h = tmp0.shape[::-1]
method = eval('cv2.TM_SQDIFF_NORMED')
res0 = cv2.matchTemplate(framegray, tmp0, method)
min_val0, max_val0, min_loc0, max_loc0 = cv2.minMaxLoc(res0)
xf, yf = min_loc0[0], min_loc0[1]
if (1 - min_val0) < 1:
    if (1 - min_val0) > 0.8:
        cv2.rectangle(frame,(xf,yf),( xf + w, yf + h ),(90,0,255),2)
        xf = xf + w/2
        yf = yf + h/2

for i, ctr in enumerate(sorted_ctrs):
    xm, ym, w, h = cv2.boundingRect(ctr)
    cv2.rectangle(frame,(xm,ym),( xm + w, ym + h ),(90,0,255),2)
    #cv2.imshow("ctr", frame)
    xm, ym = xm + w/2, ym + h/2

print(f"flag: {xf}, {yf}\n me: {xm}, {ym}")

cv2.imshow("Frame",frame)
cv2.waitKey(0)

#go left until there is 2 pixel difference
#go up until same cond
'''
Point(x=693, y=263) x left
Point(x=1432, y=754) x right
Point(x=1257, y=256) y up
Point(x=605, y=661) y down
'''

'''
while True:
    
    if xm - xf > 2:
        click(693, 263, 0.4)
'''