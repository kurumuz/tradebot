import mss
import mss.tools
from PIL import Image
import time
import sys
import numpy
import cv2
import pyautogui
import keyboard
import os 
import logging
from pynput.keyboard import Listener, Key
global x
global posleft
global y

print("started")

x = 0
posleft = ""
starttime = time.time()#q

files = os.listdir('objects/')
if files:
    for z in range(0, len(files)):
        files[z] = int(files[z].strip('.png'))

    y = max(files) + 1

else:
    y = 0

def on_press(key):  # The function that's called when a key is pressed
    global x
    global y

    if key == Key.esc:
        sys.exit(0)

    if 'char' in dir(key):
        if key.char == 'q':

            with mss.mss() as sct:
                #mss.tools.to_png(ss.rgb, ss.size, output = "ss/" + str(y) + ".png") # if you dont want to scale the image
                filename = sct.shot(output="objects/" + str(y) + ".png")
                #ss = numpy.array(ss)
                #ss = cv2.resize(ss ,None, fx = 5, fy = 5, interpolation = cv2.INTER_CUBIC)
                #cv2.imwrite("objects/" + str(y) + ".png", ss)

            y += 1



with Listener(on_press=on_press) as listener:  # Create an instance of Listener
    listener.join()  # Join the listener thread to the main thread to keep waiting for keys
