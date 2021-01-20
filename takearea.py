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

print("bas")

x = 0
posleft = ""
starttime = time.time()#q

files = os.listdir('ss/')
if files:
    for z in range(0, len(files)):
        files[z] = int(files[z].strip('.png'))

    y = max(files) + 1

else:
    y = 0

def on_press(key):  # The function that's called when a key is pressed
    global x
    global y
    global posleft

    if key == Key.esc:
        sys.exit(0)

    if 'char' in dir(key):

        if key.char == 'q':
            if x == 0:
                posleft = pyautogui.position()
                print(posleft)
                x = 1
            else:

                poscurr = pyautogui.position()
                print(f'"top": {posleft[1]}, "left": {posleft[0]}, "width": {poscurr[0]-posleft[0]}, "height": {poscurr[1]-posleft[1]}')
                monitor = {"top": posleft[1], "left": posleft[0], "width": poscurr[0]-posleft[0], "height": poscurr[1]-posleft[1]}
                with mss.mss() as sct:
                    ss = sct.grab(monitor)
                    mss.tools.to_png(ss.rgb, ss.size, output = "ss/" + str(y) + ".png")
                y += 1
                x = 0
                posleft = ""

        if key.char == 'x':
            x = 0
            posleft = ""

with Listener(on_press=on_press) as listener:  # Create an instance of Listener
    listener.join()  # Join the listener thread to the main thread to keep waiting for keys


