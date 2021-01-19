import mss
import mss.tools
from PIL import Image
import time
import sys
import numpy
import cv2
import pyautogui
import keyboard

with mss.mss() as sct:
    #buyorder first: Point(x=1254, y=386)
    #sellorder first: Point(x=979, y=389)
    #monitor = {"top": 360, "left":1015 , "width": 70, "height": 30} #sell1
    monitor = {"top": 416, "left": 458, "width": 58, "height": 21} #sell2

    #monitor = {"top": 360, "left":1360 , "width": 70, "height": 30} #buy2
    #monitor = {"top": 360, "left":1260 , "width": 70, "height": 30} #buy1
    x = 0
    starttime = time.time()
    while (1):
        if keyboard.is_pressed('q'):
            #monitor = {"top": 360, "left":1015 , "width": 70, "height": 40}
            sellorder = sct.grab(monitor)
            imgsell = Image.frombytes("RGB", sellorder.size, sellorder.bgra, "raw", "BGRX")
            imgsell.show()
            #x = x + 1


