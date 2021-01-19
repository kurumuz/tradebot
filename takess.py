import mss
import mss.tools
from PIL import Image
import time
import sys
import numpy
import cv2
import pyautogui

with mss.mss() as sct:
    monitor = {"top": 170, "left":560 , "width": 850, "height": 600}
    x = 0
    starttime = time.time()
    while (1):
        sct_img = sct.grab(monitor)
        #mss.tools.to_png(sct_img.rgb, sct_img.size, output = "ss/" + str(x) + ".png")
        x = x + 1 
        print(pyautogui.position())
        time.sleep(0.1 - ((time.time() - starttime) % 0.1))
        print("tick")
        if (x == 300):
            print(time.time() - starttime)
            sys.exit(0)
