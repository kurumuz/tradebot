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
import os
import pyautogui
import pytesseract
from matplotlib import cm
from pynput.keyboard import Listener, Key
import time
from threading import Event, Thread
import threading

def main():
    global enabled
    enabled = True
    exitloop = RepeatedTimer(0.1, exitfunc)
    tmp0 = cv2.imread('images/ss1.png', 0) #read the button image

    while enabled:
        isButton = clickbutton(tmp0)

        if isButton:
            get_item_info()
        
        time.sleep(0.5)
    
def clickbutton(tmp0):
    w, h = tmp0.shape[::-1]
    method = eval('cv2.TM_SQDIFF_NORMED')

    if enabled is True:
        start_timex = time.time()
        with mss.mss() as sct:
            monitor = {"top": 170, "left":560 , "width": 850, "height": 600}
            img = cv2.cvtColor(np.array(sct.grab(monitor)), cv2.COLOR_BGR2GRAY)
            res0 = cv2.matchTemplate(img, tmp0, method)
        
        min_val0, max_val0, min_loc0, max_loc0 = cv2.minMaxLoc(res0)
        sim0 = sim_score(min_val0, max_val0, method)
        

        if sim0 > 0.92:
            x = int(560 + int(min_loc0[0]) + int(w/2))
            y = int(170 + int(min_loc0[1]) + int(h/2))
            click(x, y, 0.6)
            print(sim0)
            return True
            
        else:
            return False
    
def get_item_info():
    click(572, 402, 0.2) #tier menüsünü aç
    lowtier = normalizenumber(getstring([416, 458, 58, 21]).split(' ')[1])
    print(f"Tier: {lowtier}")
    totier = 8-int(lowtier)+1
    tierstartcoord = [536, 428, 58, 21]
    enchstartcoord = [665, 431]
    goodstartcoord = [814, 429]
    goodness = ["Normal", "Good", "Outstanding", "Excellent", "Masterpiece"]
    #click(tierstartcoord[0], tierstartcoord[1], 0.5)
    #getinfo(True)
    
    x = 0
    y = 0
    z = 0

    while x < totier:
        
        click(572, 402, 0.1)
        click(tierstartcoord[0], tierstartcoord[1], 0.1)
        tierstartcoord[1] += 27
        y = 0
        enchstartcoord = [665, 431]
        while y < 4:
            
            click(725, 401, 0.1)
            click(enchstartcoord[0], enchstartcoord[1], 0.1)
            enchstartcoord[1] += 27
            z = 0
            goodstartcoord = [814, 429]
            while z < 5:
                
                click(878, 402, 0.1)
                click(goodstartcoord[0], goodstartcoord[1], 0.3),
                buy1, buy2, sell1, sell2 = getinfo(False)
                goodstartcoord[1] += 27
                print(f"TIER: {x+totier-1} | ENCH: {y} | GOODNESS: {goodness[z]} -> BUY: {buy1}, AM: {buy2} | SELL: {sell1}, AM: {sell2}")

                z += 1
            y += 1
        x += 1

def exitfunc():
    global enabled
    
    if keyboard.is_pressed('x'):
        enabled = False
        os._exit(0)

def getinfo(pflag):
    buy1 = normalizenumber(getnumber([360, 1260, 70, 30]))
    buy2 = normalizenumber(getnumber([360, 1345, 70, 30]))
    sell1 = normalizenumber(getnumber([360, 1015, 70, 30]))
    sell2 = normalizenumber(getnumber([360, 1075, 70, 30]))
    if pflag:
        print(f"BUY: {buy1}, {buy2}")
        print(f"SELL: {sell1}, {sell2}")
    return [buy1, buy2, sell1, sell2]

def click(x, y, sleep=0):
    win32api.SetCursorPos((x, y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)
    time.sleep(sleep)

def programok():
    return

def normalizenumber(text):
    istenmeyen = [' ', '\n', '\x0c', ',', '.']
    rtrtext = ""
    for char in text:
        if not char in istenmeyen:
            rtrtext += char
    
    return rtrtext

def sim_score(min_val, max_val, method):
    if (method == cv2.TM_SQDIFF_NORMED):
        if(1 - min_val) <= 1:
            return(1-min_val)
        else:
            return(0)

    if (method == cv2.TM_SQDIFF):
        if(min_val <= 1):
            return(min_val)
        else:
            return(0)

    else:
        if max_val <= 1:
            return(max_val)
        else:
            return(0)

def getnumber(coordl):
    with mss.mss() as sct:
        x, y, w, h = coordl
        method = eval('cv2.TM_CCOEFF_NORMED')
        monitor = {"top": x, "left": y, "width": w, "height": h}
        ss = sct.grab(monitor)
        imageor = cv2.cvtColor(np.array(ss), cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(np.array(ss), cv2.COLOR_BGR2GRAY)
        #img = Image.frombytes("RGB", ss.size, ss.bgra, "raw", "BGRX")
        image = cv2.resize(image ,None, fx = 10, fy = 10, interpolation = cv2.INTER_CUBIC)
        image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        #cv2.imshow('test', image)
        custom_config = r'--oem 1 --psm 10 outputbase digits -c tessedit_char_whitelist=0123456789'
        cc = r'--oem 1 -c tessedit_char_whitelist=0123456789'
        text = pytesseract.image_to_string(image, config=cc)
        text = normalizenumber(text)
        #print(text)
        if text == '':
            
            rtrvalue = pytesseract.image_to_string(image, config=custom_config)
            if normalizenumber(rtrvalue) == '':
                rtrvalue = '1'
            return rtrvalue
        return text


def getstring(coordl):
    with mss.mss() as sct:
        x, y, w, h = coordl
        monitor = {"top": x, "left": y, "width": w, "height": h}
        ss = sct.grab(monitor)
        '''
        imgsell = Image.frombytes("RGB", ss.size, ss.bgra, "raw", "BGRX")
        imgsell.show()
        '''

        imageor = cv2.cvtColor(np.array(ss), cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(np.array(ss), cv2.COLOR_BGR2GRAY)
        #img = Image.frombytes("RGB", ss.size, ss.bgra, "raw", "BGRX")
        image = cv2.resize(image ,None, fx = 10, fy = 10, interpolation = cv2.INTER_CUBIC)
        image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        #cv2.imshow('test', image)
        #cc = r'--oem 0 -c tessedit_char_whitelist=0123456789'
        text = pytesseract.image_to_string(imageor)
        return text


class RepeatedTimer:

    """Repeat `function` every `interval` seconds."""
    stopped = False
    def __init__(self, interval, function, *args, **kwargs):
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.start = time.time()
        self.event = Event()
        self.thread = Thread(target=self._target)
        self.thread.start()

    def _target(self):
        if self.interval == -1:
            self.function(*self.args, **self.kwargs)
        else:
            while not self.event.wait(self._time) and not self.stopped:
                self.function(*self.args, **self.kwargs)

    @property
    def _time(self):
        return self.interval - ((time.time() - self.start) % self.interval)

    def stop(self):
        self.event.set()
        self.thread.join()

    def cum(self):
        self.stopped = True


if __name__ == '__main__':
    main()