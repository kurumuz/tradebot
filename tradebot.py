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

def getinfo(pflag):
    buy1 = normalizenumber(getnumber([360, 1260, 70, 30]))
    buy2 = normalizenumber(getnumber([360, 1360, 70, 30]))
    sell1 = normalizenumber(getnumber([360, 1015, 70, 30]))
    sell2 = normalizenumber(getnumber([360, 1090, 70, 30]))
    if pflag:
        print(f"BUY: {buy1}, {buy2}")
        print(f"SELL: {sell1}, {sell2}")

def click(x, y, sleep=0):
    win32api.SetCursorPos((x, y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)

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
        method = eval('cv2.TM_SQDIFF_NORMED')
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
            tmplist = []
            simlist = []
            for x in range(0, 10):
                tmplist.append(cv2.imread('images/' + str(x) + '.png', 0))

            for x in range(0, 10):
                match = cv2.matchTemplate(image, tmplist[x], method)
                min_val0, max_val0, min_loc0, max_loc0 = cv2.minMaxLoc(match)
                #print(max_val0)
                sim = sim_score(min_val0, max_val0, method)
                #print(sim)
                sim = int(str(sim)[2:9] + str(x))
                simlist.append(sim)
            
            print(simlist)
            maxsim = max(simlist)
            strmaxsim = str(maxsim)
            rtrvalue = strmaxsim[len(strmaxsim)-1]
            print("DÜZELTİLDİ: " + rtrvalue)
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

# Model
enabled = True
url = 'ss'
starttime = time.time()
countertime = 0
first = True
tmp0 = cv2.imread('images/ss1.png', 0)
tmp1 = cv2.imread('images/ss2.png', 0)
w, h = tmp0.shape[::-1]
method = eval('cv2.TM_SQDIFF_NORMED')
per = 1

while (1):
    #cv2.circle(img, (top_left[0] + int(w/2), top_left[1] + int(h/2)), 6, (0, 255, 255), -1)
    #cv2.imshow("Test", img)

    countertime += 1
    if (time.time() - starttime) > per:
        #print("fps: {}".format(countertime / (time.time() - starttime)))
        countertime = 0
        starttime = time.time()

    if keyboard.is_pressed('q'):
        enabled = not enabled

    if keyboard.is_pressed('x'):
        sys.exit(0)

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
            click(x, y)
            print("--- %s seconds ---" % (time.time() - start_timex))
            time.sleep(0.6)
            click(572, 402) #tier menüsünü aç
            time.sleep(0.2)
            lowtier = normalizenumber(getstring([416, 458, 58, 21]).split(' ')[1])
            print(f"Tier: {lowtier}")
            totier = 8-int(lowtier)
            startcoord = [536, 428, 58, 21]
            click(startcoord[0], startcoord[1])
            time.sleep(0.5)
            getinfo(True)
            '''
            x = 0
            startcoord
            while x < totier:
                startcoord[1] += 27
                click(572, 402)
                click(startcoord[0], startcoord[1])
                time.sleep(0.2)
                getinfo(True)
                x += 1
                '''

            '''
            buy1 = normalizenumber(getstring([360, 1260, 70, 30]))
            buy2 = normalizenumber(getstring([360, 1360, 70, 30]))
            sell1 = normalizenumber(getstring([360, 1015, 70, 30]))
            sell2 = normalizenumber(getstring([360, 1090, 70, 30]))
            
            print(f"BUY: {buy1}, {buy2}")
            print(f"SELL: {sell1}, {sell2}")
            '''

            print(sim0)
            #f = open('tradebot.py', 'a+')
            #f.write(f"\n#BUY: {buy1}, {buy2}\n#SELL: {sell1}, {sell2}")
            #f.close()
