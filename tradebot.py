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
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

#TODO: CLEAN THE CODE UP
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

global net
net = Net()
PATH = './weights/digits.pth'
net.load_state_dict(torch.load(PATH))

def main():
    print("basladi")
    global enabled
    global sscount
    sscount = 0
    
    ss_count_init()
    enabled = True
    exitloop = RepeatedTimer(0.1, exitfunc)
    tmp0 = cv2.imread('images/ss1.png', 0) #read the button image
    funclist = []
    #funclist.append("get_price_info(x, y, z, goodness, int(lowtier))")
    funclist.append("save_price_images()")
    while enabled:
        isButton = clickbutton(tmp0)

        if isButton:
            start_time = time.time()
            run_item_loop(funclist)
            print("ZAMAN: " + str(time.time()-start_time))
        
        time.sleep(0.5)
    
def get_price_info(x, y, z, goodness, totier):
    buy1, buy2, sell1, sell2 = getinfo()
    print(f"TIER: {x+totier} | ENCH: {3-y} | GOODNESS: {goodness[z]} -> BUY: {buy1}, AM: {buy2} | SELL: {sell1}, AM: {sell2}")

def get_ss(x, y, w, h):
    monitor = {"top": y, "left": x, "width": w, "height": h}
    with mss.mss() as sct:
        ss = sct.grab(monitor)
    return ss
    
def ss_count_init():
    global sscount
    files = os.listdir('ss/')
    if files:
        for z in range(0, len(files)):
            files[z] = int(files[z].strip('.png'))

        sscount = max(files) + 1

    else:
        sscount = 0

def save_price_images():
    global sscount
    sell = get_ss(1263, 366, 1446-1263, 524-366)
    sell = np.array(sell)
    sell = cv2.resize(sell ,None, fx = 5, fy = 5, interpolation = cv2.INTER_CUBIC)
    cv2.imwrite("ss/" + str(sscount) + ".png", sell)
    #mss.tools.to_png(sell.rgb, sell.size, output = "ss/" + str(sscount) + ".png")
    sscount += 1
    buy = get_ss(1021, 366, 1183-1021, 525-366)
    buy = np.array(buy)
    buy = cv2.resize(buy ,None, fx = 5, fy = 5, interpolation = cv2.INTER_CUBIC)
    cv2.imwrite("ss/" + str(sscount) + ".png", buy)
    #mss.tools.to_png(buy.rgb, buy.size, output = "ss/" + str(sscount) + ".png")
    sscount += 1
    
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
    
def run_item_loop(funclist):
    click(572, 402, 0.2) #tier menüsünü aç
    lowtier = normalizenumber(getstring([416, 458, 58, 21]).split(' ')[1])
    print(f"Tier: {lowtier}")
    totier = 8-int(lowtier)+1
    tierstartcoord = [536, 428, 58, 21]
    enchstartcoord = [665, 431 + 3 * 27]
    goodstartcoord = [814, 429]
    goodness = ["Normal", "Good", "Outstanding", "Excellent", "Masterpiece"]
    #click(tierstartcoord[0], tierstartcoord[1], 0.5)
    #getinfo(True)
    nonench = 4-int(lowtier)

    x = 0
    y = 0
    z = 0

    while x < totier:
        click(572, 402, 0.1)
        click(tierstartcoord[0], tierstartcoord[1], 0.1)
        tierstartcoord[1] += 27
        y = 0
        z = 0
        nonench -= 1
        enchstartcoord = [665, 431 + 3 * 27]
        while y < 4:
            if nonench < 0:
                click(725, 401, 0.1)
                click(enchstartcoord[0], enchstartcoord[1], 0.1)
                enchstartcoord[1] -= 27
                z = 0

            goodstartcoord = [814, 429]
            while z < 5:
                
                click(878, 402, 0.1)
                click(goodstartcoord[0], goodstartcoord[1], 0.3),
                goodstartcoord[1] += 27
                #time.sleep(0.3)
                
                if funclist:
                    for func in funclist:
                        eval(func)

                z += 1
            y += 1
        x += 1

def exitfunc():
    global enabled
    
    if keyboard.is_pressed('x'):
        enabled = False
        os._exit(0)

def getinfo():
    buy1 = normalizenumber(getnumbernn([360, 1260, 70, 30]))
    buy2 = normalizenumber(getnumbernn([360, 1345, 70, 30]))
    sell1 = normalizenumber(getnumbernn([360, 1015, 70, 30]))
    sell2 = normalizenumber(getnumbernn([360, 1075, 70, 30]))
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

def getnumbernn(coordl):
    global net
    with mss.mss() as sct:
        x, y, w, h = coordl
        method = eval('cv2.TM_CCOEFF_NORMED')
        monitor = {"top": x, "left": y, "width": w, "height": h}
        ss = numpy.array(sct.grab(monitor))
        img = cv2.resize(ss ,None, fx = 5, fy = 5, interpolation = cv2.INTER_CUBIC)
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
        predicted_str = ""
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
                #cv2.imsave()
                if roi.any():
                    roi = cv2.resize(roi, (28, 28), interpolation = cv2.INTER_AREA)
                    transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
                    roitensor = transformation(roi).float()
                    roitensor = roitensor.unsqueeze_(0)
                    #NN
                    with torch.no_grad():
                        output = net(roitensor)
                        _, predicted = torch.max(output.data, 1)
                        predicted_str += str(predicted.item())
                else:
                    print("FUCK")
        return predicted_str

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