import cv2
import torch
from PIL import Image
import numpy as np
import mss
import mss.tools
from PIL import Image
import sys
import cv2
import keyboard
import win32api, win32com
import win32con, win32clipboard
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
import math
import sqlite3
from datetime import datetime

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

LOAD_NET=True
LOAD_YOLOV=False
global net
global output
global database
database = ""
global itemname
itemname = ""
output = open("output", "a+")

if LOAD_NET:
    device = torch.device('cpu')
    net = Net()
    PATH = './weights/digits.pth'
    net.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))

if LOAD_YOLOV:
    device = torch.device("cuda")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', classes=4)  # for file/URI/PIL/cv2/np inputs and NMS
    model.load_state_dict(torch.load('weights/best.pt')['model'].state_dict())
    model = model.fuse().autoshape()
    model.to(device)

def main():
    init()
    print("basladi")
    global enabled
    global sscount
    global itemname
    global database
    sscount = 0
    
    ss_count_init()

    enabled = True
    exitloop = RepeatedTimer(0.1, exitfunc)
    tmp0 = cv2.imread('images/ss1.png', 0) #read the button image
    funclist = []

    funclist.append("get_price_info(x, y, z, goodness, int(lowtier))")
    #funclist.append("save_price_images()")
    coordlist = [[401, 661, 284, 54], [492, 661, 284, 54], [584, 661, 284, 54], [675, 661, 284, 54], [761, 661, 284, 54], [851, 661, 284, 54]]
    dict_file = open("item_names", "r")
    item_dict = dict_file.read().split('\n')
    dict_file.close()
    #item_dict = ["Novice's Mercenary Shoes", "Novice's Soldier Helmet", "Adept's Dagger Pair"]
    button_coord = [int(406//1.406), int(1224//1.406), int(110//1.406), int(38//1.406)]

    firstseen = False
    enabled = True
    enabled2 = False
    enabled3 = False
    time.sleep(3)
    while enabled:
        name = ""
        is_clicked = False

        for x in range(0, len(item_dict)):
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            test = item_dict[x]
            win32clipboard.SetClipboardText(test, win32con.CF_TEXT)
            win32clipboard.CloseClipboard()
            #Point(x=609, y=265)
            win32api.SetCursorPos((int(609//1.406), int(265//1.406)))
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,int(609//1.406),int(265//1.406),0,0)
            time.sleep(0.1)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,int(609//1.406),int(265//1.406),0,0)
            time.sleep(0.1)
            keyboard.press_and_release('ctrl+a')
            keyboard.press_and_release('backspace')
            keyboard.press_and_release('ctrl+v')
            time.sleep(0.1)
            #name = normalizestring(getstring(coordlist[0]))
            print("name:" + item_dict[x] + "|")
            output.write("| " + item_dict[x] + "| : \n")
            itemname = item_dict[x]
            if True: #if not name == ''
                is_clicked = clickbutton(tmp0, button_coord)
                if is_clicked:
                    start_time = time.time()
                    run_info_loop(funclist)
                    click(937, 312, 0.3) #close the menu
                    print("ZAMAN: " + str(time.time()-start_time))
                else:
                    print("Item yok, atlanıyor.") #TODO: handle non existant items better by skipping over them.

        print("BİTTİ!")
        output.close()
        enabled = False
        database.close()

    
    #time.sleep(1)
    while enabled3:
        sct_img = get_ss(0, 0, 1920, 1080)
        result = model(sct_img, size=800)
        if result.xyxy:
            for match in result.xyxy[0]:
                if (match[5] == 1.0 and match[4] > 0.6):
                    xcenter, ycenter = match[0] + (match[2] - match[0]) / 2, match[1] + (match[3] - match[1]) / 2
                    click(int(xcenter), int(ycenter), 0.9)
                    enabled3 = False


    
    while enabled2:
        img = get_ss_numpy(1562, 806, 357, 273)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = rotate_image(img, 45)
        xf, yf = findflag(img)
        xm, ym = findme(img)
        
        #flength = math.sqrt(xf*xf + yf*yf)  
        #mlength = math.sqrt(xm*xm + ym*ym)
        #yf = -xf*math.sin(45) + yf*math.cos(45)
        #ym = -xm*math.sin(45) + ym*math.cos(45)
        print(f"flag: {xf}, {yf}\n me: {xm}, {ym}")
        move = True
        if move:

            if xm - xf > 20: #moveleft
                #right_click(693, 263, 1)
                right_click(837, 358, 0.2)
                print("LEFT")


            if xf - xm > 20: #moveright
                #right_click(1432, 754, 1)
                right_click(1075, 548, 0.2)
            
            
            if ym - yf > 4: #moveup
                #right_click(1257, 256, 1)
                right_click(1086, 397, 0.2)
                print("UP")
                

            if yf - ym > 4: #movedown
                #right_click(605, 661, 1)
                right_click(832, 540, 0.2)
                print("DOWn")

            
        time.sleep(0.7)
    

def init():
    global database
    if not os.path.exists("db"):
        os.makedirs("db")
        print("db klasörü oluşturuldu!")
    database = sqlite3.connect("db/item.db")
    cursor = database.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS items(id INTEGER PRIMARY KEY, date TEXT, name TEXT, tier TEXT, ench TEXT, goodness TEXT, buyqty TEXT, buyprice TEXT, sellqty TEXT, sellprice TEXT)
                            ''')

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def findme(img):
    frame = img
    #cv2.imshow("test", img)
    #cv2.waitKey(0)
    lower_blue = np.array([255,160,80])
    upper_blue = np.array([255,200,110])

    mask = cv2.inRange(img,lower_blue,upper_blue)
    #cv2.imshow("mask", mask)
    xm, ym = 0, 0
    cnts, hier = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    sorted_ctrs = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0])
    for i, ctr in enumerate(sorted_ctrs):
        xm, ym, w, h = cv2.boundingRect(ctr)
        cv2.rectangle(frame,(xm,ym),( xm + w, ym + h ),(90,0,255),2)
        #cv2.imshow("ctr", frame)
        xm, ym = xm + w/2, ym + h/2
    #cv2.imshow("me", frame)
    #cv2.waitKey(0)
    return xm, ym

def findflag(img):
    xm, ym = 0, 0
    framegray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tmp0 = cv2.imread("flag3.png", 0)
    w, h = tmp0.shape[::-1]
    method = eval('cv2.TM_SQDIFF_NORMED')
    res0 = cv2.matchTemplate(framegray, tmp0, method)
    min_val0, max_val0, min_loc0, max_loc0 = cv2.minMaxLoc(res0)
    xf, yf = min_loc0[0], min_loc0[1]
    if (1 - min_val0) < 1:
        if (1 - min_val0) > 0.8:
            cv2.rectangle(img,(xf,yf),( xf + w, yf + h ),(90,0,255),2)
            xf = xf + w/2
            yf = yf + h/2

    #cv2.imshow("flag", img)
    #cv2.waitKey(0)
    return xf, yf
    
def get_price_info(x, y, z, goodness, totier):
    global itemname
    global database
    cursor = database.cursor()
    buy1, buy2, sell1, sell2 = getinfo()
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%dT%H:%M")
    cursor.execute(f"INSERT INTO items(date, name, tier, ench, goodness, buyqty, buyprice, sellqty, sellprice) VALUES (?,?,?,?,?,?,?,?,?)", (str(date_string), str(itemname), str(x+totier), str(3-y), str(goodness[z]), str(buy2), str(buy1), str(sell2), str(sell1)))
    database.commit()

    if buy1 == "6009091160905401011":
        buy1 = "NOT"

    if buy2 == "959146124900112666":
        buy2 = "BUYING"

    if sell1 == "0260905401019296696":
        sell1 = "NOT"

    if sell2 == "9591461254601002663":
        sell2 = "SELLING"

    info = f"TIER: {x+totier} | ENCH: {3-y} | GOODNESS: {goodness[z]} -> BUY: {buy1}, AM: {buy2} | SELL: {sell1}, AM: {sell2}"

    print(info)
    output.writelines(info + "\n")

def get_ss(x, y, w, h):
    monitor = {"top": int(y/1.406), "left": int(x/1.406), "width": int(w/1.406), "height": int(h/1.406)}
    with mss.mss() as sct:
        ss = sct.grab(monitor)
    return ss

def get_ss_numpy(x, y, w, h):
    monitor = {"top": int(y/1.406), "left": int(x/1.406), "width": int(w/1.406), "height": int(h/1.406)}
    with mss.mss() as sct:
        ss = sct.grab(monitor)
        ss = np.array(ss)
        #ss = cv2.resize(ss ,None, fx = 5, fy = 5, interpolation = cv2.INTER_CUBIC)
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
    
def clickbutton(tmp0, coord):
    w, h = tmp0.shape[::-1]
    method = eval('cv2.TM_SQDIFF_NORMED')

    if enabled is True:
        start_timex = time.time()
        with mss.mss() as sct:
            monitor = {"top": coord[0], "left":coord[1] , "width": coord[2], "height": coord[3]}
            img = cv2.cvtColor(np.array(sct.grab(monitor)), cv2.COLOR_BGR2GRAY)
            res0 = cv2.matchTemplate(img, tmp0, method)
        
        min_val0, max_val0, min_loc0, max_loc0 = cv2.minMaxLoc(res0)
        sim0 = sim_score(min_val0, max_val0, method)
        
        print(sim0)
        if sim0 > 0.70:
            x = int(coord[1] + int(min_loc0[0]) + int(w/1.406/2))
            y = int(coord[0] + int(min_loc0[1]) + int(h/1.406/2))
            click(int(x*1.406), int(y*1.406), 0.6)
            print(sim0)
            return True
            
        else:
            return False

def run_list_loop():
    return

def run_info_loop(funclist):
    click(572, 402, 0.2) #tier menüsünü aç
    #lowtier = normalizenumber(getstring([416, 458, 58, 21]).split(' ')[1])
    lowtier = getnumbernn([421, 495, 14, 14]) #read lowest tier with NN instead.
    #"top": 421, "left": 497, "width": 14, "height": 14
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
    win32api.SetCursorPos((int(x//1.406), int(y//1.406)))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,int(x//1.406),int(y//1.406),0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,int(x//1.406),int(y//1.406),0,0)
    time.sleep(sleep)

def right_click(x, y, sleep=0):
    win32api.SetCursorPos((x//1.406, y//1.406))
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN,int(x//1.406),int(y//1.406),0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP,int(x//1.406),int(y//1.406),0,0)
    time.sleep(sleep)

def wheel_event(move):
    win32api.mouse_event(MOUSEEVENTF_WHEEL, x, y, move, 0)

def programok():
    return

def normalizenumber(text):
    istenmeyen = [' ', '\n', '\x0c', ',', '.']
    rtrtext = ""
    for char in text:
        if not char in istenmeyen:
            rtrtext += char
    
    return rtrtext

def normalizestring(text):
    istenmeyen = ['\n', '\x0c', ',', '.']
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
        monitor = {"top": int(x//1.406), "left": int(y//1.406), "width": int(w//1.406), "height": int(h//1.406)}
        ss = np.array(sct.grab(monitor))
        img = cv2.resize(ss ,None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        #binarize 
        ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
        cv2.waitKey(0)

        #find contours
        ctrs, hier = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #sort contours
        sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
        predicted_str = ""
        for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
            x, y, w, h = cv2.boundingRect(ctr)
            if h > 10:
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
                    cv2.rectangle(img,(x,y),( x + w, y + h ),(90,0,255),2)

                else:
                    print("FUCK")

        cv2.imshow('marked areas',gray)
        cv2.waitKey(0)
        return predicted_str

def getnumber(coordl):
    with mss.mss() as sct:
        x, y, w, h = coordl
        method = eval('cv2.TM_CCOEFF_NORMED')
        monitor = {"top": int(x//1.406), "left": int(y//1.406), "width": int(w//1.406), "height": int(h//1.406)}
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
        monitor = {"top": int(x//1.406), "left": int(y//1.406), "width": int(w//1.406), "height": int(h//1.406)}
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