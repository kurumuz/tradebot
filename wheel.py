import win32api, win32com
import win32con
import time
import keyboard

def pass_one_item(x, y, add):
    #scrollthing loc:
    #Point(x=1348, y=410)
    win32api.SetCursorPos((x, y))
    time.sleep(0.2)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
    time.sleep(0.1)
    for i in range(0, add):
        y-=1
        time.sleep(0.0000005)
        win32api.SetCursorPos((x, y))

    time.sleep(0.2)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)
    #time.sleep(0.3)

def click(x, y, sleep=0):
    win32api.SetCursorPos((x, y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)
    time.sleep(sleep)

def movesmooth(x, y, add):
    win32api.SetCursorPos((x, y))
    for i in range(0, add):
        y += 1
        win32api.SetCursorPos((x, y))
        time.sleep(0.0000005)
    
x, y = 1348, 410

x, y = 958, 596

art = int(20 - (16 / 4.54))

art = 28.5

art = 95
while True:
    #Point(x=965, y=589)
    try:  
        if keyboard.is_pressed('q'):
            #win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL, 0, 0, -1, 0)
            pass_one_item(x, y, art)
            #movesmooth(100, 0, 1000)

        if keyboard.is_pressed('x'):
            y = 410
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)
        time.sleep(0.1)

    except:
        break
    
