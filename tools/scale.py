import os
import cv2

folder = os.listdir('ss')
'''
for imgpath in folder:
    img = cv2.imread('ss/' + imgpath)
    img = cv2.resize(img ,None, fx = 5, fy = 5, interpolation = cv2.INTER_CUBIC)
    cv2.imwrite('ss/' + imgpath, img)
'''
img = cv2.imread('ss/' + "520.png")
img = cv2.resize(img ,None, fx = 5, fy = 5, interpolation = cv2.INTER_CUBIC)
cv2.imwrite('ss/' + "520.png", img)