#EXPERIMENT RESULTS: NO INVERSE DOESN'T MAKE IT FASTER, PYOCR DOESNT MAKE IT FASTER

from PIL import Image
import sys

import pyocr
import pyocr.builders
import pytesseract
import time


def normalizenumber(text):
    istenmeyen = ['\n', '\x0c', ',', '.']
    rtrtext = ""
    for char in text:
        if not char in istenmeyen:
            rtrtext += char
    
    return rtrtext

pyocren = False
tools = pyocr.get_available_tools()
tool = tools[0]
langs = tool.get_available_languages()
lang = langs[0]
#print(tool.get_name() + " " + lang)
builder = pyocr.builders.TextBuilder()
img = Image.open('ss/46.png')

if pyocren == False:
    #print("test")
    start_time = time.time()
    config=r""
    test = normalizenumber(pytesseract.image_to_string(img, config=config))
    print(time.time() - start_time)
    #print("pytesseract")
    print(test)


else:
    start_time = time.time()
    test = tool.image_to_string(img, lang=lang, builder=builder)
    print(time.time() - start_time)
    print("pyocr")
    print(test)
