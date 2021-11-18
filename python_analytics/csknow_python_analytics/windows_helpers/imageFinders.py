import pyautogui
import pydirectinput
import cv2
from tesserocr import PyTessBaseAPI
from PIL import Image
import re
import numpy as np


def getRegionFromImage(image_path, name):
    try:
        location = pyautogui.locateOnScreen(str(image_path), confidence=0.9)
    except pyautogui.ImageNotFoundException:
        print(f'''couldn't find {name}''')
        quit(1)
    if location is None:
        print(f'''couldn't find {name}''')
        quit(1)
    return location


def moveToRegion(location):
    pydirectinput.moveTo(int(location.left + location.width / 3), int(location.top + location.height / 2))


def getTick(death_region, tick_width, tick_height, tessocr_api):
    # get the tick number
    (death_left, death_top, death_width, death_height) = death_region
    tick_shot_pil = pyautogui.screenshot(region=(death_left + death_width, death_top, tick_width, tick_height))
    tick_shot_cv = np.array(tick_shot_pil)

    tick_shot_cv_gray = cv2.cvtColor(tick_shot_cv, cv2.COLOR_RGB2GRAY)
    (_, tick_shot_cv_black) = cv2.threshold(tick_shot_cv_gray, 190, 255, cv2.THRESH_BINARY_INV)

    pil_cap_black_text_demoUI = Image.fromarray(tick_shot_cv_black)
    tessocr_api.SetImage(pil_cap_black_text_demoUI)
    text = tessocr_api.GetUTF8Text()

    cur_tick_string_match = re.search('k[^\d]*(\d+)[^\d]*\/[^\d]*(\d+)', text)
    if cur_tick_string_match:
        return int(cur_tick_string_match.group(1))
    else:
        print("skipping as didn't find tick")
        return None
