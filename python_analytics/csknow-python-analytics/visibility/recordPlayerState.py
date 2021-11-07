import pyautogui
import pydirectinput
import keyboard
import argparse
import re
import os
from PIL import Image
import numpy as np
import cv2
from tesserocr import PyTessBaseAPI
import pandas as pd
import time

parser = argparse.ArgumentParser()
parser.add_argument("death_image", help="file with image to look for next death control",
                    type=str)
parser.add_argument("tick_image", help="file with image of example current game tick",
                    type=str)
parser.add_argument("output_dir", help="directory for data output",
                    type=str)
parser.add_argument("config", help="config for loading demo from player's perspective",
                    type=str)
args = parser.parse_args()

action_strings = ['holding', 'pushing', 'combat', 'rotating', 'utility']
actions = []

def addActionDict(action_number, game_tick):
    actions.append({'action_name': action_strings[action_number], 'action_number': action_number, 'game_tick_number': game_tick})
    print(f'''adding {actions[len(actions)-1]}''')

instruction_str = "Press "
for i in range(len(action_strings)):
    instruction_str += f'''{str(i+1)} for {action_strings[i]}, '''
instruction_str += "z to quit, o to orient (must orient each time demo ui is moved)"

#
if match_config := re.search('(.*)_pre_load_(.*)_(\d+)\.cfg', args.config):
    match_prefix = match_config.group(1)
    player_name = match_config.group(2)
    team_number = match_config.group(3)
else:
    print(f'''bad config {args.config}''')
    exit(1)

# start game
pyautogui.moveTo(164, 1185)
pyautogui.click()
time.sleep(2)
with pyautogui.hold('alt'):
    pyautogui.press('f')
pyautogui.press('enter')
pyautogui.press('enter')

time.sleep(25)

# load demo
pyautogui.moveTo(950, 763)
pyautogui.click()
pyautogui.write(f'''exec {args.config}\n''')

time.sleep(40)

# run post load configs
pydirectinput.press('`')
pyautogui.write(f'''exec {match_prefix}_post_load_{team_number}\n''')

time.sleep(3)

pyautogui.write(f'''mirv_streams previewEnd\n''')
time.sleep(3)

pydirectinput.press('`')

# process data
tick_image = Image.open(args.tick_image)
tick_width, tick_height = tick_image.size
tessocr_api = PyTessBaseAPI()
first_time = False
while True:
    print(instruction_str)
    key = keyboard.read_key()
    time.sleep(0.4)
    # stop everything if z, skip if invalid key
    if key == 'z':
        break
    elif key != 'o' and key not in [str(i) for  i  in range(1,len(action_strings)+1)]:
        print(f'''{key} not in valid range 1 to {len(action_strings)+1}''')
        continue


    # get the death image location, and continue if just orienting
    if first_time or key == 'o':
        try:
            death_location = pyautogui.locateOnScreen(args.death_image)
            found_end = True
        except pyautogui.ImageNotFoundException:
            found_end = False
        if death_location is None:
            found_end = False
        if not found_end:
            print("skipping as didn't find death controls")
            continue
    if key == 'o':
        continue

    action_number = int(key)-1

    # get the tick number
    (death_left, death_top, death_width, death_height) = death_location
    tick_shot_pil = pyautogui.screenshot(region=(death_left + death_width, death_top, tick_width, tick_height))
    tick_shot_cv = np.array(tick_shot_pil)

    tick_shot_cv_gray = cv2.cvtColor(tick_shot_cv, cv2.COLOR_RGB2GRAY)
    (_, tick_shot_cv_black) = cv2.threshold(tick_shot_cv_gray, 190, 255, cv2.THRESH_BINARY_INV)

    pil_cap_black_text_demoUI = Image.fromarray(tick_shot_cv_black)
    tessocr_api.SetImage(pil_cap_black_text_demoUI)
    text = tessocr_api.GetUTF8Text()

    cur_tick_string_match = re.search('k[^\d]*(\d+)[^\d]*\/[^\d]*(\d+)', text)
    if cur_tick_string_match:
        cur_tick = int(cur_tick_string_match.group(1))
    else:
        print("skipping as didn't find tick")
        continue

    addActionDict(action_number, cur_tick)

df_actions = pd.DataFrame(actions)
df_actions.to_csv(os.join(args.output_dir, "actions_" + args.player_and_match + ".csv"), index=False)
