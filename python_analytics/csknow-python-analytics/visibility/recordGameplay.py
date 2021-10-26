import pyautogui
import pydirectinput
import time
import argparse
import re

if False:
    i = 0
    while True:
        print(f'''mouse position {i}: {pyautogui.position()}''')
        i += 1
        time.sleep(2)

def csv_list(string):
   return string.split(',')

parser = argparse.ArgumentParser()
parser.add_argument("configs", help="comma separated list of configs for recording",
                    type=csv_list)
parser.add_argument("end_image", help="file with image to look for when demo is done",
                    type=str)
args = parser.parse_args()

for config in args.configs:
    print(f'''processing {config}''')
    if match_config := re.search('(.*)_pre_load_(.*)_(\d+)\.cfg', config):
        match_prefix = match_config.group(1)
        team_number = match_config.group(3)
    else:
        print(f'''bad config {config}''')
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
    pyautogui.write(f'''exec {config}\n''')

    time.sleep(40)

    # run post load configs
    pydirectinput.press('`')
    pyautogui.write(f'''exec {match_prefix}_post_load_{team_number}\n''')

    time.sleep(0.5)

    # move demoui to top left corner
    pydirectinput.moveTo(620, 639)
    pydirectinput.mouseDown(button='left')
    pydirectinput.moveTo(405, 326)
    pydirectinput.mouseUp(button='left')

    # start playback and recording
    pydirectinput.moveTo(950, 763)
    pyautogui.click()
    pyautogui.write('mirv_streams previewEnd\n')
    time.sleep(0.5)
    pyautogui.write('demo_resume\n')
    pydirectinput.press('`')
    pydirectinput.press('F1')

    # wait until demo over
    found_end = False
    while not found_end:
        found_end = True
        try:
            pyautogui.locateOnScreen(args.end_image)
        except pyautogui.ImageNotFoundException:
            found_end = False
            time.sleep(2.5)


    pydirectinput.press('F2')
    pydirectinput.press('`')
    pyautogui.write('quit\n')
    time.sleep(5)
