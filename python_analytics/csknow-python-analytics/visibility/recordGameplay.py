import pyautogui
import pydirectinput
import time
import argparse
import re
import os
import pathlib

if False:
    i = 0
    while True:
        print(f'''mouse position {i}: {pyautogui.position()}''')
        i += 1
        time.sleep(2)

def csv_list(string):
   return string.split(',')

parser = argparse.ArgumentParser()
parser.add_argument("end_image", help="file with image to look for when demo is done",
                    type=str)
parser.add_argument("video_folder", help="file with image to look for when demo is done",
                    type=str)
parser.add_argument("configs", help="comma separated list of configs for recording",
                    type=csv_list)
args = parser.parse_args()

num_hlae_icon_clicks = 0
for config in args.configs:
    print(f'''processing {config}''')
    if match_config := re.search('(.*)_pre_load_(.*)_(\d+)\.cfg', config):
        match_prefix = match_config.group(1)
        player_name = match_config.group(2)
        team_number = match_config.group(3)
    else:
        print(f'''bad config {config}''')
        exit(1)
    # start game
    pyautogui.moveTo(164, 1185)
    pyautogui.click()
    # do in loop to ensure hlae on top
    num_hlae_icon_clicks += 1
    while num_hlae_icon_clicks % 2 != 1:
        pyautogui.click()
        num_hlae_icon_clicks += 1
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

    time.sleep(3)

    # move demoui to top left corner
    end_image_path = pathlib.Path(args.end_image)
    demo_playback_name_path = end_image_path.parent / 'demo_playback_name.png'
    try:
        demo_playback_name_location = pyautogui.locateOnScreen(str(demo_playback_name_path))
    except pyautogui.ImageNotFoundException:
        print("couldn't find demo playback name")
        quit(1)
    if demo_playback_name_location is None:
        print("couldn't find demo playback name")
        quit(1)
    pydirectinput.moveTo(demo_playback_name_location.left + demo_playback_name_location.width,
                         demo_playback_name_location.top + demo_playback_name_location.height / 2)
    pydirectinput.mouseDown(button='left')
    time.sleep(1.0)
    pydirectinput.moveTo(-215, -324)
    pydirectinput.mouseUp(button='left')

    # start playback and recording
    console_text_entry_path = end_image_path.parent / 'console_text_entry.png'
    try:
        console_text_entry_location = pyautogui.locateOnScreen(str(console_text_entry_path))
    except pyautogui.ImageNotFoundException:
        print("couldn't find console text entry")
        quit(1)
    if demo_playback_name_location is None:
        print("couldn't find console text entry")
        quit(1)
    pydirectinput.moveTo(console_text_entry_location.left + console_text_entry_location.width / 3,
                         console_text_entry_location.top + console_text_entry_location.height / 2)
    pydirectinput.click()
    #pyautogui.write('mirv_streams previewEnd\n')
    #time.sleep(0.5)
    pyautogui.write('demo_resume\n')
    pydirectinput.moveTo(30, 30)
    pydirectinput.press('`')
    pydirectinput.press('F1')

    # wait until demo over
    found_end = False
    while not found_end:
        found_end = True
        pydirectinput.moveTo(35, 35)
        time.sleep(7)
        pydirectinput.moveTo(30, 30)
        try:
            location = pyautogui.locateOnScreen(args.end_image)
        except pyautogui.ImageNotFoundException:
            found_end = False
            time.sleep(2.5)
        if location is None:
            found_end = False

    pydirectinput.moveTo(console_text_entry_location.left + console_text_entry_location.width / 3,
                         console_text_entry_location.top + console_text_entry_location.height / 2)
    pydirectinput.click()
    pydirectinput.press('F2')
    pydirectinput.press('`')
    pyautogui.write('quit\n')

    time.sleep(1)

    # get latest video file and name it appropriately
    files = os.listdir(args.video_folder)
    paths = [os.path.join(args.video_folder, basename) for basename in files]
    newest_video_path = max(paths, key=os.path.getmtime)
    os.rename(newest_video_path, os.path.join(args.video_folder, f'''{match_prefix}_{player_name}_{team_number}.mp4'''))

    time.sleep(5)

# click off any windows at end so next script can start fressh
pydirectinput.moveTo(1600, 1170)
pydirectinput.click()
