import csknow_python_analytics.windows_helpers.windowManager as w
from csknow_python_analytics.windows_helpers.imageFinders import *
import pyautogui
import pydirectinput
import time
import argparse
import re
import os
import pathlib
import subprocess
import pandas as pd
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("state_images", help="folder with image to look for understanding demo state",
                    type=str)
parser.add_argument("video_folder", help="file with image to look for when demo is done",
                    type=str)
parser.add_argument("config", help="csgo config for player",
                    type=str)
parser.add_argument("snippets", help="csv of start and end snippets to record",
                    type=str)
args = parser.parse_args()

# get windows that will exist for entire video
hlae_wmgr = w.WindowMgr()
hlae_wmgr.find_window_wildcard("Half-Life Advanced.*")
if not hlae_wmgr.found_window():
    subprocess.call(['C:\\Users\\Administrator\\Documents\\hlae_2_123_0\\HLAE.exe'])
    time.sleep(0.1)
    hlae_wmgr.find_window_wildcard("Half-Life Advanced.*")
if not hlae_wmgr.found_window():
    print("couldn't find HLAE")
    quit(1)

obs_wmgr = w.WindowMgr()
obs_wmgr.find_window_wildcard("OBS.*")
if not obs_wmgr.found_window():
    subprocess.call(['C:\\Program Files\\obs-studio\\bin\\64bit\\obs64.exe'])
    time.sleep(0.1)
    hlae_wmgr.find_window_wildcard("OBS.*")
if not obs_wmgr.found_window():
    print("couldn't find OBS")
    quit(1)

print(f'''processing {args.config}''')
if match_config := re.search('(.*)_pre_load_(.*)_(\d+)\.cfg', args.config):
    match_prefix = match_config.group(1)
    player_name = match_config.group(2)
    team_number = match_config.group(3)
else:
    print(f'''bad config {args.config}''')
    exit(1)
# start game
hlae_wmgr.set_foreground()
time.sleep(1)
with pyautogui.hold('alt'):
    pyautogui.press('f')
pyautogui.press('enter')
pyautogui.press('enter')

time.sleep(25)

csgo_wmgr = w.WindowMgr()
csgo_wmgr.find_window_wildcard("Counter.*")
if not csgo_wmgr.found_window():
    print("couldn't find CSGO")
    quit(1)
csgo_wmgr.set_foreground()

time.sleep(1)

# load demo
state_images_path = pathlib.Path(args.state_images)
console_text_entry_path = state_images_path / 'console_text_entry.png'
console_text_entry_region = getRegionFromImage(console_text_entry_path, "console text entry")
moveToRegion(console_text_entry_region)
pyautogui.click()
pyautogui.write(f'''exec {args.config}\n''')

time.sleep(40)

# for some reason shift gets pressed, this hack disables it
pyautogui.press('shift')

pydirectinput.press('`')
pyautogui.write(f'''exec {match_prefix}_post_load_{team_number}\n''')
pyautogui.write(f'''mirv_streams previewEnd\n''')

tessocr_api = PyTessBaseAPI()
snippets_df = pd.read_csv(args.snippets)
misread_ticks = 0
time.sleep(1)
pydirectinput.press('`')
for index, row in snippets_df.iterrows():
    # run post load configs
    time.sleep(1)
    pydirectinput.press('`')
    pyautogui.write(f'''r_drawothermodels 2\n''')
    pyautogui.write(f'''demo_goto {row["start_game_tick"]}\n''')

    time.sleep(3)

    # get death and tick locations
    death_region = getRegionFromImage(state_images_path / 'just_death.png', "death selector")
    tick_image = Image.open(state_images_path / 'tick_no_death.png')
    tick_width, tick_height = tick_image.size

    # get resume button location
    resume_path = state_images_path / 'just_resume.png'
    resume_region = getRegionFromImage(resume_path, "just resume button")

    # start playback and recording
    pydirectinput.press('`')
    moveToRegion(resume_region)
    pydirectinput.click()
    pydirectinput.moveTo(30, 30)
    pydirectinput.press('F1')

    # wait until end of snippet
    while True:
        tick = getTick(death_region, tick_width, tick_height, tessocr_api)
        # skip invalid or misread ticks
        if tick is None or tick > 5 * row['end_game_tick']:
            misread_ticks += 1
            if misread_ticks >= 30:
                quit(1)
            continue
        if tick > row['end_game_tick']:
            misread_ticks = 0
            break
        time.sleep(2)


    # stop now that done with snippet
    pydirectinput.press('F2')
    pydirectinput.press('`')
    moveToRegion(resume_region)
    pydirectinput.click()

    time.sleep(1)

    # get latest video file and name it appropriately
    files = os.listdir(args.video_folder)
    paths = [os.path.join(args.video_folder, basename) for basename in files]
    newest_video_path = max(paths, key=os.path.getmtime)
    os.rename(newest_video_path, os.path.join(args.video_folder,
                                              f'''{match_prefix}_{player_name}_{team_number}_{row["start_game_tick"]}_{row["end_game_tick"]}.mp4'''))

    time.sleep(2)

csgo_wmgr.set_foreground()
pydirectinput.press('`')
pyautogui.write('quit\n')
