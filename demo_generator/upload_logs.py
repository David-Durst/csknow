#!/usr/bin/python3 
import glob
import os
import time
import uuid
import urllib.request
import json
from pathlib import Path

cur_path = os.path.dirname(os.path.realpath(__file__))

machine_id = uuid.uuid1()
print(f"machine uuid: {machine_id}")
print("starting while loop of uploading")
num_sleeps = 0
while True:
    files = glob.glob(os.environ['NONVOLUMESTEAMAPPDIR'] + '/csgo/*.dem') # * means all if need specific format then *.csv
    files.sort(key=os.path.getmtime)
    print(f"found {len(files)} files, need to upload {len(files[:-1])} of them")
    csknow_bot_style = os.environ["CSKNOW_BOT_STYLE"]
    csgo_bot_style = os.environ["CSGO_BOT_STYLE"]
    # leave the most recently touched demo, as cs is still writing to it
    for f in files[:-1]:
        print("moving file:" + str(f))
        p = Path(f)
        aws_name = p.stem + "_" + csknow_bot_style + "_" + csgo_bot_style + "_" + str(machine_id) + p.suffix 
        os.system(f"aws s3 cp {f} s3://csknow/demos/bot_retakes_data/unprocessed/bots/{aws_name}")
        os.remove(f)
    time.sleep(60)
    num_sleeps += 1
    if num_sleeps == 10:
        num_sleeps = 0
print("done while loop of uploading")
