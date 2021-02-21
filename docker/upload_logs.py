#!/usr/bin/python3 
import glob
import os
import time
import uuid
from pathlib import Path

cur_path = os.path.dirname(os.path.realpath(__file__))

machine_id = uuid.uuid1()
print(f"machine uuid: {machine_id}")
print("starting while loop of uploading")
while True:
    files = glob.glob(cur_path + '/csgo-dedicated-non-volumne/csgo/*.dem') # * means all if need specific format then *.csv
    files.sort(key=os.path.getmtime)
    print(f"found {len(files)} files, need to upload {len(files[:-1])} of them")
    # leave the most recently touched demo, as cs is still writing to it
    for f in files[:-1]:
        print("moving file:" + str(f))
        p = Path(f)
        aws_name = p.stem + str(machine_id) + p.suffix 
        os.system(f"aws s3 cp {f} s3://csknow/demos/unprocessed/{aws_name}")
        os.remove(f)
    time.sleep(60)
print("done while loop of uploading")
