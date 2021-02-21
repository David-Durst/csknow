#!/usr/bin/python3 
import glob
import os
import time

cur_path = os.path.dirname(os.path.realpath(__file__))
print("starting while loop of uploading")
while True:
    files = glob.glob(cur_path + '/csgo-dedicated-non-volumne/csgo/*.dem') # * means all if need specific format then *.csv
    files.sort(key=os.path.getmtime)
    # leave the most recently touched demo, as cs is still writing to it
    for f in files[:-1]:
        print("moving file:" + str(f))
        os.system(f"aws s3 cp {f} s3://csknow/demos/")
        os.remove(f)
    time.sleep(60)
print("done while loop of uploading")
