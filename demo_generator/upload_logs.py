#!/usr/bin/python3 
import glob
import os
import time
import uuid
import urllib.request
import json
from pathlib import Path

cur_path = os.path.dirname(os.path.realpath(__file__))

if "RUNNING_IN_EC2" in os.environ:
    url = "http://169.254.169.254/latest/meta-data/iam/security-credentials/" + os.environ["ROLE"]
else:
    url = "http://169.254.170.2" + os.environ["AWS_CONTAINER_CREDENTIALS_RELATIVE_URI"]

machine_id = uuid.uuid1()
print(f"machine uuid: {machine_id}")
print("starting while loop of uploading")
num_sleeps = 0
while True:
    if num_sleeps == 0:
        req = urllib.request.Request(url)
        response = urllib.request.urlopen(req)
        data = response.read()
        values = json.loads(data)
        os.environ["AWS_ACCESS_KEY_ID"] = values["AccessKeyId"]
        os.environ["AWS_SECRET_ACCESS_KEY"] = values["SecretAccessKey"]
        os.environ["AWS_SESSION_TOKEN"] = values["Token"]
    files = glob.glob(cur_path + '/csgo-dedicated-non-volumne/csgo/*.dem') # * means all if need specific format then *.csv
    files.sort(key=os.path.getmtime)
    print(f"found {len(files)} files, need to upload {len(files[:-1])} of them")
    # leave the most recently touched demo, as cs is still writing to it
    for f in files[:-1]:
        print("moving file:" + str(f))
        p = Path(f)
        aws_name = p.stem + str(machine_id) + p.suffix 
        os.system(f"aws s3 cp {f} s3://csknow/demos/train_data/unprocessed/{aws_name}")
        os.remove(f)
    time.sleep(60)
    num_sleeps += 1
    if num_sleeps == 10:
        num_sleeps = 0
print("done while loop of uploading")
