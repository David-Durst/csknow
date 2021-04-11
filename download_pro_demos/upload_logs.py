#!/usr/bin/python3 
import glob
import os
import time
import uuid
import urllib.request
import json
import sys
from pathlib import Path

cur_path = os.path.dirname(os.path.realpath(__file__))

if "RUNNING_IN_EC2" in os.environ:
    url = "http://169.254.169.254/latest/meta-data/iam/security-credentials/" + os.environ["ROLE"]
else:
    url = "http://169.254.170.2" + os.environ["AWS_CONTAINER_CREDENTIALS_RELATIVE_URI"]

machine_id = uuid.uuid1()
print(f"machine uuid: {machine_id}")
req = urllib.request.Request(url)
response = urllib.request.urlopen(req)
data = response.read()
values = json.loads(data)
os.environ["AWS_ACCESS_KEY_ID"] = values["AccessKeyId"]
os.environ["AWS_SECRET_ACCESS_KEY"] = values["SecretAccessKey"]
os.environ["AWS_SESSION_TOKEN"] = values["Token"]

os.system("7z e temp_downloads/*.rar -otemp_downloads/ *dust2*.dem -y")

files = glob.glob(cur_path + '/temp_downloads/*.dem') # * means all if need specific format then *.csv
print(f"uploading {len(files)} files")
# leave the most recently touched demo, as cs is still writing to it
for f in files:
    print("moving file:" + str(f))
    p = Path(f)
    aws_name = sys.argv[1] + "_" + sys.argv[2] + "_" + p.stem + "_" + str(machine_id) + p.suffix 
    os.system(f"aws s3 cp {f} s3://csknow/demos/unprocessed2/pros/{aws_name}")

os.system("rm temp_downloads/*.rar")
os.system("rm temp_downloads/*.dem")
