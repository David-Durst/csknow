#!/usr/bin/python3 
import glob
import os
import time
import urllib.request
import json
from pathlib import Path

cur_path = os.path.dirname(os.path.realpath(__file__))

if "RUNNING_IN_EC2" in os.environ:
    url = "http://169.254.169.254/latest/meta-data/iam/security-credentials/" + os.environ["ROLE"]
else:
    url = "http://169.254.170.2" + os.environ["AWS_CONTAINER_CREDENTIALS_RELATIVE_URI"]

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
    os.system("go run .")
    time.sleep(60)
    num_sleeps += 1
    if num_sleeps == 10:
        num_sleeps = 0
print("done while loop of uploading")
