import glob
import os

cur_path = os.path.dirname(os.path.realpath(__file__))
print("cur_path:" + cur_path)
files = glob.glob(cur_path + '/csgo-dedicated-non-volumne/csgo/*.dem') # * means all if need specific format then *.csv
files.sort(key=os.path.getmtime)
for f in files[:-1]:
    print("moving file:" + str(f))
    os.system(f"aws s3 cp {f} s3://csknow/demos/")

