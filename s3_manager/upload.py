from typing import List
import sys
import boto3
from pathlib import Path
from utils import *
import os


def run(args: List[str]):
    s3 = boto3.client('s3')
    # make manual folder structure if doesn't exists
    update_dir(args[0])
    generate_data_folder(s3)
    for arg in args[1:]:
        print(f"uploading {arg} to {get_demos_folder()}")
        os.system(f"aws s3 cp {arg} s3://{BUCKET.bucket}/{(get_demos_folder() / Path(arg).name).key}")


if __name__ == "__main__":
    run(sys.argv[1:])
