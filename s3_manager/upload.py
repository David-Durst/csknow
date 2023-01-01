from typing import List
import sys
import boto3
from constants import *
from pathlib import Path
from utils import generate_data_folder


def run(args: List[str]):
    s3 = boto3.client('s3')
    # make manual folder structure if doesn't exists
    generate_data_folder(s3)
    for arg in args:
        print(f"uploading {arg}")
        s3.upload_file(Filename=arg, Bucket=BUCKET.bucket, Key=(MANUAL_DATA / MANUAL_UNPROCESSED / "bots" / Path(arg).name).key)


if __name__ == "__main__":
    run(sys.argv[1:])
