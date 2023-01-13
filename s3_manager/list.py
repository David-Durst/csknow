import pathlib

import boto3
import s3path

from constants import *
from utils import generate_data_folder
import s3fs

def print_folder(fs: s3fs.S3FileSystem, remote_path: s3path.PureS3Path):
    for index, k in enumerate(fs.ls(remote_path)):
        if index == 0:
            print(k)
        else:
            print(pathlib.Path(k).name, end="    ")
    print("")


def run():
    s3 = boto3.client('s3')

    # make sure folder structure exists
    generate_data_folder(s3)

    fs = s3fs.S3FileSystem()
    print_folder(fs, MANUAL_DISABLED)
    print_folder(fs, MANUAL_UNPROCESSED_BOTS)
    print_folder(fs, MANUAL_UNPROCESSED_PROS)
    print_folder(fs, MANUAL_PROCESSED_BOTS)
    print_folder(fs, MANUAL_PROCESSED_PROS)
    print_folder(fs, MANUAL_CSVS / "local")


if __name__ == "__main__":
    run()
