import pathlib
import sys

import boto3
import s3path

from utils import *
import s3fs

def print_folder(fs: s3fs.S3FileSystem, remote_path: s3path.PureS3Path):
    for index, k in enumerate(fs.ls(remote_path)):
        if index == 0:
            print(k)
        else:
            print(pathlib.Path(k).name, end="    ")
    print("")


def run(arg: str):
    s3 = boto3.client('s3')

    update_dir(arg)
    # make sure folder structure exists
    generate_data_folder(s3)

    fs = s3fs.S3FileSystem()
    print_folder(fs, get_data_root())
    print_folder(fs, get_disabled_folder())
    print_folder(fs, get_demos_folder())
    print_folder(fs, get_hdf5_folder())


if __name__ == "__main__":
    run(sys.argv[1])
