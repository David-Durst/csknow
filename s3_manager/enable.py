from typing import List

import boto3
from utils import *
import s3fs
import os


def run(args: List[str]):
    s3 = boto3.client('s3')
    update_dir(args[0])
    # make sure folder structure exists
    generate_data_folder(s3)

    fs = s3fs.S3FileSystem()

    file_name = args[1]
    src = get_disabled_folder() / file_name
    dst = get_demos_folder() / file_name
    print(f'''moving {src} to {dst}''')
    os.system(f"aws s3 mv s3://{s3path_str(src, False)} s3://{s3path_str(dst, False)}")


if __name__ == "__main__":
    run(sys.argv[1:])
