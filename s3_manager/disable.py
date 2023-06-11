import boto3
from utils import *
import s3fs
import os


def run(arg: str):
    s3 = boto3.client('s3')
    update_dir(arg)
    # make sure folder structure exists
    generate_data_folder(s3)

    fs = s3fs.S3FileSystem()
    for f in fs.glob(s3path_str(get_demos_folder() / "*", False)):
        print(f'''moving {f} to {get_disabled_folder()}''')
        os.system(f"aws s3 mv s3://{f} s3://{s3path_str(get_disabled_folder(), True)}")


if __name__ == "__main__":
    run(sys.argv[1])
