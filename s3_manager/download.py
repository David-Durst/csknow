from typing import List
import sys
import boto3
from pathlib import Path
from utils import *


def run(args: List[str]):
    s3 = boto3.client('s3')
    # make manual folder structure if doesn't exists
    update_dir(args[0])
    generate_data_folder(s3)
    hdf5_file_name = get_data_name() + ".hdf5"
    local_path = str(Path(__file__).parent / ".." / "demo_parser" / "hdf5" / hdf5_file_name)
    print(f"downloading {hdf5_file_name} from {get_data_root()} to {local_path}")
    s3.download_file(Bucket=BUCKET.bucket, Key=(get_data_root() / hdf5_file_name).key, 
                     Filename=local_path)


if __name__ == "__main__":
    run(sys.argv[1:])
