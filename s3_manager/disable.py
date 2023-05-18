import boto3
from utils import *
import s3fs


def run(arg: str):
    s3 = boto3.client('s3')
    update_dir(arg)
    # make sure folder structure exists
    generate_data_folder(s3)

    fs = s3fs.S3FileSystem()
    for f in fs.glob(s3path_str(DEMOS_FOLDER / "*", False)):
        fs.mv(f, s3path_str(DISABLED_FOLDER, True))

if __name__ == "__main__":
    run(sys.argv[1])
