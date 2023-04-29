import boto3
from constants import *
from utils import generate_data_folder
import s3fs


def run():
    s3 = boto3.client('s3')
    # make sure folder structure exists
    generate_data_folder(s3)

    # make sure nothing considered already processed, but leave unprocessed in place
    fs = s3fs.S3FileSystem()
    for f in fs.glob(s3path_str(BOT_RETAKES_PROCESSED_PROS / "*", False)):
        fs.mv(f, s3path_str(BOT_RETAKES_UNPROCESSED_PROS, True))
    for f in fs.glob(s3path_str(BOT_RETAKES_PROCESSED_BOTS / "*", False)):
        fs.mv(f, s3path_str(BOT_RETAKES_UNPROCESSED_BOTS, True))

    # clear the csv data
    s3_resource = boto3.resource('s3')
    bucket = s3_resource.Bucket(BUCKET.bucket)
    bucket.objects.filter(Prefix=BOT_RETAKES_CSVS.key).delete()

    # create csv folders (as well as everything else, doesn't hurt since it's not slow and won't delete stuff)
    generate_data_folder(s3)

if __name__ == "__main__":
    run()
