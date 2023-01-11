from constants import *


def s3_mkdir(s3, bucket_path: PureS3Path):
    s3.put_object(Bucket=BUCKET.bucket, Key=bucket_path.key + "/")


def generate_data_folder(s3):
    s3_mkdir(s3, MANUAL_DATA)
    s3_mkdir(s3, MANUAL_DISABLED)
    s3_mkdir(s3, MANUAL_CSVS)
    s3_mkdir(s3, MANUAL_CSVS / "global")
    s3_mkdir(s3, MANUAL_CSVS / "local")
    s3_mkdir(s3, MANUAL_PROCESSED)
    s3_mkdir(s3, MANUAL_PROCESSED_BOTS)
    s3_mkdir(s3, MANUAL_PROCESSED_PROS)
    s3_mkdir(s3, MANUAL_UNPROCESSED)
    s3_mkdir(s3, MANUAL_UNPROCESSED_BOTS)
    s3_mkdir(s3, MANUAL_UNPROCESSED_PROS)
