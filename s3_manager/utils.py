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

def generate_bot_retakes_data_folder(s3):
    s3_mkdir(s3, BOT_RETAKES_DATA)
    s3_mkdir(s3, BOT_RETAKES_DISABLED)
    s3_mkdir(s3, BOT_RETAKES_CSVS)
    s3_mkdir(s3, BOT_RETAKES_CSVS / "global")
    s3_mkdir(s3, BOT_RETAKES_CSVS / "local")
    s3_mkdir(s3, BOT_RETAKES_PROCESSED)
    s3_mkdir(s3, BOT_RETAKES_PROCESSED_BOTS)
    s3_mkdir(s3, BOT_RETAKES_PROCESSED_PROS)
    s3_mkdir(s3, BOT_RETAKES_UNPROCESSED)
    s3_mkdir(s3, BOT_RETAKES_UNPROCESSED_BOTS)
    s3_mkdir(s3, BOT_RETAKES_UNPROCESSED_PROS)

def generate_big_train_data_folder(s3):
    s3_mkdir(s3, BIG_TRAIN_DATA)
    s3_mkdir(s3, BIG_TRAIN_DISABLED)
    s3_mkdir(s3, BIG_TRAIN_CSVS)
    s3_mkdir(s3, BIG_TRAIN_CSVS / "global")
    s3_mkdir(s3, BIG_TRAIN_CSVS / "local")
    s3_mkdir(s3, BIG_TRAIN_PROCESSED)
    s3_mkdir(s3, BIG_TRAIN_PROCESSED_BOTS)
    s3_mkdir(s3, BIG_TRAIN_PROCESSED_PROS)
    s3_mkdir(s3, BIG_TRAIN_UNPROCESSED)
    s3_mkdir(s3, BIG_TRAIN_UNPROCESSED_BOTS)
    s3_mkdir(s3, BIG_TRAIN_UNPROCESSED_PROS)

def generate_rollout_data_folder(s3):
    s3_mkdir(s3, ROLLOUT_DATA)
    s3_mkdir(s3, ROLLOUT_DISABLED)
    s3_mkdir(s3, ROLLOUT_CSVS)
    s3_mkdir(s3, ROLLOUT_CSVS / "global")
    s3_mkdir(s3, ROLLOUT_CSVS / "local")
    s3_mkdir(s3, ROLLOUT_PROCESSED)
    s3_mkdir(s3, ROLLOUT_PROCESSED_BOTS)
    s3_mkdir(s3, ROLLOUT_PROCESSED_PROS)
    s3_mkdir(s3, ROLLOUT_UNPROCESSED)
    s3_mkdir(s3, ROLLOUT_UNPROCESSED_BOTS)
    s3_mkdir(s3, ROLLOUT_UNPROCESSED_PROS)
