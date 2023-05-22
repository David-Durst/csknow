import sys

from s3path import PureS3Path
BUCKET = PureS3Path.from_uri("s3://csknow")
DATA_ROOT = BUCKET / "demos" / "manual_data"
DISABLED_FOLDER = DATA_ROOT / "disabled"
DEMOS_FOLDER = DATA_ROOT / "demos"
HDF5_FOLDER = DATA_ROOT / "hdf5"

MANUAL_DATA_NAME = "manual_data"
BOT_RETAKES_NAME = "bot_retakes_data"
BIG_TRAIN_NAME = "big_train_data"
ROLLOUT_NAME = "rollout_data"


def get_data_root():
    return DATA_ROOT


def get_disabled_folder():
    return DISABLED_FOLDER


def get_demos_folder():
    return DEMOS_FOLDER


def get_hdf5_folder():
    return HDF5_FOLDER


def get_hdf5_file():
    if dir_cmd == "manual":
        return MANUAL_DATA_NAME
    elif dir_cmd == "bot_retakes":
        return BOT_RETAKES_NAME
    elif dir_cmd == "big_train":
        return BIG_TRAIN_NAME
    elif dir_cmd == "rollout":
        return ROLLOUT_NAME


def update_dir(dir_cmd: str):
    global DATA_ROOT, DISABLED_FOLDER, DEMOS_FOLDER, HDF5_FOLDER
    if dir_cmd == "manual":
        DATA_ROOT = BUCKET / "demos" / MANUAL_DATA_NAME
    elif dir_cmd == "bot_retakes":
        DATA_ROOT = BUCKET / "demos" / BOT_RETAKES_NAME
    elif dir_cmd == "big_train":
        DATA_ROOT = BUCKET / "demos" / BIG_TRAIN_NAME
    elif dir_cmd == "rollout":
        DATA_ROOT = BUCKET / "demos" / ROLLOUT_NAME
    else:
        print("invalid data set command")
        sys.exit(1)
    DISABLED_FOLDER = DATA_ROOT / "disabled"
    DEMOS_FOLDER = DATA_ROOT / "demos"
    HDF5_FOLDER = DATA_ROOT / "hdf5"


def s3_mkdir(s3, bucket_path: PureS3Path):
    s3.put_object(Bucket=BUCKET.bucket, Key=bucket_path.key + "/")


def s3path_str(p: PureS3Path, is_dir) -> str:
    return p.bucket + "/" + p.key + ("/" if is_dir else "")


def generate_data_folder(s3):
    s3_mkdir(s3, DATA_ROOT)
    s3_mkdir(s3, DISABLED_FOLDER)
    s3_mkdir(s3, DEMOS_FOLDER)
    s3_mkdir(s3, HDF5_FOLDER)
