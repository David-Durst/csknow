from s3path import PureS3Path
BUCKET = PureS3Path.from_uri("s3://csknow")
PIPELINE_TEMPLATE = BUCKET / "demos" / "train_data_template"
TRAIN_DATA = BUCKET / "demos" / "train_data"
MANUAL_DATA = BUCKET / "demos" / "manual_data"
MANUAL_CSVS = MANUAL_DATA / "csvs"
MANUAL_PROCESSED = MANUAL_DATA / "processed"
MANUAL_PROCESSED_BOTS = MANUAL_DATA / "processed" / "bots"
MANUAL_PROCESSED_PROS = MANUAL_DATA / "processed" / "pros"
MANUAL_UNPROCESSED = MANUAL_DATA / "unprocessed"
MANUAL_UNPROCESSED_BOTS = MANUAL_DATA / "unprocessed" / "bots"
MANUAL_UNPROCESSED_PROS = MANUAL_DATA / "unprocessed" / "pros"


def s3path_str(p: PureS3Path, is_dir) -> str:
    return p.bucket + "/" + p.key + ("/" if is_dir else "")
