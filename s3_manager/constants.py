from s3path import PureS3Path
BUCKET = PureS3Path.from_uri("s3://csknow")
PIPELINE_TEMPLATE = BUCKET / "demos" / "train_data_template"
TRAIN_DATA = BUCKET / "demos" / "train_data"
MANUAL_DATA = BUCKET / "demos" / "manual_data"
MANUAL_DISABLED = MANUAL_DATA / "disabled"
MANUAL_CSVS = MANUAL_DATA / "csvs"
MANUAL_PROCESSED = MANUAL_DATA / "processed"
MANUAL_PROCESSED_BOTS = MANUAL_DATA / "processed" / "bots"
MANUAL_PROCESSED_PROS = MANUAL_DATA / "processed" / "pros"
MANUAL_UNPROCESSED = MANUAL_DATA / "unprocessed"
MANUAL_UNPROCESSED_BOTS = MANUAL_DATA / "unprocessed" / "bots"
MANUAL_UNPROCESSED_PROS = MANUAL_DATA / "unprocessed" / "pros"
BOT_RETAKES_DATA = BUCKET / "demos" / "bot_retakes_data"
BOT_RETAKES_DISABLED = BOT_RETAKES_DATA / "disabled"
BOT_RETAKES_CSVS = BOT_RETAKES_DATA / "csvs"
BOT_RETAKES_PROCESSED = BOT_RETAKES_DATA / "processed"
BOT_RETAKES_PROCESSED_BOTS = BOT_RETAKES_DATA / "processed" / "bots"
BOT_RETAKES_PROCESSED_PROS = BOT_RETAKES_DATA / "processed" / "pros"
BOT_RETAKES_UNPROCESSED = BOT_RETAKES_DATA / "unprocessed"
BOT_RETAKES_UNPROCESSED_BOTS = BOT_RETAKES_DATA / "unprocessed" / "bots"
BOT_RETAKES_UNPROCESSED_PROS = BOT_RETAKES_DATA / "unprocessed" / "pros"



def s3path_str(p: PureS3Path, is_dir) -> str:
    return p.bucket + "/" + p.key + ("/" if is_dir else "")
