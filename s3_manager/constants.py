from s3path import PureS3Path
BUCKET = PureS3Path.from_uri("s3://csknow")
DISABLED_FOLDER = DATA_ROOT / "disabled"
DEMOS_FOLDER = DATA_ROOT / "demos"
HA_FOLDER = DATA_ROOT / "demos"
MANUAL_UNPROCESSED = MANUAL_DATA / "unprocessed"
MANUAL_UNPROCESSED_BOTS = MANUAL_DATA / "unprocessed" / "bots"
MANUAL_UNPROCESSED_PROS = MANUAL_DATA / "unprocessed" / "pros"
BOT_RETAKES_DISABLED = BOT_RETAKES_DATA / "disabled"
BOT_RETAKES_CSVS = BOT_RETAKES_DATA / "csvs"
BOT_RETAKES_PROCESSED = BOT_RETAKES_DATA / "processed"
BOT_RETAKES_PROCESSED_BOTS = BOT_RETAKES_DATA / "processed" / "bots"
BOT_RETAKES_PROCESSED_PROS = BOT_RETAKES_DATA / "processed" / "pros"
BOT_RETAKES_UNPROCESSED = BOT_RETAKES_DATA / "unprocessed"
BOT_RETAKES_UNPROCESSED_BOTS = BOT_RETAKES_DATA / "unprocessed" / "bots"
BOT_RETAKES_UNPROCESSED_PROS = BOT_RETAKES_DATA / "unprocessed" / "pros"
BIG_TRAIN_DISABLED = BIG_TRAIN_DATA / "disabled"
BIG_TRAIN_CSVS = BIG_TRAIN_DATA / "csvs"
BIG_TRAIN_PROCESSED = BIG_TRAIN_DATA / "processed"
BIG_TRAIN_PROCESSED_BOTS = BIG_TRAIN_DATA / "processed" / "bots"
BIG_TRAIN_PROCESSED_PROS = BIG_TRAIN_DATA / "processed" / "pros"
BIG_TRAIN_UNPROCESSED = BIG_TRAIN_DATA / "unprocessed"
BIG_TRAIN_UNPROCESSED_BOTS = BIG_TRAIN_DATA / "unprocessed" / "bots"
BIG_TRAIN_UNPROCESSED_PROS = BIG_TRAIN_DATA / "unprocessed" / "pros"
ROLLOUT_DISABLED = ROLLOUT_DATA / "disabled"
ROLLOUT_CSVS = ROLLOUT_DATA / "csvs"
ROLLOUT_PROCESSED = ROLLOUT_DATA / "processed"
ROLLOUT_PROCESSED_BOTS = ROLLOUT_DATA / "processed" / "bots"
ROLLOUT_PROCESSED_PROS = ROLLOUT_DATA / "processed" / "pros"
ROLLOUT_UNPROCESSED = ROLLOUT_DATA / "unprocessed"
ROLLOUT_UNPROCESSED_BOTS = ROLLOUT_DATA / "unprocessed" / "bots"
ROLLOUT_UNPROCESSED_PROS = ROLLOUT_DATA / "unprocessed" / "pros"



def s3path_str(p: PureS3Path, is_dir) -> str:
    return p.bucket + "/" + p.key + ("/" if is_dir else "")
