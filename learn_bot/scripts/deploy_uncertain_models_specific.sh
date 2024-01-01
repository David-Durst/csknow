aws s3 cp s3://csknow/models/$1/ saved_models/uncertain_model/ --recursive
mkdir -p models/uncertain_model/
cp saved_models/uncertain_model/* models/uncertain_model/
