aws s3 cp s3://csknow/models/$1/ saved_models/latent_model/ --recursive
cp saved_models/latent_model/* models/latent_model/