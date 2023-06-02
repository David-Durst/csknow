#!/bin/bash
#
cp learn_bot/latent/checkpoints/$1/delta_pos_* saved_models/latent_model/

aws_timestamp=$(date '+%Y_%m_%d_%H_%M_%S')
for f in saved_models/latent_model/*; do
    aws s3 cp ${f} s3://csknow/models/${aws_timestamp}/
    aws s3 cp ${f} s3://csknow/models/latest/
done
