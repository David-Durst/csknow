#!/bin/bash

for p in learn_bot/latent/checkpoints/$1*; do
    ./scripts/save_models_specific.sh $(basename $p)
done
