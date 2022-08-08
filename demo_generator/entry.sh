#!/bin/bash
set -x
bash update.sh
python3 -u upload_logs.py >> upload.log 2>> upload.log &

# had problems getting csknow to update itself, so just forcing it here
cd ~/csknow
git pull
git checkout v0.1.1
bash ~/csknow/analytics/scripts/bot_build_docker.sh

bash ~/install_link.sh

bash ~/csknow/analytics/scripts/bot_bt_run_docker.sh >> bot.log 2>> bot.log &

# Believe it or not, if you don't do this srcds_run shits itself
cd ${NONVOLUMESTEAMAPPDIR}

bash docker_run_csgo_bots.sh
