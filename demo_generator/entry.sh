#!/bin/bash
set -x
bash update.sh
python3 -u upload_logs.py >> upload.log 2>> upload.log &

# had problems getting csknow to update itself, so just forcing it here
cd ~/csknow
git pull --tags
git checkout v0.2.0
bash ~/csknow/analytics/scripts/bot_build_docker.sh

bash ~/install_link.sh

if [ ! -v HEURISTICS ]; then
    bash ~/csknow/analytics/scripts/bot_bt_run_docker_heuristics.sh >> bot.log 2>> bot.log &
else 
    bash ~/csknow/analytics/scripts/bot_bt_run_docker.sh >> bot.log 2>> bot.log &
fi


# Believe it or not, if you don't do this srcds_run shits itself
cd ${NONVOLUMESTEAMAPPDIR}

bash docker_run_csgo_bots.sh
