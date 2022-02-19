#!/bin/bash
set -x
bash update.sh
python3 -u upload_logs.py >> upload.log 2>> upload.log &

bash ~/csknow/analytics/scripts/bot_run_docker.sh &

bash ~/install_link.sh

# Believe it or not, if you don't do this srcds_run shits itself
cd ${NONVOLUMESTEAMAPPDIR}

bash docker_run_csgo_bots_1v1.sh
