#!/bin/bash
set -x
bash update.sh
python3 -u upload_logs.py >> upload.log 2>> upload.log &

# had problems getting csknow to update itself, so just forcing it here
cd ~/csknow
git pull
git pull --tags
git checkout v0.2.1

ssh-keygen -b 2048 -t rsa -f /tmp/sshkey -q -N ""

#cd analytics
#scripts/install_dependencies.sh
#cd ..

bash ~/csknow/analytics/scripts/bot_build_docker.sh

bash ~/install_link.sh

cd learn_bot
scripts/deploy_latent_models.sh
cd ..



if [ ! -v HEURISTICS ]; then
    bash ~/csknow/analytics/scripts/bot_bt_run_docker.sh >> bot.log 2>> bot.log &
else 
    bash ~/csknow/analytics/scripts/bot_bt_run_docker_heuristics.sh >> bot.log 2>> bot.log &
fi


# Believe it or not, if you don't do this srcds_run shits itself
cd ${NONVOLUMESTEAMAPPDIR}

bash docker_run_csgo_bots.sh
