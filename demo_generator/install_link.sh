#!/bin/bash
home_link_path=~/bot-link
if [ ! -d ${home_link_path} ]; then
    cd ~
    git clone https://github.com/David-Durst/bot-link.git
    git checkout v0.1.1
fi

csgo_link_path=${NONVOLUMESTEAMAPPDIR}/csgo/addons/sourcemod/scripting/bot-link
if [ ! -d ${csgo_link_path} ]; then
    ln -s ${home_link_path} ${csgo_link_path}
fi

cd ${home_link_path}
git pull --tags
echo "${NONVOLUMESTEAMAPPDIR}" > .csgo_path

cd ${csgo_link_path}/..
./bot-link/deploy.sh link
./bot-link/deploy.sh training
