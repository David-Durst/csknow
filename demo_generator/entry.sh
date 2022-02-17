#!/bin/bash
set -x
bash update.sh
python3 -u upload_logs.py >> upload.log 2>> upload.log &

cd ~/csknow/analytics.sh
bash bot_run_docker.sh &

# We assume that if the config is missing, that this is a fresh container
#if [ ! -f "${NONVOLUMESTEAMAPPDIR}/${STEAMAPP}/cfg/server.cfg" ]; then
#	# Download & extract the config
#	wget -qO- "${DLURL}/master/etc/cfg.tar.gz" | tar xvzf - -C "${NONVOLUMESTEAMAPPDIR}/${STEAMAPP}"
#	
#	# Are we in a metamod container?
#	if [ ! -z "$METAMOD_VERSION" ]; then
#		LATESTMM=$(wget -qO- https://mms.alliedmods.net/mmsdrop/"${METAMOD_VERSION}"/mmsource-latest-linux)
#		wget -qO- https://mms.alliedmods.net/mmsdrop/"${METAMOD_VERSION}"/"${LATESTMM}" | tar xvzf - -C "${NONVOLUMESTEAMAPPDIR}/${STEAMAPP}"	
#	fi
#
#	# Are we in a sourcemod container?
#	if [ ! -z "$SOURCEMOD_VERSION" ]; then
#		LATESTSM=$(wget -qO- https://sm.alliedmods.net/smdrop/"${SOURCEMOD_VERSION}"/sourcemod-latest-linux)
#		wget -qO- https://sm.alliedmods.net/smdrop/"${SOURCEMOD_VERSION}"/"${LATESTSM}" | tar xvzf - -C "${NONVOLUMESTEAMAPPDIR}/${STEAMAPP}"
#	fi
#
#	# Change hostname on first launch (you can comment this out if it has done it's purpose)
#	sed -i -e 's/{{SERVER_HOSTNAME}}/'"${SRCDS_HOSTNAME}"'/g' "${NONVOLUMESTEAMAPPDIR}/${STEAMAPP}/cfg/server.cfg"
#fi

# Believe it or not, if you don't do this srcds_run shits itself
cd ${NONVOLUMESTEAMAPPDIR}

bash docker_run_csgo_bots.sh
