#!/bin/bash
set -x
mkdir -p "${NONVOLUMESTEAMAPPDIR}" || true  

bash "${STEAMCMDDIR}/steamcmd.sh" +login anonymous \
				+force_install_dir "${NONVOLUMESTEAMAPPDIR}" \
				+app_update "${NONVOLUMESTEAMAPPID}" \
				+quit

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

bash "${NONVOLUMESTEAMAPPDIR}/srcds_run" -game "${STEAMAPP}" -console -autoupdate \
			-steam_dir "${STEAMCMDDIR}" \
			-steamcmd_script "${HOMEDIR}/${STEAMAPP}_update.txt" \
			-usercon \
			+fps_max "${SRCDS_FPSMAX}" \
			-tickrate "${SRCDS_TICKRATE}" \
			-port "${SRCDS_PORT}" \
			+tv_port "${SRCDS_TV_PORT}" \
			+clientport "${SRCDS_CLIENT_PORT}" \
			-maxplayers_override "${SRCDS_MAXPLAYERS}" \
			+game_type "${SRCDS_GAMETYPE}" \
			+game_mode "${SRCDS_GAMEMODE}" \
			+mapgroup "${SRCDS_MAPGROUP}" \
			+map "${SRCDS_STARTMAP}" \
			+rcon_password "${SRCDS_RCONPW}" \
			+sv_password "${SRCDS_PW}" \
			+sv_region "${SRCDS_REGION}" \
			+sv_lan 1
