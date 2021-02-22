#!/bin/bash
mkdir -p "${NONVOLUMESTEAMAPPDIR}" || true  

bash "${STEAMCMDDIR}/steamcmd.sh" +login anonymous \
				+force_install_dir "${NONVOLUMESTEAMAPPDIR}" \
				+app_update "${STEAMAPPID}" \
				+quit			+quit
