#!/bin/bash
mkdir -p "${NONVOLUMESTEAMAPPDIR}" || true  

bash "${STEAMCMDDIR}/steamcmd.sh" +force_install_dir "${NONVOLUMESTEAMAPPDIR}" \
                +login anonymous \
				+app_update "${STEAMAPPID}" \
				+quit
