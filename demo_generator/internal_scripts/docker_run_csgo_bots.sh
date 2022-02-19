MAP="${MAP:-bot_playground}"
echo ${MAP} > ${NONVOLUMESTEAMAPPDIR}/csgo/mapcycle.txt
./srcds_run  -game csgo -console -usercon +sv_autoexec_mapname_cfg 1 +game_type 0 +game_mode 0 +sm_nextmap ${MAP} +mapgroup mg_active +map ${MAP} -tickrate 128 -port 27016
