set -x
MAP="${MAP:-bot_playground}"
echo ${MAP} > ${NONVOLUMESTEAMAPPDIR}/csgo/mapcycle.txt
gslt_arg=""
echo $GSLT
if [ ! -z ${GSLT+x} ]
then
    gslt_arg="+sv_setsteamaccount $GSLT -net_port_try 1"
fi
if [ $MAP == "de_dust2" ]
then 
    echo "in"
    ./srcds_run  -game csgo -console -usercon +game_type 0 +game_mode 0 +sv_skirmish_id 12 +mapgroup mg_de_dust2 +map de_dust2 +sv_lan 0 ${gslt_arg} -tickrate 128 -port 27015
else
    echo "out"
    ./srcds_run  -game csgo -console -usercon +sv_autoexec_mapname_cfg 1 +game_type 0 +game_mode 0 +sm_nextmap ${MAP} +mapgroup mg_active +map ${MAP} -tickrate 128
fi
