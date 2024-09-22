rm -f csgo-ds/bin/libgcc_s.so.1
cd csgo-ds
./srcds_run -game csgo -console -usercon -con_logfile +game_type 0 +game_mode 0 +sv_skirmish_id 12 +mapgroup mg_de_dust2 +map de_dust2 -tickrate 128 

