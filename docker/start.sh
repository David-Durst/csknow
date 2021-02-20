mkdir -p ../demos
docker run --name durst_csgo \
    --mount type=bind,source="$(pwd)"/../demos,target=/home/csgo/demos \
    --rm -d -p 27015:27015 -p 27015:27015/udp durst/csgo:0.1 \
    -console -usercon +game_type 0 +game_mode 1 +mapgroup mg_de_dust2 +map de_dust2 +sv_lan 1
