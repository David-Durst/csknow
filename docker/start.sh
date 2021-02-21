mkdir -p ../demos
docker run --name durst_csgo \
    --rm --net=host \
    -e SRCDS_MAPGROUP=mg_de_durst2 -e SRCDS_STARTMAP=de_dust2 \
    --mount type=bind,source="$(pwd)"/../demos,target=/home/steam/demos \
    durst/csgo:0.1

