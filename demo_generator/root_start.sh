mkdir -p ../demos
docker run --name durst_csgo \
    --rm -it --net=host --user root \
    -e SRCDS_MAPGROUP=mg_de_durst2 -e SRCDS_STARTMAP=de_dust2 \
    --mount type=bind,source="$(pwd)"/../demos,target=/home/steam/demos \
    --entrypoint /bin/bash \
    durst/csgo:0.3

