mkdir -p ../demos
docker run --name durst_csgo \
    --rm -it --net=host \
    --mount type=bind,source="$(pwd)"/../demos,target=/home/steam/demos \
    --entrypoint /bin/bash \
    durst/csgo:0.4

