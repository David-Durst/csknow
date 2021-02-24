mkdir -p ../demos
docker run --name durst_csgo \
    --rm -it --net=host \
    --mount type=bind,source="$(pwd)"/../local_data,target=/go/src/local_data \
    --entrypoint /bin/bash \
    durst/download-merge-s3-csvs:0.1

