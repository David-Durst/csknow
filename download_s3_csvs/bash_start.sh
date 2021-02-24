docker run --name durst_download_merge_s3_csvs \
    --rm -it \
    --mount type=bind,source="$(pwd)"/../local_data,target=/go/src/local_data \
    --entrypoint /bin/bash \
    --cap-add=SYS_PTRACE \
    --cap-add=SYS_ADMIN \
    durst/download-merge-s3-csvs:0.1

