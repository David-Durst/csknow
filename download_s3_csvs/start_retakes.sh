docker run --name durst_download_merge_s3_csvs \
    --rm \
    --mount type=bind,source="$(pwd)"/../manual_data,target=/go/src/local_data \
    --cap-add=SYS_PTRACE \
    --cap-add=SYS_ADMIN \
    durst/download-merge-s3-csvs:0.2 go run . -rd -t=false
