docker run --name durst_sql_loader \
    --rm -it \
    --mount type=bind,source="$(pwd)"/../local_data,target=/go/src/local_data2 \
    --entrypoint /bin/bash \
    --cap-add=SYS_PTRACE \
    --cap-add=SYS_ADMIN \
    durst/sql_loader:0.1

