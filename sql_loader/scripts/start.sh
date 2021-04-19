docker run --name durst_sql_loader \
    --rm \
    --mount type=bind,source="$(pwd)"/../local_data,target=/go/src/local_data2 \
    -p 127.0.0.1:3306:3306 \
    -e MYSQL_ROOT_PASSWORD=local_password \
    durst/sql_loader:0.1
