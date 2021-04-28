until docker exec durst_sql_loader psql -h 127.0.0.1 -U root  --password=${pass} -e "source /sql/privileges.sql"
do
    sleep 2;
done
docker exec -e pass=${pass} durst_sql_loader bash /scripts/start_helper_docker.sh
