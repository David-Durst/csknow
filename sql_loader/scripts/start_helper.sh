until docker exec durst_sql_loader mysql --host=localhost --user=root --password=${pass} -e "source /sql/privileges.sql"
do
    sleep 2;
done
docker exec durst_sql_loader bash /scripts/start_helper_docker.sh
