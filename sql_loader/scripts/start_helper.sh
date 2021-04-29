until docker exec durst_sql_loader createdb csknow --user=postgres
do
    sleep 2;
done
docker exec durst_sql_loader psql --user=postgres -f /sql/privileges.sql -d csknow
docker exec durst_sql_loader bash /scripts/start_helper_docker.sh

