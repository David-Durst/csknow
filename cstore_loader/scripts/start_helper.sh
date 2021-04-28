until docker exec durst_cstore_loader mysql --host=localhost --user=root --password=${pass} -e "source /sql/privileges.sql"
do
    sleep 2;
done
docker exec -e pass=${pass} durst_cstore_loader bash /scripts/start_helper_docker.sh
