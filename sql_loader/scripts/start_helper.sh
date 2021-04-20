until docker exec durst_sql_loader mysql --host=localhost --user=root --password=${pass} -e "source sql/privileges.sql"
do
    sleep 2;
done
docker exec durst_sql_loader mysql --host=localhost --user=root --password=da_b3ars_d000d -e "source sql/create_schema_v6_mysql.sql" csknow
