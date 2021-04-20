until docker exec durst_sql_loader mysql --host=localhost --user=root --password=${pass} -e "source sql/privileges.sql"
do
    sleep 2;
done
