until docker exec -it durst_sql_loader mysql --host=localhost --user=root --password=${pass} -e "sql/privileges.sql"
do
    sleep 2;
done
