until docker exec durst_sql_loader mysql --host=localhost --user=root --password=${pass} -e "source sql/privileges.sql"
do
    sleep 2;
done
docker exec durst_sql_loader mysql --host=localhost --user=root --password=${pass} -e "source sql/create_schema_v7_mysql.sql" csknow

dir_path=local_data2

for f in ${dir_path}/defusals/*.csv
do
    docker exec durst_sql_loader mysql --host=localhost --user=root --password=${pass} -e "LOAD DATA INFILE ${f} INTO TABLE defusals FIELDS TERMINATED BY ',' LINES TERMINATED BY '/n' IGNORE ROWS 1;" csknow
done

#defusals  explosions flashed  grenade_trajectories  grenades  hurt  kills  plants  player_at_tick  players  spotted  ticks  weapon_fire
