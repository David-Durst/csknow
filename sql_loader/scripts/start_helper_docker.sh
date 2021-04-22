mysql --host=localhost --user=root --password=${pass} -e "source /sql/create_schema_v7_mysql.sql" csknow

dir_path=/local_data2

mysql --host=localhost --user=root --password=${pass} -e "LOAD DATA INFILE '${dir_path}/fact_table_equipment.csv' INTO TABLE equipment FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n' IGNORE 1 ROWS;" csknow
mysql --host=localhost --user=root --password=${pass} -e "LOAD DATA INFILE '${dir_path}/fact_table_game_types.csv' INTO TABLE game_types FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n' IGNORE 1 ROWS;" csknow
mysql --host=localhost --user=root --password=${pass} -e "LOAD DATA INFILE '${dir_path}/fact_table_hit_groups.csv' INTO TABLE hit_groups FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n' IGNORE 1 ROWS;" csknow

for f in ${dir_path}/defusals/*.csv
do
    mysql --host=localhost --user=root --password=${pass} -e "LOAD DATA INFILE '${f}' INTO TABLE defusals FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n' IGNORE 1 ROWS;" csknow
done

#defusals  explosions flashed  grenade_trajectories  grenades  hurt  kills  plants  player_at_tick  players  spotted  ticks  weapon_fire
