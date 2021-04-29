psql --user=postgres -f /sql/create_schema_v7_mysql.sql -d csknow

dir_path=/local_data

#mysql --host=localhost --user=root --password=${pass} -e "LOAD DATA INFILE '${dir_path}/dimension_table_equipment.csv' INTO TABLE equipment FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n' IGNORE 1 ROWS;" csknow
#mysql --host=localhost --user=root --password=${pass} -e "LOAD DATA INFILE '${dir_path}/dimension_table_game_types.csv' INTO TABLE game_types FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n' IGNORE 1 ROWS;" csknow
#mysql --host=localhost --user=root --password=${pass} -e "LOAD DATA INFILE '${dir_path}/dimension_table_hit_groups.csv' INTO TABLE hit_groups FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n' IGNORE 1 ROWS;" csknow
#mysql --host=localhost --user=root --password=${pass} -e "LOAD DATA INFILE '${dir_path}/global_games.csv' INTO TABLE games FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n' IGNORE 1 ROWS;" csknow
#
#for name in players rounds ticks player_at_tick spotted weapon_fire kills hurt grenades flashed grenade_trajectories plants defusals explosions
#do
#    echo "loading ${name}$"
#    mysqlimport --fields-terminated-by=, --user=root --password=${pass} csknow ${dir_path}/${name}.csv
#done
echo "done loading"
#defusals  explosions flashed  grenade_trajectories  grenades  hurt  kills  plants  player_at_tick  players  spotted  ticks  weapon_fire
