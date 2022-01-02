psql --user=postgres -f /sql/create_schema_v8_postgres.sql -d csknow
psql --user=postgres -f /sql/create_schema_v1_hand_visibility.sql -d csknow

dir_path=/local_data

psql --user=postgres -d csknow -c "\\copy equipment FROM '${dir_path}/dimension_table_equipment.csv' csv header null '\\N';"
psql --user=postgres -d csknow -c "\\copy game_types FROM '${dir_path}/dimension_table_game_types.csv' csv header null '\\N';"
psql --user=postgres -d csknow -c "\\copy hit_groups FROM '${dir_path}/dimension_table_hit_groups.csv' csv header null '\\N';"
psql --user=postgres -d csknow -c "\\copy cover_origins FROM '${dir_path}/dimension_table_cover_origins.csv' csv header null '\\N';"
psql --user=postgres -d csknow -c "\\copy cover_edges FROM '${dir_path}/dimension_table_cover_edges.csv' csv header null '\\N';"
psql --user=postgres -d csknow -c "\\copy games FROM '${dir_path}/global_games.csv' csv header null '\\N';"
#mysql --host=localhost --user=root --password=${pass} -e "LOAD DATA INFILE  INTO TABLE equipment FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n' IGNORE 1 ROWS;" csknow
#mysql --host=localhost --user=root --password=${pass} -e "LOAD DATA INFILE '${dir_path}/dimension_table_game_types.csv' INTO TABLE game_types FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n' IGNORE 1 ROWS;" csknow
#mysql --host=localhost --user=root --password=${pass} -e "LOAD DATA INFILE '${dir_path}/dimension_table_hit_groups.csv' INTO TABLE hit_groups FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n' IGNORE 1 ROWS;" csknow
#mysql --host=localhost --user=root --password=${pass} -e "LOAD DATA INFILE '${dir_path}/global_games.csv' INTO TABLE games FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n' IGNORE 1 ROWS;" csknow
#
for name in players rounds ticks player_at_tick spotted weapon_fire kills hurt grenades flashed grenade_trajectories plants defusals explosions hand_visibility visibilities visibilities_unadjusted lookers actions \
	nearest_origin player_in_cover_edge player_looking_at_cover_edge
do
    echo "loading ${name}$"
    psql --user=postgres -d csknow -c "ALTER TABLE ${name} SET UNLOGGED;"
    psql --user=postgres -d csknow -c "\\copy ${name} FROM '${dir_path}/${name}.csv' csv null '\\N';"
    psql --user=postgres -d csknow -c "ALTER TABLE ${name} SET LOGGED;"
done
psql --user=postgres -f /sql/create_fk_v8_postgres.sql -d csknow
echo "done loading"
#defusals  explosions flashed  grenade_trajectories  grenades  hurt  kills  plants  player_at_tick  players  spotted  ticks  weapon_fire
