# for renaming local files if doing multiple of them
padded_num=$(printf "%09d\n" $1)

mv ../local_data/defusals/local_defusals.csv ../local_data/defusals/local_defusals$padded_num.csv
mv ../local_data/flashed/local_flashed.csv  ../local_data/flashed/local_flashed$padded_num.csv 
mv ../local_data/grenades/local_grenades.csv  ../local_data/grenades/local_grenades$padded_num.csv 
mv ../local_data/kills/local_kills.csv  ../local_data/kills/local_kills$padded_num.csv 
mv ../local_data/player_at_tick/local_player_at_tick.csv   ../local_data/player_at_tick/local_player_at_tick$padded_num.csv  
mv ../local_data/rounds/local_filtered_rounds.csv  ../local_data/rounds/local_filtered_rounds$padded_num.csv
mv ../local_data/rounds/local_unfiltered_rounds.csv  ../local_data/rounds/local_unfiltered_rounds$padded_num.csv
mv ../local_data/ticks/local_ticks.csv  ../local_data/ticks/local_ticks$padded_num.csv
mv ../local_data/explosions/local_explosions.csv  ../local_data/explosions/local_explosions$padded_num.csv 
mv ../local_data/grenade_trajectories/local_grenade_trajectories.csv  ../local_data/grenade_trajectories/local_grenade_trajectories$padded_num.csv 
mv ../local_data/hurt/local_hurt.csv  ../local_data/hurt/local_hurt$padded_num.csv 
mv ../local_data/plants/local_plants.csv   ../local_data/plants/local_plants$padded_num.csv  
mv ../local_data/players/local_players.csv   ../local_data/players/local_players$padded_num.csv  
mv ../local_data/spotted/local_spotted.csv  ../local_data/spotted/local_spotted$padded_num.csv 
mv ../local_data/weapon_fire/local_weapon_fire.csv  ../local_data/weapon_fire/local_weapon_fire$padded_num.csv 
