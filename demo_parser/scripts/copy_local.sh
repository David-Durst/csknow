script_dir="tmp"
#https://www.ostricher.com/2014/10/the-right-way-to-get-the-directory-of-a-bash-script/
get_script_dir () {
     SOURCE="${BASH_SOURCE[0]}"
     # While $SOURCE is a symlink, resolve it
     while [ -h "$SOURCE" ]; do
          DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
          SOURCE="$( readlink "$SOURCE" )"
          # If $SOURCE was a relative symlink (so no "/" as prefix, need to resolve it relative to the symlink base directory
          [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
     done
     DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
     script_dir="$DIR"
}
get_script_dir

cd ${script_dir}/..

cp csv_outputs/dimension_table*.csv ../local_data/
cp csv_outputs/global_games.csv ../local_data/
cp csv_outputs/local_defusals.csv ../local_data/defusals/
cp csv_outputs/local_flashed.csv ../local_data/flashed/
cp csv_outputs/local_grenades.csv ../local_data/grenades/
cp csv_outputs/local_kills.csv ../local_data/kills/
cp csv_outputs/local_player_at_tick.csv ../local_data/player_at_tick/
cp csv_outputs/local_filtered_rounds.csv ../local_data/filtered_rounds/
cp csv_outputs/local_unfiltered_rounds.csv ../local_data/unfiltered_rounds/
cp csv_outputs/local_ticks.csv ../local_data/ticks/
cp csv_outputs/local_explosions.csv ../local_data/explosions/
cp csv_outputs/local_grenade_trajectories.csv ../local_data/grenade_trajectories/
cp csv_outputs/local_hurt.csv ../local_data/hurt/
cp csv_outputs/local_plants.csv ../local_data/plants/
cp csv_outputs/local_players.csv ../local_data/players/
cp csv_outputs/local_spotted.csv ../local_data/spotted/
cp csv_outputs/local_footstep.csv ../local_data/footstep/
cp csv_outputs/local_weapon_fire.csv ../local_data/weapon_fire/

cd ${script_dir}/../../download_s3_csvs
./merge.sh
