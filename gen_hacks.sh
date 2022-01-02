rm local_data/*.csv
rm local_data/*/*.csv
cp backup_visibility_data/*.csv local_data/
cp -r backup_visibility_data/visibilities backup_visibility_data/visibilities_unadjusted local_data/
cd demo_parser
go run . -l -n=demos/no_hacks_4.dem
./copy_local.sh
./update_local.sh 1

cp output_global_id_state.csv input_global_id_state.csv 
go run . -l -f=false -n=demos/with_hacks_3.dem
./copy_local.sh
./update_local.sh 2

cp output_global_id_state.csv input_global_id_state.csv 
go run . -l -f=false -n=demos/9_19_21_durst_t_wang_ct_with_walls.dem
./copy_local.sh
./update_local.sh 3

cp output_global_id_state.csv input_global_id_state.csv 
go run . -l -f=false -n=demos/merrick_kingston_matt_gabe_rory_durst_9_20_21_no_hacks.dem
./copy_local.sh
./update_local.sh 4

cp output_global_id_state.csv input_global_id_state.csv 
go run . -l -f=false -n=demos/merrick_kingston_matt_gabe_rory_durst_9_20_21_wallhacks_2.dem
./copy_local.sh
./update_local.sh 5

declare -a ProDemos=(
    "319_titan-epsilon_de_dust2.dem" 
    "334_natus-vincere-team-ldlc_de_inferno.dem" 
    "336_natus-vincere-team-ldlc_de_overpass.dem"
    "320_titan-epsilon_de_cache.dem" 
    "335_natus-vincere-team-ldlc_de_dust2.dem"
    "337_natus-vincere-team-ldlc_de_mirage.dem"
    "2351898_128353_g2-vs-ence-m3-dust2_5f2f16a6-292a-11ec-8e27-0a58a9feac02.dem"
    "2352501_129549_nip-vs-gambit-m1-dust2_f47b6c3a-3c05-11ec-be9d-0a58a9feac02.dem")
local_files_id=6
for f in ${ProDemos[@]}; do
    cp output_global_id_state.csv input_global_id_state.csv 
    go run . -l -f=false -n=demos/$f
    ./copy_local.sh
    ./update_local.sh $local_files_id
    ((local_files_id++))
done

cd ../download_s3_csvs
./merge.sh
cd ..
rm local_data/defusals.csv
touch local_data/defusals.csv
rm local_data/plants.csv
touch local_data/plants.csv
rm local_data/explosions.csv
touch local_data/explosions.csv
cd analytics
./scripts/run.sh
cp csv_outputs/lookers.csv ../local_data/
cp csv_outputs/nearest_origin.csv ../local_data/
cp csv_outputs/player_in_cover_edge.csv ../local_data/
cp csv_outputs/player_looking_at_cover_edge.csv ../local_data/
cd ..
sed -i '1d' local_data/lookers.csv
