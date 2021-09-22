rm local_data/*.csv
rm local_data/*/*.csv
cd demo_parser
go run . -l -n=no_hacks_4.dem
./copy_local.sh
./update_local.sh 1

cp output_global_id_state.csv input_global_id_state.csv 
go run . -l -f=false -n=with_hacks_3.dem
./copy_local.sh
./update_local.sh 2

cp output_global_id_state.csv input_global_id_state.csv 
go run . -l -f=false -n=9_19_21_durst_t_wang_ct_with_walls.dem
./copy_local.sh
./update_local.sh 3

cp output_global_id_state.csv input_global_id_state.csv 
go run . -l -f=false -n=merrick_kingston_matt_gabe_rory_durst_9_20_21_no_hacks.dem
./copy_local.sh
./update_local.sh 4

cp output_global_id_state.csv input_global_id_state.csv 
go run . -l -f=false -n=merrick_kingston_matt_gabe_rory_durst_9_20_21_wallhacks_2.dem
./copy_local.sh
./update_local.sh 5

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
cd ..
sed -i '1d' local_data/lookers.csv
