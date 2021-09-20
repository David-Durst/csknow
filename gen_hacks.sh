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
cd ../download_s3_csvs
./merge.sh
cd ..
rm local_data/defusals.csv
touch local_data/defusals.csv
