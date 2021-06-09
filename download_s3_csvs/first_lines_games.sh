dir_path=../local_data

head -n $1 ${dir_path}/global_games.csv > ${dir_path}/global_games2.csv
mv ${dir_path}/global_games2.csv ${dir_path}/global_games.csv


