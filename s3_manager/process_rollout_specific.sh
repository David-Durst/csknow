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

# create tick data
get_script_dir
scp steam@54.219.195.192:csgo-ds/csgo/$1 $2.dem
python disable.py rollout
python upload.py rollout $2.dem
python list.py rollout
cd ../demo_parser
go run cmd/main.go -ro -dn _$2
cd ../analytics
./scripts/create_rollout_other_datasets.sh $2

# do comparison with human data
analytics_script_dir=${script_dir}/../analytics/scripts/

mkdir -p ${analytics_script_dir}/../build
cd ${analytics_script_dir}/../build
cmake .. -DCMAKE_BUILD_TYPE=Release
if make -j; then
    chmod a+x csknow_compare_train_datasets
    ./csknow_compare_train_datasets ${analytics_script_dir}/../rollout_outputs/behaviorTreeTeamFeatureStore_$2.hdf5 ${analytics_script_dir}/../all_train_outputs $2_to_human_trajectorySimilarity.hdf5 0 0

    ./csknow_compare_train_datasets ${analytics_script_dir}/../all_train_outputs ${analytics_script_dir}/../rollout_outputs/behaviorTreeTeamFeatureStore_$2.hdf5 human_to_$2_trajectorySimilarity.hdf5 0 0
fi
