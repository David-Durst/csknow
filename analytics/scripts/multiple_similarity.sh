 /home/durst/dev/csknow/analytics/all_train_outputs humanTrajectorySimilarity.hdf5 0 0
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


mkdir -p ${script_dir}/../build
cd ${script_dir}/../build
cmake .. -DCMAKE_BUILD_TYPE=Release
if make -j; then
    chmod a+x csknow_compare_train_datasets
    ./csknow_compare_train_datasets ${script_dir}/../rollout_outputs/behaviorTreeTeamFeatureStore_9_10_23_durst_learned.hdf5 ${script_dir}/../all_train_outputs durst_learnedHumanTrajectorySimilarity.hdf5 0 0
    ./csknow_compare_train_datasets ${script_dir}/../rollout_outputs/behaviorTreeTeamFeatureStore_9_10_23_durst_handcrafted.hdf5 ${script_dir}/../all_train_outputs durst_handcraftedHumanTrajectorySimilarity.hdf5 0 0
    ./csknow_compare_train_datasets ${script_dir}/../rollout_outputs/behaviorTreeTeamFeatureStore_9_10_23_default.hdf5 ${script_dir}/../all_train_outputs defaultHumanTrajectorySimilarity.hdf5 0 0
fi
