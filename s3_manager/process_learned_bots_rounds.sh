hdf5_file_name=9_17_23_learned_nav_correct_300_rounds

./process_rollout_specific.sh auto0-20230918-012856-1607050854-de_dust2-international_house_of_short_people.dem $hdf5_file_name

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

analytics_script_dir=${script_dir}/../analytics/scripts/

mkdir -p ${analytics_script_dir}/../build
cd ${analytics_script_dir}/../build
cmake .. -DCMAKE_BUILD_TYPE=Release
if make -j; then
    chmod a+x csknow_compare_train_datasets
    ./csknow_compare_train_datasets ${analytics_script_dir}/../rollout_outputs/behaviorTreeTeamFeatureStore_${hdf5_file_name}.hdf5 ${analytics_script_dir}/../all_train_outputs learned_to_human_trajectorySimilarity.hdf5 0 0

    ./csknow_compare_train_datasets ${analytics_script_dir}/../all_train_outputs ${analytics_script_dir}/../rollout_outputs/behaviorTreeTeamFeatureStore_${hdf5_file_name}.hdf5 human_to_learned_trajectorySimilarity.hdf5 0 0
fi
