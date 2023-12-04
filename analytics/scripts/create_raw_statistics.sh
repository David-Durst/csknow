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

echo "data path,num ticks,num shots,num kills" > ${script_dir}/../all_train_outputs/all_statistics.csv

mkdir -p ${script_dir}/../build
cd ${script_dir}/../build
cmake .. -DCMAKE_BUILD_TYPE=Release
if make -j; then
    chmod a+x csknow_create_raw_statistics
    for f in ${script_dir}/../../demo_parser/hdf5/all_train_data/*.hdf5
    do 
        echo Processing $f
        f_appendix=$(echo $f | sed -n 's/^.*\/\([[:digit:]]\+\).hdf5/\1/'p)
        ${script_dir}/../build/csknow_create_raw_statistics $f ${script_dir}/../all_train_outputs
    done
fi
