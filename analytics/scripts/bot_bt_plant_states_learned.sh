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
if make -j4; then
    chmod a+x csknow_bt_bot
    ${script_dir}/../build/csknow_bt_bot /home/steam/csknow/analytics/nav /home/steam/csgo-ds/csgo/addons/sourcemod/bot-link-data /home/steam/csknow/analytics /home/steam/csknow/learn_bot/models /home/steam/csknow/learn_bot/learn_bot/libs/saved_train_test_splits r 1
fi
