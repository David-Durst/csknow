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
if make -j 18; then
    date
    echo 'most recent demo file before test run'
    ls -tp /home/steam/csgo-ds/csgo/*.dem | grep -v /$ | head -1
    /home/steam/csgo-ds/csgo/addons/sourcemod/scripting/bot-link/end_game.sh
    sleep 40
    date
    # learned
    echo 'most recent demo file before learned run'
    ls -tp /home/steam/csgo-ds/csgo/*.dem | grep -v /$ | head -1
    ./csknow_test_bt_bot ${script_dir}/../nav /home/steam/csgo-ds/csgo/addons/sourcemod/bot-link-data ${script_dir}/../ ${script_dir}/../../learn_bot/models ${script_dir}/../../learn_bot/learn_bot/libs/saved_train_test_splits r 1 n
    sleep 40
    date
    # hand crafted
    echo 'most recent demo file before hand-crafted run'
    ls -tp /home/steam/csgo-ds/csgo/*.dem | grep -v /$ | head -1
    ./csknow_test_bt_bot ${script_dir}/../nav /home/steam/csgo-ds/csgo/addons/sourcemod/bot-link-data ${script_dir}/../ ${script_dir}/../../learn_bot/models ${script_dir}/../../learn_bot/learn_bot/libs/saved_train_test_splits rh 1 n
    sleep 40
    date
    # default
    echo 'most recent demo file before default run'
    ls -tp /home/steam/csgo-ds/csgo/*.dem | grep -v /$ | head -1
    ./csknow_test_bt_bot ${script_dir}/../nav /home/steam/csgo-ds/csgo/addons/sourcemod/bot-link-data ${script_dir}/../ ${script_dir}/../../learn_bot/models ${script_dir}/../../learn_bot/learn_bot/libs/saved_train_test_splits rh 0 n
    date
fi
