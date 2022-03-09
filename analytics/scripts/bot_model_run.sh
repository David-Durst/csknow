trap "jobs -p | xargs -r kill" EXIT
trap "jobs -p | xargs -r kill" SIGINT

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

export PYTHONPATH=${PYTHONPATH}:${script_dir}/../../learn_bot

source $(dirname $(which conda))/../etc/profile.d/conda.sh
conda activate learn_bot

mkdir -p ${script_dir}/../build
cd ${script_dir}/../build
cmake .. -DCMAKE_BUILD_TYPE=Release
if make -j4; then
    chmod a+x csknow_bot
    python -m learn_bot.inference /home/steam/csgo-ds/csgo/addons/sourcemod/bot-link-data &
    ${script_dir}/../build/csknow_bot /home/steam/csgo-ds/csgo/maps /home/steam/csgo-ds/csgo/addons/sourcemod/bot-link-data true /home/steam/csknow/learn_bot/data
fi
