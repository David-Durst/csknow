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


docker build -t durst/csgo-no-update-application:0.1 -f ${script_dir}/Dockerfile_application .
#mkdir -p ${script_dir}/../analytics/build
#cd ${script_dir}/../analytics/build
#cmake .. -DCMAKE_BUILD_TYPE=Release
#chmod a+x csknow_bot
#if make -j4; then
#    mkdir -p ${script_dir}/tmp
#    cp csknow_bot ${script_dir}/tmp/
#    cp ../scripts/bot_run_no_build.sh ${script_dir}/tmp/
#    cd ${script_dir}    
#    docker build -t durst/csgo:0.3 .
#fi

