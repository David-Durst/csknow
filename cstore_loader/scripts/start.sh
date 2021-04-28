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

export pass=$(cat ${script_dir}/../../private/.mysql_password)

${script_dir}/start_helper.sh &

docker run --name durst_cstore_loader \
    --rm \
    --mount type=bind,source="$(pwd)"/../local_data,target=/local_data \
    -p 3124:3306 \
    -e MYSQL_ROOT_PASSWORD=${pass} \
    durst/cstore_loader:0.1
