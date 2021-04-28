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

pass=$(cat ${script_dir}/../../private/.mysql_password)

docker run --name durst_cstore_loader \
    --rm -it \
    --mount type=bind,source="$(pwd)"/../local_data,target=/local_data \
    --entrypoint /bin/bash \
    -p 127.0.0.1:3306:3306 \
    -e MYSQL_ROOT_PASSWORD=${pass} \
    durst/cstore_loader:0.1

