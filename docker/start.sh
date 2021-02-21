mkdir -p ../demos

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


aws_access_key_id=$(cat ${script_dir}/../private/.aws_access_key_id)
aws_secret_access_key=$(cat ${script_dir}/../private/.aws_secret_access_key)

docker run --name durst_csgo \
    --rm --net=host \
    -e SRCDS_MAPGROUP=mg_de_dust2 -e SRCDS_STARTMAP=de_dust2 \
    -e AWS_ACCESS_KEY_ID=${aws_access_key_id} -e AWS_SECRET_ACCESS_KEY=${aws_secret_access_key} \
    --mount type=bind,source="$(pwd)"/../demos,target=/home/steam/demos \
    durst/csgo:0.1

