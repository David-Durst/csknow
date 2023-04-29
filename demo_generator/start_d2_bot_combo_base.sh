map=${1:-de_dust2}

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

if [ -f "${script_dir}/../.aws_credentials" ]; then
    source ${script_dir}/../.aws_credentials
fi

iam_role=$(cat ${script_dir}/../private/.aws_csgo_server_role)
gslt=$(cat ${script_dir}/../private/.gslt)
docker run --name durst_csgo_${map} \
    --rm \
    -e RUNNING_IN_EC2=1 -e ROLE=${iam_role} -e MAP=${map} -e GSLT=${gslt} \
    -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
    -e CSKNOW_BOT_STYLE=${CSKNOW_BOT_STYLE} -e CSGO_BOT_STYLE=${CSGO_BOT_STYLE} \
    -p 27015:27015/tcp -p 27015:27015/udp \
    --ulimit core=-1 \
    --tmpfs /home/steam/csgo-dedicated-non-volume/csgo/addons/sourcemod/bot-link-data \
    --mount type=bind,source=$script_dir/../analytics/external,target=/home/steam/csknow/analytics/external \
    durst/csgo:0.4

