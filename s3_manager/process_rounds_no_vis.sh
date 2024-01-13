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

# create tick data
get_script_dir
export IFS=","
i=0
for demo in $1; do
    python disable.py rollout
    # try both servers
    scp steam@54.219.195.192:csgo-ds/csgo/$demo $2_$i.dem
    scp steam@54.215.94.224:csgo-ds/csgo/$demo $2_$i.dem
    python upload.py rollout $2_$i.dem
    cd ../demo_parser
    go run cmd/main.go -ro -dn _$2_$i
    cd ../analytics
    ./scripts/create_rollout_other_datasets.sh $2_$i
    cd ../s3_manager
    ((i+=1))
done
