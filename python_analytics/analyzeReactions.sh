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

export pass=$(cat ${script_dir}/../private/.mysql_password)

source $(dirname $(which conda))/../etc/profile.d/conda.sh
conda activate csknow

cd ${script_dir}/csknow-python-analytics/reactions/

python individual.py ${pass} ${script_dir}/../sql_analytics/visibility.sql ${script_dir}/reactionPlots/
# change 4 to 3 for best cpu result
python grouped.py ${pass} ${script_dir}/../sql_analytics/unadjusted_visibility.sql ${script_dir}/unadjustedReactionPlots/ 3
python grouped.py ${pass} ${script_dir}/../sql_analytics/visibility.sql ${script_dir}/reactionPlots/ 3
