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

cd ${script_dir}/csknow-python-analytics/visibility/

#python computeVisibility.py ${script_dir}/10_4_21_no_wallhacks_merrick.mp4 ${script_dir}/../local_data/visibilities/ ${script_dir}/visibilityLogs/ Kcirrem "i_eat_short_people_for_breakfast,,meeseeks,,Step Papi,Tom" 0 merrick_kingston_matt_gabe_rory_durst_9_20_21_no_hacks.dem ${pass}
python computeVisibility.py ${script_dir}/10_4_21_shortened_merrick.mp4 ${script_dir}/../local_data/visibilities/ ${script_dir}/visibilityLogs/ Kcirrem "i_eat_short_people_for_breakfast,,meeseeks,,Step Papi,Tom" 0 merrick_kingston_matt_gabe_rory_durst_9_20_21_no_hacks.dem ${pass}
