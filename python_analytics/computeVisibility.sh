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
# 9_20_21 hacks
python computeVisibility.py ${script_dir}/videos/9_20_21_no_hacks/merrick_9_20_21_no_hacks.dem.mp4 ${script_dir}/../local_data/visibilities/ ${script_dir}/visibilityLogs/ Kcirrem "i_eat_short_people_for_breakfast,,meeseeks,,Step Papi," 0 merrick_kingston_matt_gabe_rory_durst_9_20_21_no_hacks.dem ${pass}
python computeVisibility.py ${script_dir}/videos/9_20_21_no_hacks/kingston_9_20_21_no_hacks.dem.mp4 ${script_dir}/../local_data/visibilities/ ${script_dir}/visibilityLogs/ "Malia Obama" "i_eat_short_people_for_breakfast,,meeseeks,,Step Papi," 0 merrick_kingston_matt_gabe_rory_durst_9_20_21_no_hacks.dem ${pass}
python computeVisibility.py ${script_dir}/videos/9_20_21_no_hacks/rory_9_20_21_no_hacks.dem.mp4 ${script_dir}/../local_data/visibilities/ ${script_dir}/visibilityLogs/ Pinkay "i_eat_short_people_for_breakfast,,meeseeks,,Step Papi," 0 merrick_kingston_matt_gabe_rory_durst_9_20_21_no_hacks.dem ${pass}

python computeVisibility.py ${script_dir}/videos/9_20_21_no_hacks/durst_9_20_21_no_hacks.dem.mp4 ${script_dir}/../local_data/visibilities/ ${script_dir}/visibilityLogs/ i_eat_short_people_for_breakfast "Malia Obama,,Kcirrem,,Pinkay,Tom" 0 merrick_kingston_matt_gabe_rory_durst_9_20_21_no_hacks.dem ${pass}
python computeVisibility.py ${script_dir}/videos/9_20_21_no_hacks/gabe_9_20_21_no_hacks.dem.mp4 ${script_dir}/../local_data/visibilities/ ${script_dir}/visibilityLogs/ meeseeks "Malia Obama,,Kcirrem,,Pinkay,Tom" 0 merrick_kingston_matt_gabe_rory_durst_9_20_21_no_hacks.dem ${pass}
python computeVisibility.py ${script_dir}/videos/9_20_21_no_hacks/matt_9_20_21_no_hacks.dem.mp4 ${script_dir}/../local_data/visibilities/ ${script_dir}/visibilityLogs/ "Step Papi" "Malia Obama,,Kcirrem,,Pinkay,Tom" 0 merrick_kingston_matt_gabe_rory_durst_9_20_21_no_hacks.dem ${pass}


# 9_20_21 no hacks
python computeVisibility.py ${script_dir}/videos/9_20_21_wallhacks_2/merrick_9_20_21_wallhacks_2.dem.mp4 ${script_dir}/../local_data/visibilities/ ${script_dir}/visibilityLogs/ i_eat_short_people_for_breakfast "i_eat_short_people_for_breakfast,,meeseeks,,Step Papi,Alfred" 1 merrick_kingston_matt_gabe_rory_durst_9_20_21_wallhacks_2.dem ${pass}
python computeVisibility.py ${script_dir}/videos/9_20_21_wallhacks_2/kingston_9_20_21_wallhacks_2.dem.mp4 ${script_dir}/../local_data/visibilities/ ${script_dir}/visibilityLogs/ "Malia Obama" "i_eat_short_people_for_breakfast,,meeseeks,,Step Papi,Alfred" 1 merrick_kingston_matt_gabe_rory_durst_9_20_21_wallhacks_2.dem ${pass}
python computeVisibility.py ${script_dir}/videos/9_20_21_wallhacks_2/rory_9_20_21_wallhacks_2.dem.mp4 ${script_dir}/../local_data/visibilities/ ${script_dir}/visibilityLogs/ Pinkay "i_eat_short_people_for_breakfast,,meeseeks,,Step Papi,Alfred" 1 merrick_kingston_matt_gabe_rory_durst_9_20_21_wallhacks_2.dem ${pass}

python computeVisibility.py ${script_dir}/videos/9_20_21_wallhacks_2/durst_9_20_21_wallhacks_2.dem.mp4 ${script_dir}/../local_data/visibilities/ ${script_dir}/visibilityLogs/ i_eat_short_people_for_breakfast "Malia Obama,,Kcirrem,,Pinkay," 1 merrick_kingston_matt_gabe_rory_durst_9_20_21_wallhacks_2.dem ${pass}
python computeVisibility.py ${script_dir}/videos/9_20_21_wallhacks_2/gabe_9_20_21_wallhacks_2.dem.mp4 ${script_dir}/../local_data/visibilities/ ${script_dir}/visibilityLogs/ meeseeks "Malia Obama,,Kcirrem,,Pinkay," 1 merrick_kingston_matt_gabe_rory_durst_9_20_21_wallhacks_2.dem ${pass}
python computeVisibility.py ${script_dir}/videos/9_20_21_wallhacks_2/matt_9_20_21_wallhacks_2.dem.mp4 ${script_dir}/../local_data/visibilities/ ${script_dir}/visibilityLogs/ "Step Papi" "Malia Obama,,Kcirrem,,Pinkay," 1 merrick_kingston_matt_gabe_rory_durst_9_20_21_wallhacks_2.dem ${pass}

# 9_19_21 hacks
python computeVisibility.py ${script_dir}/videos/9_19_21_walls/durst_9_19_21_walls.dem.mp4 ${script_dir}/../local_data/visibilities/ ${script_dir}/visibilityLogs/ i_eat_short_people_for_breakfast "Michelle Obama,,Gary,,Chris,Martin" 1 9_19_21_durst_t_wang_ct_with_walls.dem ${pass}
python computeVisibility.py ${script_dir}/videos/9_19_21_walls/wang_9_19_21_walls.dem.mp4 ${script_dir}/../local_data/visibilities/ ${script_dir}/visibilityLogs/ "Michelle Obama" "i_eat_short_people_for_breakfast,,Henry,,Adam," 1 9_19_21_durst_t_wang_ct_with_walls.dem ${pass}
python computeVisibility.py ${script_dir}/10_4_21_shortened_merrick.mp4 ${script_dir}/../local_data/visibilities/ ${script_dir}/visibilityLogs/ Kcirrem "i_eat_short_people_for_breakfast,,meeseeks,,Step Papi,Tom" 0 merrick_kingston_matt_gabe_rory_durst_9_20_21_no_hacks.dem ${pass}
