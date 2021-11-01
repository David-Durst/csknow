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
# adjusted
# 9_20_21 no hacks
#python computeVisibility.py ${script_dir}/videos/9_20_21_no_hacks_gotv_adjusted/9_20_21_no_wallhacks_merrick_ct.mp4 ${script_dir}/../local_data/visibilities/ ${script_dir}/visibilityLogs/ Kcirrem "i_eat_short_people_for_breakfast,,meeseeks,,Step Papi," 0 merrick_kingston_matt_gabe_rory_durst_9_20_21_no_hacks.dem ${pass}
#python computeVisibility.py ${script_dir}/videos/9_20_21_no_hacks_gotv_adjusted/9_20_21_no_wallhacks_malia_ct.mp4 ${script_dir}/../local_data/visibilities/ ${script_dir}/visibilityLogs/ "Malia Obama" "i_eat_short_people_for_breakfast,,meeseeks,,Step Papi," 0 merrick_kingston_matt_gabe_rory_durst_9_20_21_no_hacks.dem ${pass}
#python computeVisibility.py ${script_dir}/videos/9_20_21_no_hacks_gotv_adjusted/9_20_21_no_wallhacks_pinkay_ct.mp4 ${script_dir}/../local_data/visibilities/ ${script_dir}/visibilityLogs/ Pinkay "i_eat_short_people_for_breakfast,,meeseeks,,Step Papi," 0 merrick_kingston_matt_gabe_rory_durst_9_20_21_no_hacks.dem ${pass}
##
#python computeVisibility.py ${script_dir}/videos/9_20_21_no_hacks_gotv_adjusted/9_20_21_no_wallhacks_durst_t.mp4 ${script_dir}/../local_data/visibilities/ ${script_dir}/visibilityLogs/ i_eat_short_people_for_breakfast "Malia Obama,,Kcirrem,,Pinkay,Tom" 0 merrick_kingston_matt_gabe_rory_durst_9_20_21_no_hacks.dem ${pass}
#python computeVisibility.py ${script_dir}/videos/9_20_21_no_hacks_gotv_adjusted/9_20_21_no_wallhacks_meeseeks_t.mp4 ${script_dir}/../local_data/visibilities/ ${script_dir}/visibilityLogs/ meeseeks "Malia Obama,,Kcirrem,,Pinkay,Tom" 0 merrick_kingston_matt_gabe_rory_durst_9_20_21_no_hacks.dem ${pass}
#python computeVisibility.py ${script_dir}/videos/9_20_21_no_hacks_gotv_adjusted/9_20_21_no_wallhacks_step_papi_t.mp4 ${script_dir}/../local_data/visibilities/ ${script_dir}/visibilityLogs/ "Step Papi" "Malia Obama,,Kcirrem,,Pinkay,Tom" 0 merrick_kingston_matt_gabe_rory_durst_9_20_21_no_hacks.dem ${pass}
#
#
## 9_20_21 hacks
#python computeVisibility.py ${script_dir}/videos/9_20_21_wallhacks_2_gotv_adjusted/9_20_21_wallhacks_2_merrick_t.mp4 ${script_dir}/../local_data/visibilities/ ${script_dir}/visibilityLogs/ Kcirrem "i_eat_short_people_for_breakfast,,meeseeks,,Step Papi,Alfred" 1 merrick_kingston_matt_gabe_rory_durst_9_20_21_wallhacks_2.dem ${pass}
#python computeVisibility.py ${script_dir}/videos/9_20_21_wallhacks_2_gotv_adjusted/9_20_21_wallhacks_2_malia_t.mp4 ${script_dir}/../local_data/visibilities/ ${script_dir}/visibilityLogs/ "Malia Obama" "i_eat_short_people_for_breakfast,,meeseeks,,Step Papi,Alfred" 1 merrick_kingston_matt_gabe_rory_durst_9_20_21_wallhacks_2.dem ${pass}
#python computeVisibility.py ${script_dir}/videos/9_20_21_wallhacks_2_gotv_adjusted/9_20_21_wallhacks_2_pinkay_t.mp4 ${script_dir}/../local_data/visibilities/ ${script_dir}/visibilityLogs/ Pinkay "i_eat_short_people_for_breakfast,,meeseeks,,Step Papi,Alfred" 1 merrick_kingston_matt_gabe_rory_durst_9_20_21_wallhacks_2.dem ${pass}
#
#python computeVisibility.py ${script_dir}/videos/9_20_21_wallhacks_2_gotv_adjusted/9_20_21_wallhacks_2_durst_ct.mp4 ${script_dir}/../local_data/visibilities/ ${script_dir}/visibilityLogs/ i_eat_short_people_for_breakfast "Malia Obama,,Kcirrem,,Pinkay," 1 merrick_kingston_matt_gabe_rory_durst_9_20_21_wallhacks_2.dem ${pass}
#python computeVisibility.py ${script_dir}/videos/9_20_21_wallhacks_2_gotv_adjusted/9_20_21_wallhacks_2_meeseeks_ct.mp4 ${script_dir}/../local_data/visibilities/ ${script_dir}/visibilityLogs/ meeseeks "Malia Obama,,Kcirrem,,Pinkay," 1 merrick_kingston_matt_gabe_rory_durst_9_20_21_wallhacks_2.dem ${pass}
#python computeVisibility.py ${script_dir}/videos/9_20_21_wallhacks_2_gotv_adjusted/9_20_21_wallhacks_2_step_papi_ct.mp4 ${script_dir}/../local_data/visibilities/ ${script_dir}/visibilityLogs/ "Step Papi" "Malia Obama,,Kcirrem,,Pinkay," 1 merrick_kingston_matt_gabe_rory_durst_9_20_21_wallhacks_2.dem ${pass}
#
## 9_19_21 hacks
#python computeVisibility.py ${script_dir}/videos/9_19_21_walls_gotv_adjusted/9_19_21_walls_durst_t.mp4 ${script_dir}/../local_data/visibilities/ ${script_dir}/visibilityLogs/ i_eat_short_people_for_breakfast "Michelle Obama,,Gary,,Chris,Martin" 1 9_19_21_durst_t_wang_ct_with_walls.dem ${pass}
#python computeVisibility.py ${script_dir}/videos/9_19_21_walls_gotv_adjusted/9_19_21_walls_wang_ct.mp4 ${script_dir}/../local_data/visibilities/ ${script_dir}/visibilityLogs/ "Michelle Obama" "i_eat_short_people_for_breakfast,,Henry,,Adam," 1 9_19_21_durst_t_wang_ct_with_walls.dem ${pass}

# unadjusted
# 9_20_21 no hacks
#python computeVisibility.py ${script_dir}/videos/9_20_21_no_hacks_gotv_unadjusted/merrick_9_20_21_no_hacks.dem.mp4 ${script_dir}/../local_data/visibilities_unadjusted/ ${script_dir}/visibilityLogs/ Kcirrem "i_eat_short_people_for_breakfast,,meeseeks,,Step Papi," 0 merrick_kingston_matt_gabe_rory_durst_9_20_21_no_hacks.dem ${pass}
#python computeVisibility.py ${script_dir}/videos/9_20_21_no_hacks_gotv_unadjusted/kingston_9_20_21_no_hacks.dem.mp4 ${script_dir}/../local_data/visibilities_unadjusted/ ${script_dir}/visibilityLogs/ "Malia Obama" "i_eat_short_people_for_breakfast,,meeseeks,,Step Papi," 0 merrick_kingston_matt_gabe_rory_durst_9_20_21_no_hacks.dem ${pass}
#python computeVisibility.py ${script_dir}/videos/9_20_21_no_hacks_gotv_unadjusted/rory_9_20_21_no_hacks.dem.mp4 ${script_dir}/../local_data/visibilities_unadjusted/ ${script_dir}/visibilityLogs/ Pinkay "i_eat_short_people_for_breakfast,,meeseeks,,Step Papi," 0 merrick_kingston_matt_gabe_rory_durst_9_20_21_no_hacks.dem ${pass}
##
#python computeVisibility.py ${script_dir}/videos/9_20_21_no_hacks_gotv_unadjusted/durst_9_20_21_no_hacks.dem.mp4 ${script_dir}/../local_data/visibilities_unadjusted/ ${script_dir}/visibilityLogs/ i_eat_short_people_for_breakfast "Malia Obama,,Kcirrem,,Pinkay,Tom" 0 merrick_kingston_matt_gabe_rory_durst_9_20_21_no_hacks.dem ${pass}
#python computeVisibility.py ${script_dir}/videos/9_20_21_no_hacks_gotv_unadjusted/gabe_9_20_21_no_hacks.dem.mp4 ${script_dir}/../local_data/visibilities_unadjusted/ ${script_dir}/visibilityLogs/ meeseeks "Malia Obama,,Kcirrem,,Pinkay,Tom" 0 merrick_kingston_matt_gabe_rory_durst_9_20_21_no_hacks.dem ${pass}
#python computeVisibility.py ${script_dir}/videos/9_20_21_no_hacks_gotv_unadjusted/matt_9_20_21_no_hacks.dem.mp4 ${script_dir}/../local_data/visibilities_unadjusted/ ${script_dir}/visibilityLogs/ "Step Papi" "Malia Obama,,Kcirrem,,Pinkay,Tom" 0 merrick_kingston_matt_gabe_rory_durst_9_20_21_no_hacks.dem ${pass}
#
#
## 9_20_21 hacks
#python computeVisibility.py ${script_dir}/videos/9_20_21_wallhacks_2_gotv_unadjusted/merrick_9_20_21_wallhacks_2.dem.mp4 ${script_dir}/../local_data/visibilities_unadjusted/ ${script_dir}/visibilityLogs/ Kcirrem "i_eat_short_people_for_breakfast,,meeseeks,,Step Papi,Alfred" 1 merrick_kingston_matt_gabe_rory_durst_9_20_21_wallhacks_2.dem ${pass}
#python computeVisibility.py ${script_dir}/videos/9_20_21_wallhacks_2_gotv_unadjusted/kingston_9_20_21_wallhacks_2.dem.mp4 ${script_dir}/../local_data/visibilities_unadjusted/ ${script_dir}/visibilityLogs/ "Malia Obama" "i_eat_short_people_for_breakfast,,meeseeks,,Step Papi,Alfred" 1 merrick_kingston_matt_gabe_rory_durst_9_20_21_wallhacks_2.dem ${pass}
#python computeVisibility.py ${script_dir}/videos/9_20_21_wallhacks_2_gotv_unadjusted/rory_9_20_21_wallhacks_2.dem.mp4 ${script_dir}/../local_data/visibilities_unadjusted/ ${script_dir}/visibilityLogs/ Pinkay "i_eat_short_people_for_breakfast,,meeseeks,,Step Papi,Alfred" 1 merrick_kingston_matt_gabe_rory_durst_9_20_21_wallhacks_2.dem ${pass}
#
#python computeVisibility.py ${script_dir}/videos/9_20_21_wallhacks_2_gotv_unadjusted/durst_9_20_21_wallhacks_2.dem.mp4 ${script_dir}/../local_data/visibilities_unadjusted/ ${script_dir}/visibilityLogs/ i_eat_short_people_for_breakfast "Malia Obama,,Kcirrem,,Pinkay," 1 merrick_kingston_matt_gabe_rory_durst_9_20_21_wallhacks_2.dem ${pass}
#python computeVisibility.py ${script_dir}/videos/9_20_21_wallhacks_2_gotv_unadjusted/gabe_9_20_21_wallhacks_2.dem.mp4 ${script_dir}/../local_data/visibilities_unadjusted/ ${script_dir}/visibilityLogs/ meeseeks "Malia Obama,,Kcirrem,,Pinkay," 1 merrick_kingston_matt_gabe_rory_durst_9_20_21_wallhacks_2.dem ${pass}
#python computeVisibility.py ${script_dir}/videos/9_20_21_wallhacks_2_gotv_unadjusted/matt_9_20_21_wallhacks_2.dem.mp4 ${script_dir}/../local_data/visibilities_unadjusted/ ${script_dir}/visibilityLogs/ "Step Papi" "Malia Obama,,Kcirrem,,Pinkay," 1 merrick_kingston_matt_gabe_rory_durst_9_20_21_wallhacks_2.dem ${pass}
#
## 9_19_21 hacks
#python computeVisibility.py ${script_dir}/videos/9_19_21_walls_gotv_unadjusted/durst_9_19_21_walls.dem.mp4 ${script_dir}/../local_data/visibilities_unadjusted/ ${script_dir}/visibilityLogs/ i_eat_short_people_for_breakfast "Michelle Obama,,Gary,,Chris,Martin" 1 9_19_21_durst_t_wang_ct_with_walls.dem ${pass}
#python computeVisibility.py ${script_dir}/videos/9_19_21_walls_gotv_unadjusted/wang_9_19_21_walls.dem.mp4 ${script_dir}/../local_data/visibilities_unadjusted/ ${script_dir}/visibilityLogs/ "Michelle Obama" "i_eat_short_people_for_breakfast,,Henry,,Adam," 1 9_19_21_durst_t_wang_ct_with_walls.dem ${pass}

#python computeVisibility.py ${script_dir}/10_4_21_shortened_merrick.mp4 ${script_dir}/../local_data/visibilities/ ${script_dir}/visibilityLogs/ Kcirrem "i_eat_short_people_for_breakfast,,meeseeks,,Step Papi,Tom" 0 merrick_kingston_matt_gabe_rory_durst_9_20_21_no_hacks.dem ${pass}

source ${script_dir}/computeVisibilityGenerated.sh
