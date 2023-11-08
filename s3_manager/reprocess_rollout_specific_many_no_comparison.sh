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
for appendix in $2; do
    python disable.py rollout
    python enable.py rollout $1_$appendix.dem
    cd ../demo_parser
    go run cmd/main.go -ro -dn _$1_$appendix
    cd ../analytics
    ./scripts/create_rollout_other_datasets.sh $1_$appendix
    cd ../learn_bot
    python -m learn_bot.latent.analyze.compare_trajectories.plot_trajectories_independently 11_06_2023__11_18_05_iw_256_bc_20_pr_0_fr_0_b_1024_it_1_ot_3_lr_4e-05_wd_0.0_l_4_h_4_n_20.0_ros_2.0_ct_TimeControl_pm_NoMask_nm_False_om_NoMask_w_None_dh_None_c_just_human_all 0 _$1_$appendix $1
    cd ../s3_manager
    ((i+=1))
done
