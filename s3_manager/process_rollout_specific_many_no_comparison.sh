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
    scp steam@54.219.195.192:csgo-ds/csgo/$demo $2_$i.dem
    python upload.py rollout $2_$i.dem
    python list.py rollout
    cd ../demo_parser
    go run cmd/main.go -ro -dn _$2
    cd ../analytics
    ./scripts/create_rollout_other_datasets.sh $2
    cd ../learn_bot
    python -m learn_bot.latent.analyze.compare_trajectories.plot_trajectories_independently 10_25_2023__01_11_56_iw_128_bc_35_pr_0_fr_0_b_1024_it_5_lr_4e-05_wd_0.0_l_2_h_4_n_20.0_ros_2.0_pm_NoMask_nm_False_om_EngagementMask_w_None_dh_None_c_just_human_all 0 _$2
    cd ../s3_manager
    ((i+=1))
done
