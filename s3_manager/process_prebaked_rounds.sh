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
    cd ../demo_parser
    go run cmd/main.go -ro -dn _$2_$i
    cd ../analytics
    ./scripts/create_rollout_other_datasets.sh $2_$i
    cd ../learn_bot
    python -m learn_bot.latent.analyze.compare_trajectories.plot_trajectories_independently 01_11_2024__00_46_41_iw_256_bc_20_pr_0_fr_0_b_1024_it_1_ot_3_lr_4e-05_wd_0.0_l_4_h_4_n_20.0_ros_2.0_ct_SimilarityControl_pm_NoMask_mpi_True_om_NoMask_w_None_dh_None_ifo_False_c_just_human_all 0 _$2_$i _$2
    cd ../s3_manager
    ((i+=1))
done
