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


mkdir -p ${script_dir}/../build
cd ${script_dir}/../build
cmake .. -DCMAKE_BUILD_TYPE=Release
if make -j 8; then
    date
    echo 'most recent demo file before test run'
    ls -tp /home/steam/csgo-ds/csgo/*.dem | grep -v /$ | head -1
    /home/steam/csgo-ds/csgo/addons/sourcemod/scripting/bot-link/end_game.sh
    sleep 40
    date

    models=(11_15_2023__15_31_15_iw_256_bc_20_pr_0_fr_0_b_1024_it_1_ot_3_lr_4e-05_wd_0.0_l_4_h_4_n_20.0_ros_2.0_ct_SimilarityControl_pm_NoMask_nm_False_om_NoMask_w_None_dh_None_c_just_human_all)
    for model in "${models[@]}"
    do
        # learned no mask with position randomization
        learned_demos=()
        for i in {9..9}
        do
            cd ${script_dir}/../../learn_bot/
            ./scripts/deploy_latent_models_specific.sh $model 
            cd ${script_dir}/../build
            echo "most recent demo file before learned $i no mask run with position randomization"
            learned_demo=$(ls -tp /home/steam/csgo-ds/csgo/*.dem | grep -v /$ | head -1)
            learned_demo_no_path=$(basename $learned_demo)
            echo $learned_demo
            learned_demos+=($learned_demo_no_path)
            ./csknow_test_bt_bot ${script_dir}/../nav /home/steam/csgo-ds/csgo/addons/sourcemod/bot-link-data ${script_dir}/../ ${script_dir}/../../learn_bot/models ${script_dir}/../../learn_bot/learn_bot/libs/saved_train_test_splits r 1 br $i
            sleep 40
            date
        done

        echo "learned demos for $model:"
        old_ifs=$IFS
        export IFS=,
        echo "${learned_demos[*]}"
        export IFS=$old_ifs
    done
fi
