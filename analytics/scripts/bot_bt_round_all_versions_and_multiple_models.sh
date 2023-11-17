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


run_csknow_rounds() {
    new_demos=()
    for i in {0..13}
    do
        cd ${script_dir}/../../learn_bot/
        ./scripts/deploy_latent_models_specific.sh $model 
        cd ${script_dir}/../build
        echo "most recent demo file before $model_type $i"
        new_demo=$(ls -tp /home/steam/csgo-ds/csgo/*.dem | grep -v /$ | head -1)
        new_demo_no_path=$(basename $new_demo)
        echo $new_demo
        new_demos+=($new_demo_no_path)
        ./csknow_test_bt_bot ${script_dir}/../nav /home/steam/csgo-ds/csgo/addons/sourcemod/bot-link-data ${script_dir}/../ ${script_dir}/../../learn_bot/models ${script_dir}/../../learn_bot/learn_bot/libs/saved_train_test_splits $bot_type $custom_bots n $i
        sleep 40
        date
    done

    type_str="${model_type} demos for $model:"
    echo $type_str
    result_strs+=("$type_str")
    old_ifs=$IFS
    export IFS=,
    new_demos_str="${new_demos[*]}"
    echo $new_demos_str
    export IFS=$old_ifs
    result_strs+=("$new_demos_str")
    result_strs+=("")
}


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

    # learned bots
    model_type="learned"
    result_strs=()
    models=(11_15_2023__15_31_15_iw_256_bc_20_pr_0_fr_0_b_1024_it_1_ot_3_lr_4e-05_wd_0.0_l_4_h_4_n_20.0_ros_2.0_ct_SimilarityControl_pm_NoMask_nm_False_om_NoMask_w_None_dh_None_c_just_human_all)
    for model in "${models[@]}"
    do
        model_type="learned"
        bot_type=r
        custom_bots=1
        run_csknow_rounds
    done

    # hand-crafted bots
    model_type="hand-crafted"
    bot_type=rh
    custom_bots=1
    run_csknow_rounds

    # default bots
    model_type="default"
    bot_type=rh
    custom_bots=0
    run_csknow_rounds

    old_ifs=$IFS
    export IFS=$'\n'
    echo "${result_strs[*]}"
    export IFS=$old_ifs
fi