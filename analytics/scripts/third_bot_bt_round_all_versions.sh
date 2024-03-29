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
    for j in {0..14}
    do
        cd ${script_dir}/../../learn_bot/
        echo "model"
        echo $model 
        echo "uncertain model"
        echo $uncertain_model 
        ./scripts/deploy_latent_models_specific.sh $model 
        ./scripts/deploy_uncertain_models_specific.sh $uncertain_model 
        cd ${script_dir}/../build
        echo "most recent demo file before $model_type $j"
        new_demo=$(ls -tp /home/steam/csgo-ds/csgo/*.dem | grep -v /$ | head -1)
        new_demo_no_path=$(basename $new_demo)
        echo $new_demo
        new_demos+=($new_demo_no_path)
        ./csknow_test_bt_bot ${script_dir}/../nav /home/steam/csgo-ds/csgo/addons/sourcemod/bot-link-data ${script_dir}/../ ${script_dir}/../../learn_bot/models ${script_dir}/../../learn_bot/learn_bot/libs/saved_train_test_splits $bot_type $custom_bots n $j
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

git log -n 1
mkdir -p ${script_dir}/../build
cd ${script_dir}/../build
cmake .. -DCMAKE_BUILD_TYPE=Release
if make -j 8; then
    date
    echo 'most recent demo file before test run'
    ls -tp /home/steam/csgo-ds/csgo/*.dem | grep -v /$ | head -1
    /home/steam/csgo-ds/csgo/addons/sourcemod/scripting/bot-link/end_game.sh
    sleep 50
    date

    # learned bots
    model_type="learned"
    result_strs=()
    #models=(11_26_2023__22_28_22_iw_256_bc_20_pr_0_fr_0_b_1024_it_1_ot_3_lr_4e-05_wd_0.0_l_4_h_4_n_20.0_ros_2.0_ct_SimilarityControl_pm_NoMask_nm_False_om_NoMask_w_None_dh_None_c_just_human_all)
    #models=(01_06_2024__00_05_43_iw_256_bc_20_pr_0_fr_0_b_1024_it_1_ot_3_lr_4e-05_wd_0.0_l_4_h_4_n_20.0_ros_2.0_ct_SimilarityControl_pm_NoMask_mpi_False_om_NoMask_w_None_dh_None_c_just_human_all)
    #models=(01_15_2024__05_35_49_iw_256_bc_20_pr_0_fr_0_b_1024_it_1_ot_3_lr_4e-05_wd_0.0_l_4_h_4_n_20.0_ros_2.0_ct_SimilarityControl_pm_NoMask_mpi_False_om_NoMask_w_None_dh_None_ifo_False_c_just_human_all)
    #models=(01_18_2024__10_25_27_iw_256_bc_20_pr_0_fr_0_b_1024_it_1_ot_3_lr_4e-05_wd_0.0_l_4_h_4_n_20.0_ros_2.0_ct_TimeControl_pm_NoMask_mpi_False_om_NoMask_w_None_dh_None_ifo_False_c_just_human_all)
    #models=(01_15_2024__05_35_49_iw_256_bc_40_pr_0_fr_0_b_1024_it_1_ot_3_lr_4e-05_wd_0.0_l_4_h_4_n_20.0_ros_2.0_ct_SimilarityControl_pm_EveryoneMask_mpi_False_om_NoMask_w_None_dh_None_ifo_False_c_just_human_all)
    #models=(01_16_2024__10_18_32_iw_256_bc_40_pr_0_fr_0_b_1024_it_1_ot_3_lr_4e-05_wd_0.0_l_4_h_4_n_20.0_ros_2.0_ct_SimilarityControl_pm_EveryoneMask_mpi_True_om_NoMask_w_None_dh_None_ifo_False_c_just_human_all \
    #    01_16_2024__10_18_32_iw_256_bc_40_pr_0_fr_0_b_1024_it_1_ot_3_lr_4e-05_wd_0.0_l_4_h_4_n_20.0_ros_2.0_ct_SimilarityControl_pm_EveryoneMask_mpi_True_om_NoMask_w_None_dh_None_ifo_False_c_just_human_all)
    models=(01_15_2024__05_35_49_iw_256_bc_20_pr_0_fr_0_b_1024_it_1_ot_3_lr_4e-05_wd_0.0_l_4_h_4_n_20.0_ros_2.0_ct_SimilarityControl_pm_NoMask_mpi_False_om_NoMask_w_None_dh_None_ifo_False_c_just_human_all \
        01_15_2024__05_35_49_iw_256_bc_20_pr_0_fr_0_b_1024_it_1_ot_3_lr_4e-05_wd_0.0_l_4_h_4_n_20.0_ros_2.0_ct_SimilarityControl_pm_NoMask_mpi_False_om_NoMask_w_None_dh_None_ifo_False_c_just_human_all)
    uncertain_models=(03_25_2024__12_22_32_iw_256_bc_20_pr_0_fr_0_b_1024_it_1_ot_3_lr_4e-05_wd_0.0_l_4_h_4_n_20.0_ros_2.0_ct_SimilarityControl_pm_NoMask_mpi_False_om_NoMask_w_None_ws_50.0_wns_0.0_dh_None_ifo_False_c_just_human_all \
        03_25_2024__12_22_32_iw_256_bc_20_pr_0_fr_0_b_1024_it_1_ot_3_lr_4e-05_wd_0.0_l_4_h_4_n_20.0_ros_2.0_ct_SimilarityControl_pm_NoMask_mpi_False_om_NoMask_w_None_ws_50.0_wns_0.0_dh_None_ifo_False_c_just_human_all)
    #models=(01_06_2024__00_05_43_iw_256_bc_20_pr_0_fr_0_b_1024_it_1_ot_3_lr_4e-05_wd_0.0_l_4_h_4_n_20.0_ros_2.0_ct_NoControl_pm_NoMask_mpi_False_om_NoMask_w_None_dh_None_c_just_human_all)
    #uncertain_models=(12_29_2023__22_46_01_iw_256_bc_20_pr_0_fr_0_b_1024_it_1_ot_3_lr_4e-05_wd_0.0_l_4_h_4_n_20.0_ros_2.0_ct_TimeControl_pm_NoMask_mpi_False_om_NoMask_w_None_dh_None_c_just_human_all)
    #models=(12_25_2023__21_22_14_iw_256_bc_20_pr_0_fr_0_b_1024_it_1_ot_3_lr_4e-05_wd_0.0_l_4_h_4_n_20.0_ros_2.0_ct_SimilarityControl_pm_NoMask_mpi_False_om_NoMask_w_None_dh_None_c_just_human_all)
    #models=(12_28_2023__17_14_50_iw_256_bc_20_pr_0_fr_0_b_1024_it_1_ot_3_lr_4e-05_wd_0.0_l_4_h_4_n_20.0_ros_2.0_ct_SimilarityControl_pm_NoMask_mpi_False_om_NoMask_w_None_dh_None_c_just_human_all)
    #models=(12_28_2023__17_14_50_iw_256_bc_20_pr_0_fr_0_b_1024_it_1_ot_3_lr_4e-05_wd_0.0_l_4_h_4_n_20.0_ros_2.0_ct_SimilarityControl_pm_NoMask_mpi_False_om_NoMask_w_None_dh_None_c_just_human_all \
    #    12_28_2023__17_14_50_iw_256_bc_20_pr_0_fr_0_b_1024_it_3_ot_3_lr_4e-05_wd_0.0_l_4_h_4_n_20.0_ros_2.0_ct_SimilarityControl_pm_NoMask_mpi_False_om_NoMask_w_None_dh_None_c_just_human_all)
    for (( i=0; i<${#models[*]}; i++ ))
    do
        model="${models[$i]}"
        uncertain_model="${uncertain_models[$i]}"

        /home/steam/csgo-ds/csgo/addons/sourcemod/scripting/bot-link/make_push.sh
        model_type="learned push"
        bot_type=r
        custom_bots=1
        run_csknow_rounds

        #/home/steam/csgo-ds/csgo/addons/sourcemod/scripting/bot-link/make_save.sh
        #model_type="learned save"
        #bot_type=r
        #custom_bots=1
        #run_csknow_rounds
    done

    ## hand-crafted bots
    #model_type="hand-crafted"
    #bot_type=rh
    #custom_bots=1
    #run_csknow_rounds

    ## default bots
    #model_type="default"
    #bot_type=rh
    #custom_bots=0
    #run_csknow_rounds

    old_ifs=$IFS
    export IFS=$'\n'
    echo "${result_strs[*]}"
    export IFS=$old_ifs
fi
