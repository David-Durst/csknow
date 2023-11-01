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

    # learned no mask offense with position randomization
    cd ${script_dir}/../../learn_bot/
    ./scripts/deploy_latent_models_specific.sh 10_17_2023__15_07_01_iw_128_bc_25_pr_0_fr_0_b_1024_it_1_lr_4e-05_wd_0.0_l_2_h_4_n_20.0_ros_2.0_m_NoMask_w_None_dh_None_c_just_human_all
    cd ${script_dir}/../build
    echo 'most recent demo file before learned no mask offense run with position randomization'
    offense_learned_demo=$(ls -tp /home/steam/csgo-ds/csgo/*.dem | grep -v /$ | head -1)
    echo $offense_learned_demo
    ./csknow_test_bt_bot ${script_dir}/../nav /home/steam/csgo-ds/csgo/addons/sourcemod/bot-link-data ${script_dir}/../ ${script_dir}/../../learn_bot/models ${script_dir}/../../learn_bot/learn_bot/libs/saved_train_test_splits r 1 bro
    sleep 40
    date

    # learned no mask defense a with position randomization
    cd ${script_dir}/../../learn_bot/
    ./scripts/deploy_latent_models_specific.sh 10_17_2023__15_07_01_iw_128_bc_25_pr_0_fr_0_b_1024_it_1_lr_4e-05_wd_0.0_l_2_h_4_n_20.0_ros_2.0_m_NoMask_w_None_dh_None_c_just_human_all
    cd ${script_dir}/../build
    echo 'most recent demo file before learned no mask defense a run with position randomization'
    defense_a_learned_demo=$(ls -tp /home/steam/csgo-ds/csgo/*.dem | grep -v /$ | head -1)
    echo $defense_a_learned_demo
    ./csknow_test_bt_bot ${script_dir}/../nav /home/steam/csgo-ds/csgo/addons/sourcemod/bot-link-data ${script_dir}/../ ${script_dir}/../../learn_bot/models ${script_dir}/../../learn_bot/learn_bot/libs/saved_train_test_splits r 1 brda
    sleep 40
    date

    # learned no mask defense b with position randomization
    cd ${script_dir}/../../learn_bot/
    ./scripts/deploy_latent_models_specific.sh 10_17_2023__15_07_01_iw_128_bc_25_pr_0_fr_0_b_1024_it_1_lr_4e-05_wd_0.0_l_2_h_4_n_20.0_ros_2.0_m_NoMask_w_None_dh_None_c_just_human_all
    cd ${script_dir}/../build
    echo 'most recent demo file before learned no mask defense b run with position randomization'
    defense_b_learned_demo=$(ls -tp /home/steam/csgo-ds/csgo/*.dem | grep -v /$ | head -1)
    echo $defense_b_learned_demo
    ./csknow_test_bt_bot ${script_dir}/../nav /home/steam/csgo-ds/csgo/addons/sourcemod/bot-link-data ${script_dir}/../ ${script_dir}/../../learn_bot/models ${script_dir}/../../learn_bot/learn_bot/libs/saved_train_test_splits r 1 brdb
    sleep 40
    date
fi
