no_mask_demos=(auto0-20240113-080140-565444521-de_dust2-international_house_of_short_people_east_franchise.dem \
    auto0-20240113-084924-1138527264-de_dust2-international_house_of_short_people_east_franchise.dem \
    #auto0-20240113-093956-286563230-de_dust2-international_house_of_short_people_east_franchise.dem \
    auto0-20240113-103025-1428504254-de_dust2-international_house_of_short_people_east_franchise.dem)
everyone_mask_demos=(auto0-20240113-111606-545576288-de_dust2-international_house_of_short_people_east_franchise.dem \
    auto0-20240113-121013-197926321-de_dust2-international_house_of_short_people_east_franchise.dem \
    #auto0-20240113-130351-1391499459-de_dust2-international_house_of_short_people_east_franchise.dem \
    auto0-20240113-135635-1731730609-de_dust2-international_house_of_short_people_east_franchise.dem)
# third no mask demo got corrupted

run_prefix=1_12_24

for (( i=2; i<${#no_mask_demos[*]}; i++ ))
do
    no_mask_demo="${no_mask_demos[$i]}"
    everyone_mask_demo="${everyone_mask_demos[$i]}"
    no_mask_hdf5=${run_prefix}_no_mask_demo_$i
    everyone_mask_hdf5=${run_prefix}_everyone_mask_demo_$i
    ./process_rounds_no_vis.sh $no_mask_demo $no_mask_hdf5
    ./process_rounds_no_vis.sh $everyone_mask_demo $everyone_mask_hdf5
    cd ../learn_bot
    echo ${no_mask_hdf5},${everyone_mask_hdf5}
    python -m learn_bot.latent.analyze.plot_trajectory_heatmap.plot_trajectory_heatmap 01_11_2024__00_46_41_iw_256_bc_20_pr_0_fr_0_b_1024_it_1_ot_3_lr_4e-05_wd_0.0_l_4_h_4_n_20.0_ros_2.0_ct_SimilarityControl_pm_NoMask_mpi_False_om_NoMask_w_None_dh_None_ifo_False_c_just_human_all ${no_mask_hdf5},${everyone_mask_hdf5} no_diff
    cd -
done
