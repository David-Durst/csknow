dir_path=../local_data

for name in players rounds ticks player_at_tick spotted weapon_fire kills hurt grenades flashed grenade_trajectories plants defusals explosions visibilities visibilities_unadjusted actions
do
    touch ${dir_path}/${name}.csv
    awk 'FNR > 1' ${dir_path}/${name}/*.csv > ${dir_path}/${name}.csv
done

#for name in ticks player_at_tick spotted weapon_fire kills hurt grenades flashed grenade_trajectories plants defusals explosions
#do
#    for f in ${dir_path}/${name}/*.csv
#    do
#        echo "loading ${f}$"
#        cpimport -s , csknow ${name} ${f}
#    done
#done
echo "done loading"
#defusals  explosions flashed  grenade_trajectories  grenades  hurt  kills  plants  player_at_tick  players  spotted  ticks  weapon_fire
