cd learn_bot/latent/analyze/similarity_plots/1_15_24_learned_push
mkdir -p ../final_results
cp diff/emd_no_filter.txt ../final_results/
cp diff/emd_only_kill.txt ../final_results/
cp diff/emd_only_kill.txt ../final_results/
cp lifetimes_no_filter.txt ../final_results/
cp lifetimes_no_filter.pdf ../final_results/
cp kills.pdf ../final_results/
cp shots_per_kill_no_filter.pdf ../final_results/
cp offense_flanks_rounds.pdf ../final_results/
cp offense_flanks_pct.txt ../final_results/
cp offense_flanks_pct_num_rounds_ticks.txt ../final_results/
cp defense_spread_rounds.pdf ../final_results/
cp defense_spread_pct.txt ../final_results/
cp defense_spread_pct_num_rounds_ticks.txt ../final_results/
cp mistakes.pdf ../final_results/regular_mistakes.pdf
cd ../mask_1_15_24_learned_push
cp mistakes.pdf ../final_results/mask_mistakes.pdf
cd ../../user_responses/user_plots
cp count_by_rank.pdf ../../similarity_plots/final_results/
cp trueskill.pdf ../../similarity_plots/final_results/
cp trueskill.txt ../../similarity_plots/final_results/
cp trueskill_win_prob.pdf ../../similarity_plots/final_results/
cd ../../similarity_plots/simulation
cp result.txt ../final_results/simulation_results.txt
