from learn_bot.latent.analyze.compare_trajectories.process_trajectory_comparison import set_pd_print_options
from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd

set_pd_print_options()
fast_df = load_hdf5_to_pd("/home/durst/dev/csknow/analytics/all_train_outputs/humanVsSmallHumanTrajectorySimilarity.hdf5")
slow_df = load_hdf5_to_pd("/home/durst/dev/csknow/analytics/all_train_outputs/saved_humanVsSmallHumanTrajectorySimilarity.hdf5")
slow_df['length dtw matched indices'] += 1
fast_df.drop('start dtw matched indices', axis=1, inplace=True)
slow_df.drop('start dtw matched indices', axis=1, inplace=True)
fast_df.drop('length dtw matched indices', axis=1, inplace=True)
slow_df.drop('length dtw matched indices', axis=1, inplace=True)
for i in range(len(fast_df)):
    if not fast_df.iloc[i].equals(slow_df.iloc[i]):
        print(f"{i}: {str(fast_df.iloc[i])} {str(slow_df.iloc[i])}")
        print(fast_df.iloc[i] == slow_df.iloc[i])
        exit(0)
z = fast_df.compare(slow_df)
print(z[:100])