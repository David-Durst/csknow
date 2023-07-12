from learn_bot.latent.analyze.process_trajectory_comparison import set_pd_print_options
from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd

set_pd_print_options()
fast_df = load_hdf5_to_pd("/home/durst/dev/csknow/analytics/all_train_outputs/behaviorTreeTeamFeatureStore_28.hdf5")
slow_df = load_hdf5_to_pd("/home/durst/dev/csknow/analytics/all_train_outputs_pre_fast/behaviorTreeTeamFeatureStore_28.hdf5")
z = fast_df.compare(slow_df)
print(z[:100])