from learn_bot.libs.pd_printing import set_pd_print_options
from learn_bot.latent.analyze.create_test_plant_states import hdf5_key_column
from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd

set_pd_print_options()
fst_df = load_hdf5_to_pd("/home/durst/dev/csknow/learn_bot/learn_bot/libs/saved_train_test_splits/push_only_test_plant_states.hdf5")
snd_df = load_hdf5_to_pd("/home/durst/dev/csknow/learn_bot/learn_bot/libs/saved_train_test_splits/tmp_push_only_test_plant_states.hdf5")
fst_df.drop(hdf5_key_column, axis=1, inplace=True)
z = fst_df.compare(snd_df)
print(z[:100])