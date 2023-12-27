from learn_bot.latent.place_area.column_names import flat_input_float_place_area_columns, \
    flat_input_distribution_cat_place_area_columns, hdf5_id_columns
from learn_bot.libs.pd_printing import set_pd_print_options
from learn_bot.latent.analyze.create_test_plant_states import hdf5_key_column
from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd

set_pd_print_options()
cols_to_compare = hdf5_id_columns + flat_input_float_place_area_columns + flat_input_distribution_cat_place_area_columns
fst_df = load_hdf5_to_pd("/home/durst/dev/csknow/analytics/all_train_outputs/behaviorTreeTeamFeatureStore_28.hdf5", cols_to_get=cols_to_compare)
snd_df = load_hdf5_to_pd("/home/durst/dev/csknow/analytics/all_train_outputs/behaviorTreeTeamFeatureStore_28_old.hdf5", cols_to_get=cols_to_compare)
z = fst_df.compare(snd_df)
print(z[:100])