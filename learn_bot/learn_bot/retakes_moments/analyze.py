from pathlib import Path

import numpy as np
import pandas as pd
from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd
from learn_bot.retakes_moments.column_names import *

human_retakes_per_round_moments_path = Path(__file__).parent / '..' / '..' / '..' / 'analytics' / 'csv_outputs' / 'humanRetakesPerRoundMoments.hdf5'
bot_retakes_per_round_moments_path = Path(__file__).parent / '..' / '..' / '..' / 'analytics' / 'csv_outputs' / 'botRetakesPerRoundMoments.hdf5'


def analyze(all_per_round_df: pd.DataFrame):
    all_per_round_df.replace(
        {ctMomentsStrs.botType: retake_bot_types_to_names, tMomentsStrs.botType: retake_bot_types_to_names}, inplace=True)
    with open(Path(__file__).parent / 'per_round_moments.csv', 'w+') as f:
        for ct_pair in column_names:
            ct_mean_str = ct_pair.ct_column_name + " mean"
            ct_stddev_str = ct_pair.ct_column_name + " stddev"
            ct_median_str = ct_pair.ct_column_name + " median"
            ct_iqr_str = ct_pair.ct_column_name + " iqr"
            t_mean_str = ct_pair.t_column_name + " mean"
            t_stddev_str = ct_pair.t_column_name + " stddev"
            t_median_str = ct_pair.t_column_name + " median"
            t_iqr_str = ct_pair.t_column_name + " iqr"
            # quantile skis na: https://stackoverflow.com/questions/52171333/does-the-quantile-function-in-pandas-ignore-nan
            all_agg_df = all_per_round_df.groupby([ctMomentsStrs.botType, tMomentsStrs.botType])\
                .agg(**{
                    ct_mean_str: (ct_pair.ct_column_name, lambda x: x.mean(skipna=True)),
                    ct_stddev_str: (ct_pair.ct_column_name, lambda x: x.std(skipna=True)),
                    ct_median_str: (ct_pair.ct_column_name, lambda x: x.median(skipna=True)),
                    ct_iqr_str: (ct_pair.ct_column_name, lambda x: x.quantile(0.75) - x.quantile(0.25)),
                    t_mean_str: (ct_pair.t_column_name, lambda x: x.mean(skipna=True)),
                    t_stddev_str: (ct_pair.t_column_name, lambda x: x.std(skipna=True)),
                    t_median_str: (ct_pair.t_column_name, lambda x: x.median(skipna=True)),
                    t_iqr_str: (ct_pair.t_column_name, lambda x: x.quantile(0.75) - x.quantile(0.25)),
            })
            all_agg_df.to_csv(f)
            f.write("\n")




if __name__ == "__main__":
    human_per_round_df = load_hdf5_to_pd(human_retakes_per_round_moments_path)
    human_per_round_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(f'''num human rounds: {len(human_per_round_df)}''')
    bot_per_round_df = load_hdf5_to_pd(bot_retakes_per_round_moments_path)
    bot_per_round_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(f'''num bot rounds: {len(bot_per_round_df)}''')
    all_per_round_df = pd.concat([human_per_round_df, bot_per_round_df])
    analyze(all_per_round_df)
