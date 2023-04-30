from pathlib import Path

import numpy as np
import pandas as pd
from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd
from learn_bot.retakes_moments.column_names import *

human_retakes_per_round_moments_path = Path(__file__).parent / '..' / '..' / '..' / 'analytics' / 'csv_outputs' / 'humanRetakesPerRoundMoments.hdf5'
bot_retakes_per_round_moments_path = Path(__file__).parent / '..' / '..' / '..' / 'analytics' / 'csv_outputs' / 'botRetakesPerRoundMoments.hdf5'


def analyze(human_per_round_df: pd.DataFrame, bot_per_round_df: pd.DataFrame):
    human_per_round_df.replace(
        {ctMomentsStrs.botType: retake_bot_types_to_names, tMomentsStrs.botType: retake_bot_types_to_names}, inplace=True)
    bot_per_round_df.replace(
        {ctMomentsStrs.botType: retake_bot_types_to_names, tMomentsStrs.botType: retake_bot_types_to_names}, inplace=True)
    bot_agg_df = \
        bot_per_round_df.groupby([ctMomentsStrs.botType, tMomentsStrs.botType]).agg({ctMomentsStrs.win: lambda x: x.mean(skipna=True)})
    #bot_agg_df = bot_per_round_df.groupby([ctMomentsStrs.botType, tMomentsStrs.botType]).mean(skipna=True)
    human_agg_df = human_per_round_df.agg(['avg']).mean(skipna=True)



if __name__ == "__main__":
    human_per_round_df = load_hdf5_to_pd(human_retakes_per_round_moments_path)
    human_per_round_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    bot_per_round_df = load_hdf5_to_pd(bot_retakes_per_round_moments_path)
    bot_per_round_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    all_per_round_df = pd.concat([human_per_round_df, bot_per_round_df])
    analyze(human_per_round_df, bot_per_round_df)
