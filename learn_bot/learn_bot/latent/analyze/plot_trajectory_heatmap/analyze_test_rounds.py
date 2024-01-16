from typing import Optional

import pandas as pd

from learn_bot.latent.analyze.compare_trajectories.process_trajectory_comparison import get_test_plant_states_pd
from learn_bot.latent.analyze.create_test_plant_states import num_ct_alive_column, plant_tick_id_column, \
    num_t_alive_column

rounds_by_ct_alive: Optional[pd.DataFrame] = None
rounds_by_t_alive: Optional[pd.DataFrame] = None
rounds_by_both_alive: Optional[pd.DataFrame] = None
num_rounds: int = -1


def analyze_push_test_rounds():
    global rounds_by_t_alive, rounds_by_ct_alive, rounds_by_both_alive, num_rounds
    push_test_plant_test_df = get_test_plant_states_pd(True)
    num_rounds = len(push_test_plant_test_df)
    rounds_by_ct_alive = push_test_plant_test_df.groupby(num_ct_alive_column)[plant_tick_id_column].count()
    rounds_by_t_alive = push_test_plant_test_df.groupby(num_t_alive_column)[plant_tick_id_column].count()
    rounds_by_both_alive = push_test_plant_test_df.groupby([num_ct_alive_column, num_t_alive_column])[plant_tick_id_column].count()
