from typing import List, Tuple

import pandas as pd
import torch
from einops import rearrange
import numpy as np

from learn_bot.latent.order.column_names import team_strs, all_prior_and_cur_ticks
from learn_bot.latent.transformer_nested_hidden_latent_model import range_list_to_index_list, get_player_columns_by_str, \
    TransformerNestedHiddenLatentModel
from learn_bot.libs.hdf5_wrapper import HDF5Wrapper


def get_id_df_and_alive_pos_np(hdf5_wrapper: HDF5Wrapper, model: TransformerNestedHiddenLatentModel,
                               num_ct_alive: int, num_t_alive: int) -> Tuple[pd.DataFrame, np.ndarray]:
    # get alive cols
    ct_alive_cols: List[int] = \
        range_list_to_index_list(model.cts.get_name_ranges(True, True, contained_str="alive " + team_strs[0]))
    t_alive_cols: List[int] = \
        range_list_to_index_list(model.cts.get_name_ranges(True, True, contained_str="alive " + team_strs[1]))
    alive_cols: List[int] = ct_alive_cols + t_alive_cols

    # get rows which have right number of players alive
    num_ct_alive_np = np.sum(hdf5_wrapper.get_all_input_data()[:, ct_alive_cols], axis=1)
    num_t_alive_np = np.sum(hdf5_wrapper.get_all_input_data()[:, t_alive_cols], axis=1)

    valid = (num_ct_alive_np >= num_ct_alive) & (num_t_alive_np == num_t_alive)
    valid_id_df = hdf5_wrapper.id_df[valid]
    valid_whole_np = hdf5_wrapper.get_all_input_data()[valid]

    # get pos cols
    pos_cols: List[List[int]] = []
    all_players_pos_columns = get_player_columns_by_str(model.cts, "player pos")
    all_players_pos_columns_tensor = torch.IntTensor(all_players_pos_columns)
    nested_players_pos_columns_tensor = rearrange(all_players_pos_columns_tensor, '(p t d) -> p t d',
                                                  p=model.num_players, t=all_prior_and_cur_ticks, d=model.num_dim)
    for p in range(model.num_players):
        pos_cols.append(nested_players_pos_columns_tensor[p, 0].tolist())

    # for each row, record all pos indices of alive players
    alive_players_pos_list: List[np.ndarray] = []
    for i in range(len(valid_whole_np)):
        alive_players_pos_cur_row: List[int] = []
        for i, alive_col in enumerate(alive_cols):
            if valid_whole_np[i, alive_col]:
                alive_players_pos_cur_row += pos_cols[i]
        alive_players_pos_list.append(np.ndarray(alive_players_pos_cur_row))
    alive_players_pos_np: np.ndarray = np.stack(alive_players_pos_list, axis=0)

    # get pos cols for alive players, nest it so each pos is inner dimension
    valid_alive_pos_np = np.take_along_axis(valid_whole_np, alive_players_pos_np, 1)
    nested_valid_alive_pos_np = valid_alive_pos_np.reshape((valid_alive_pos_np.shape[0],
                                                            valid_alive_pos_np.shape[1] // 3,
                                                            3))

    return valid_id_df, nested_valid_alive_pos_np







