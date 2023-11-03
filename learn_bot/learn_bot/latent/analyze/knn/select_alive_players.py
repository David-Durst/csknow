from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
import torch
from einops import rearrange
import numpy as np

from learn_bot.latent.order.column_names import team_strs, all_prior_and_cur_ticks
from learn_bot.latent.place_area.column_names import get_similarity_column
from learn_bot.latent.transformer_nested_hidden_latent_model import range_list_to_index_list, get_player_columns_by_str, \
    TransformerNestedHiddenLatentModel
from learn_bot.libs.hdf5_wrapper import HDF5Wrapper


class PlayerColumnIndices:
    ct_alive_cols: List[int]
    t_alive_cols: List[int]
    alive_cols: List[int]
    pos_cols: List[List[int]]

    def __init__(self, model: TransformerNestedHiddenLatentModel):
        # get alive cols
        self.ct_alive_cols = \
            range_list_to_index_list(model.cts.get_name_ranges(True, True, contained_str="alive " + team_strs[0]))
        self.t_alive_cols = \
            range_list_to_index_list(model.cts.get_name_ranges(True, True, contained_str="alive " + team_strs[1]))
        self.alive_cols = self.ct_alive_cols + self.t_alive_cols

        # get pos cols
        all_players_pos_columns = get_player_columns_by_str(model.cts, "player pos")
        all_players_pos_columns_tensor = torch.IntTensor(all_players_pos_columns)
        nested_players_pos_columns_tensor = rearrange(all_players_pos_columns_tensor, '(p t d) -> p t d',
                                                      p=model.num_players, t=all_prior_and_cur_ticks, d=model.num_dim)
        self.pos_cols = []
        for p in range(model.num_players):
            self.pos_cols.append(nested_players_pos_columns_tensor[p, 0].tolist())


def get_id_df_and_alive_pos_and_full_table_id_np(hdf5_wrapper: HDF5Wrapper, model: TransformerNestedHiddenLatentModel,
                                                 num_ct_alive: int, num_t_alive: int) -> Tuple[pd.DataFrame, np.ndarray]:
    player_column_indices = PlayerColumnIndices(model)

    # get rows which have right number of players alive
    num_ct_alive_np = np.sum(hdf5_wrapper.get_all_input_data()[:, player_column_indices.ct_alive_cols], axis=1)
    num_t_alive_np = np.sum(hdf5_wrapper.get_all_input_data()[:, player_column_indices.t_alive_cols], axis=1)

    valid = (num_ct_alive_np == num_ct_alive) & (num_t_alive_np == num_t_alive) & \
            hdf5_wrapper.id_df[get_similarity_column(0)].to_numpy() # require pushing
    valid_id_df = hdf5_wrapper.id_df[valid]
    valid_whole_np = hdf5_wrapper.get_all_input_data()[valid]

    # for each row, record all pos indices of alive players
    alive_players_pos_list: List[np.ndarray] = []
    # also record id of players in full table
    alive_players_id_list: List[np.ndarray] = []
    for i in range(len(valid_whole_np)):
        alive_players_pos_cur_row: List[int] = []
        alive_players_id_cur_row: List[int] = []
        for j, alive_col in enumerate(player_column_indices.alive_cols):
            if valid_whole_np[i, alive_col]:
                alive_players_pos_cur_row += player_column_indices.pos_cols[j]
                alive_players_id_cur_row.append(j)
        alive_players_pos_list.append(np.asarray(alive_players_pos_cur_row))
        alive_players_id_list.append(np.array(alive_players_id_cur_row))
    alive_players_pos_np: np.ndarray = np.stack(alive_players_pos_list, axis=0)
    alive_players_id_np: np.ndarray = np.stack(alive_players_id_list, axis=0)

    # get pos cols for alive players, nest it so each pos is inner dimension
    valid_alive_pos_np = np.take_along_axis(valid_whole_np, alive_players_pos_np, 1)
    nested_valid_alive_pos_np = valid_alive_pos_np.reshape((valid_alive_pos_np.shape[0],
                                                            valid_alive_pos_np.shape[1] // 3,
                                                            3))

    return valid_id_df, nested_valid_alive_pos_np, alive_players_id_np







