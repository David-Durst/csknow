from dataclasses import dataclass

import sys
from typing import Optional, Dict

import numpy as np
import pandas as pd
import torch
from torch import nn

from learn_bot.latent.dataset import LatentDataset
from learn_bot.latent.engagement.column_names import max_enemies
from learn_bot.latent.hyperparameter_options import HyperparameterOptions
from learn_bot.latent.latent_hdf5_dataset import MultipleLatentHDF5Dataset
from learn_bot.latent.order.column_names import num_radial_ticks
from learn_bot.latent.place_area.filter import filter_region
from learn_bot.latent.place_area.load_data import LoadDataResult
from learn_bot.latent.place_area.pos_abs_from_delta_grid_or_radial import delta_pos_grid_num_cells, AABB
from learn_bot.latent.place_area.column_names import round_id_column, place_area_input_column_types, \
    radial_vel_output_column_types, test_success_col, num_radial_bins
from learn_bot.libs.df_grouping import make_index_column
from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd, load_hdf5_extra_column
from learn_bot.libs.hdf5_wrapper import HDF5Wrapper
from learn_bot.libs.io_transforms import IOColumnTransformers, CUDA_DEVICE_STR
from learn_bot.latent.transformer_nested_hidden_latent_model import TransformerNestedHiddenLatentModel, PlayerMaskType
from learn_bot.latent.train_paths import checkpoints_path


class LoadedModel:
    column_transformers: IOColumnTransformers
    model: TransformerNestedHiddenLatentModel
    dataset: MultipleLatentHDF5Dataset
    filename_to_hdf5_index: Dict[str, int]
    cur_hdf5_index: int
    cur_loaded_df: pd.DataFrame
    cur_demo_names: np.ndarray
    cur_dataset: LatentDataset
    cur_inference_df: Optional[pd.DataFrame]

    def __init__(self, column_transformers: IOColumnTransformers, model: TransformerNestedHiddenLatentModel,
                 dataset: MultipleLatentHDF5Dataset):
        self.column_transformers = column_transformers
        self.model = model
        self.dataset = dataset
        self.filename_to_hdf5_index = {}
        for i, hdf5_wrapper in enumerate(self.dataset.data_hdf5s):
            self.filename_to_hdf5_index[str(hdf5_wrapper.hdf5_path.name)] = i
        self.cur_hdf5_index = 0
        self.load_cur_hdf5_as_pd()
        # this will be set later depending on if doing off policy or on policy inference
        self.cur_inference_df = None

    def load_cur_hdf5_as_pd(self, load_cur_dataset=True, cast_bool_to_int=True):
        self.cur_loaded_df = load_hdf5_to_pd(self.dataset.data_hdf5s[self.cur_hdf5_index].hdf5_path,
                                             cast_bool_to_int=cast_bool_to_int)
        # done to apply limit on hdf5 wrapper's id df to actual df
        self.cur_loaded_df = self.cur_loaded_df.iloc[self.dataset.data_hdf5s[self.cur_hdf5_index].id_df['id'], :]
        make_index_column(self.cur_loaded_df)
        self.cur_demo_names = self.load_cur_hdf5_demo_names()
        if load_cur_dataset:
            self.cur_dataset = LatentDataset(self.cur_loaded_df, self.column_transformers,
                                             self.dataset.data_hdf5s[self.cur_hdf5_index].id_df)

    def load_cur_hdf5_demo_names(self) -> np.ndarray:
        return load_hdf5_extra_column(self.dataset.data_hdf5s[self.cur_hdf5_index].hdf5_path, 'demo file').astype('U')

    def get_cur_id_df(self):
        return self.dataset.data_hdf5s[self.cur_hdf5_index].id_df

    def get_cur_hdf5_filename(self) -> str:
        return str(self.dataset.data_hdf5s[self.cur_hdf5_index].hdf5_path.name)


def load_model_file(loaded_data: LoadDataResult, use_test_data_only: bool = False,
                    model_name_override: Optional[str] = None) -> LoadedModel:
    if len(sys.argv) < 2 and model_name_override is None:
        raise Exception("must pass checkpoint folder name as argument, like "
                        "07_02_2023__14_32_51_e_60_b_512_lr_4e-05_wd_0.0_l_2_h_4_n_20.0_t_5_c_human_with_added_bot_nav")
    cur_checkpoints_path = checkpoints_path
    if model_name_override is not None:
        cur_checkpoints_path = cur_checkpoints_path / model_name_override
    elif len(sys.argv) > 1:
        cur_checkpoints_path = cur_checkpoints_path / sys.argv[1]
    model_file = torch.load(cur_checkpoints_path / "delta_pos_checkpoint.pt")

    hdf5_wrappers = loaded_data.multi_hdf5_wrapper.test_hdf5_wrappers if use_test_data_only else \
        loaded_data.multi_hdf5_wrapper.hdf5_wrappers

    column_transformers = IOColumnTransformers(place_area_input_column_types, radial_vel_output_column_types,
                                               hdf5_wrappers[0].sample_df)

    hyperparameter_options: HyperparameterOptions = model_file['hyperparameter_options']
    model = TransformerNestedHiddenLatentModel(model_file['column_transformers'], hyperparameter_options.internal_width,
                                               2 * max_enemies, hyperparameter_options.num_input_time_steps,
                                               num_radial_ticks, num_radial_bins, 2, 4,
                                               hyperparameter_options.player_mask_type)
    model.load_state_dict(model_file['model_state_dict'])
    model.to(CUDA_DEVICE_STR)

    return LoadedModel(column_transformers, model, MultipleLatentHDF5Dataset(hdf5_wrappers, column_transformers))
