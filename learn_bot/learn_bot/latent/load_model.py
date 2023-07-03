from dataclasses import dataclass

import sys
from typing import Optional

import pandas as pd
import torch
from torch import nn

from learn_bot.latent.dataset import LatentDataset
from learn_bot.latent.engagement.column_names import max_enemies
from learn_bot.latent.latent_hdf5_dataset import MultipleLatentHDF5Dataset
from learn_bot.latent.place_area.filter import filter_region
from learn_bot.latent.place_area.load_data import LoadDataResult
from learn_bot.latent.place_area.pos_abs_delta_conversion import delta_pos_grid_num_cells, AABB
from learn_bot.latent.place_area.column_names import round_id_column, place_area_input_column_types, \
    delta_pos_output_column_types, test_success_col
from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd
from learn_bot.libs.hdf5_wrapper import HDF5Wrapper
from learn_bot.libs.io_transforms import IOColumnTransformers, CUDA_DEVICE_STR
from learn_bot.latent.transformer_nested_hidden_latent_model import TransformerNestedHiddenLatentModel
from learn_bot.latent.train import checkpoints_path, TrainResult, ColumnsToFlip


class LoadedModel:
    column_transformers: IOColumnTransformers
    model: TransformerNestedHiddenLatentModel
    dataset: MultipleLatentHDF5Dataset
    cur_hdf5_index: int
    cur_loaded_pd: pd.DataFrame
    cur_dataset: LatentDataset
    cur_inference_pd: Optional[pd.DataFrame]

    def __init__(self, column_transformers: IOColumnTransformers, model: TransformerNestedHiddenLatentModel,
                 dataset: MultipleLatentHDF5Dataset):
        self.column_transformers = column_transformers
        self.model = model
        self.dataset = dataset
        self.cur_hdf5_index = 0
        self.load_cur_hdf5_as_pd()
        # this will be set later depending on if doing off policy or on policy inference
        self.cur_inference_pd = None

    def load_cur_hdf5_as_pd(self):
        self.cur_loaded_pd = load_hdf5_to_pd(self.dataset.data_hdf5s[self.cur_hdf5_index].hdf5_path)
        self.cur_dataset = LatentDataset(self.cur_loaded_pd, self.column_transformers)


def load_model_file(loaded_data: LoadDataResult) -> LoadedModel:
    if len(sys.argv) < 2:
        raise Exception("must pass checkpoint folder name as argument, like "
                        "07_02_2023__14_32_51_e_60_b_512_lr_4e-05_wd_0.0_l_2_h_4_n_20.0_t_5_c_human_with_added_bot_nav")
    cur_checkpoints_path = checkpoints_path
    if len(sys.argv) > 1:
        cur_checkpoints_path = cur_checkpoints_path / sys.argv[1]
    model_file = torch.load(cur_checkpoints_path / "delta_pos_checkpoint.pt")

    column_transformers = IOColumnTransformers(place_area_input_column_types, delta_pos_output_column_types,
                                               loaded_data.multi_hdf5_wrapper.hdf5_wrappers[0].sample_df)

    model = TransformerNestedHiddenLatentModel(model_file['column_transformers'], 2 * max_enemies,
                                               delta_pos_grid_num_cells, 2, 4)
    model.load_state_dict(model_file['model_state_dict'])
    model.to(CUDA_DEVICE_STR)

    return LoadedModel(column_transformers, model,
                       MultipleLatentHDF5Dataset(loaded_data.multi_hdf5_wrapper.hdf5_wrappers, column_transformers))
