import math

import pandas as pd
import torch
from torch.utils.data import Dataset
import tarfile
from PIL import Image, ImageDraw
from pathlib import Path
from typing import List
from torchvision import transforms
from torch import Tensor
from dataclasses import dataclass
from os.path import exists
import random
from joblib import parallel_backend, Parallel, delayed

from learn_bot.navigation.io_transforms import IOColumnAndImageTransformers

def pil_to_tensor(img: Image) -> Tensor:
    return transforms.ToTensor()(img).squeeze_(0)


def tensor_to_pil(tensor: Tensor) -> Image:
    return transforms.ToPILImage()(tensor.unsqueeze_(0))


class NavDataAndLabel:
    non_img_data: torch.Tensor
    img_data: torch.Tensor
    label: torch.Tensor

    def __init__(self, args: List):
        self.non_img_data = args[0]
        self.img_data = args[1]
        self.label = args[2]


# https://gist.github.com/rwightman/5a7c9232eb57da20afa80cedb8ac2d38
class NavDataset(Dataset):
    root = "trainNavData"

    def __init__(self, df: pd.DataFrame, tar_path: Path, img_cols_names: List[str]):
        self.tar_path = tar_path
        self.tarfiles = {}  # lazy init in __getitem__
        self.df = df

        self.id = df.loc[:, 'id']
        self.tick_id = df.loc[:, 'tick id']
        self.tick_id = df.loc[:, 'demo tick id']
        self.round_id = df.loc[:, 'round id']
        self.player_id = df.loc[:, 'player id']
        self.player_name = df.loc[:, 'player name']

        # none is used for vis mode, no need to create all training columns
        self.non_img_X = None
        self.Y = None

        self.img_cols_names = img_cols_names
        self.img_cols = {}
        for img_col_name in img_cols_names:
            self.img_cols[img_col_name] = df.loc[:, img_col_name]

    def add_column_transformers(self, cts: IOColumnAndImageTransformers = None):
        self.non_img_X = torch.tensor(self.df.loc[:, cts.input_types.column_names()].values).float()
        self.Y = torch.tensor(self.df.loc[:, cts.output_types.column_names()].values).float()

    # https://stackoverflow.com/questions/67416496/does-pytorch-dataset-getitem-have-to-return-a-dict
    # no strict API for __getitem__ to implement
    def __getitem__(self, index) -> List:
        round_id = self.round_id.iloc[index]
        if round_id not in self.tarfiles:
            for inner_round_id in self.round_id.unique().tolist():
                tar_path = Path(self.tar_path) / (self.root + "_" + str(inner_round_id) + ".tar")
                if exists(tar_path):
                    self.tarfiles[inner_round_id] = tarfile.open(tar_path)
        imgs_tensors = []
        for img_col_name, img_col in self.img_cols.items():
            tarinfo = self.tarfiles[round_id].getmember(img_col.iloc[index])
            iob = self.tarfiles[round_id].extractfile(tarinfo)
            imgs_tensors.append(pil_to_tensor(Image.open(iob)))
        if self.non_img_X is None:
            return [None, torch.stack(imgs_tensors), None]
        else:
            return [self.non_img_X[index], torch.stack(imgs_tensors), self.Y[index]]

    def __len__(self):
        return len(self.df)

    def precache_all_tars(self):
        for round_id in self.tarfiles.keys():
            self.tarfiles[round_id].getmembers()

    def get_image_grid(self, index):
        images_tensor = NavDataAndLabel(self[index]).img_data
        num_images = images_tensor.shape[0]
        height = images_tensor.shape[1]
        width = images_tensor.shape[2]
        cols = int(math.ceil(math.sqrt(num_images)))
        rows = math.ceil(num_images / cols)
        grid = Image.new('L', size=(cols * width, rows * (height+20)))

        for i in range(num_images):
            grid.paste(tensor_to_pil(images_tensor[i]), box=(i % cols * width, i // cols * (height+20) + 20))

        draw = ImageDraw.Draw(grid)
        for i in range(num_images):
            text_width, text_height = draw.textsize(self.img_cols_names[i])
            draw.text(
                (i % cols * width + width * 0.5 - text_width / 2, i // cols * (height+20) + 10 - text_height / 2),
                self.img_cols_names[i],
                255
            )
        return grid

    def get_img_sample(self) -> torch.Tensor:
        #sample_indices = [self.round_id.index[i] for i in random.sample(range(len(self)), 30)]
        sample_indices = random.sample(range(len(self)), 30)
        images_list = [NavDataAndLabel(self[i]).img_data for i in sample_indices]
        return torch.stack(images_list)
