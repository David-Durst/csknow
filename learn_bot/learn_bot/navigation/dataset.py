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


def pil_to_tensor(img: Image) -> Tensor:
    return transforms.ToTensor()(img).squeeze_(0)


def tensor_to_pil(tensor: Tensor) -> Image:
    return transforms.ToPILImage()(tensor.unsqueeze_(0))


# https://gist.github.com/rwightman/5a7c9232eb57da20afa80cedb8ac2d38
class NavDataset(Dataset):
    root: str = "trainNavData"

    def __init__(self, df: pd.DataFrame, tar_path: Path, img_cols_names: List[str]):
        self.tar_path = tar_path
        self.tarfile = None  # lazy init in __getitem__
        self.df = df

        self.id = df.loc[:, 'id']
        self.tick_id = df.loc[:, 'tick id']
        self.tick_id = df.loc[:, 'demo tick id']
        self.round_id = df.loc[:, 'round id']
        self.player_id = df.loc[:, 'player id']
        self.player_name = df.loc[:, 'player name']
        self.img_cols_names = img_cols_names
        self.img_cols = {}
        for img_col_name in img_cols_names:
            self.img_cols[img_col_name] = df.loc[:, img_col_name]

    def __getitem__(self, index):
        if self.tarfile is None:
            self.tarfile = tarfile.open(self.tar_path)
        imgs_tensors = []
        for img_col_name, img_col in self.img_cols.items():
            tarinfo = self.tarfile.getmember(self.root + "/" + img_col[index])
            iob = self.tarfile.extractfile(tarinfo)
            imgs_tensors.append(pil_to_tensor(Image.open(iob)))
        return torch.stack(imgs_tensors)

    def __len__(self):
        return len(self.df)

    def get_image_grid(self, index):
        images_tensor = self[index]
        num_images = images_tensor.shape[0]
        height = images_tensor.shape[1]
        width = images_tensor.shape[2]
        cols = int(math.ceil(math.sqrt(num_images)))
        rows = math.ceil(num_images / cols)
        grid = Image.new('L', size=(cols * width, rows * height))

        for i in range(num_images):
            grid.paste(tensor_to_pil(images_tensor[i]), box=(i % cols * width, i // cols * height))

        draw = ImageDraw.Draw(grid)
        for i in range(num_images):
            text_width, text_height = draw.textsize(self.img_cols_names[i])
            draw.text(
                (i % cols * width + width * 0.5 - text_width / 2, i // cols * height + 10 - text_height / 2),
                self.img_cols_names[i],
                255
            )
        return grid
