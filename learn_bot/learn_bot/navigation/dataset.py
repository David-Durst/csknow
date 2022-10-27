import pandas as pd
from torch.utils.data import Dataset
import tarfile
from PIL import Image
from pathlib import Path
from typing import List
from torchvision import transforms
from torch import Tensor


def pil_to_tensor(img: Image) -> Tensor:
    return transforms.ToTensor()(img).unsqueeze_(0)


def tensor_to_pil(tensor: Tensor) -> Image:
    return transforms.ToPILImage()(tensor.squeeze_(0))


# https://gist.github.com/rwightman/5a7c9232eb57da20afa80cedb8ac2d38
class NavDataset(Dataset):
    root: Path = "trainNavData"

    def __init__(self, df: pd.DataFrame, tar_path: Path, img_cols_names: List[str]):
        self.tar_path = tar_path
        self.tarfile = None  # lazy init in __getitem__
        self.df = df

        self.id = df.loc[:, 'id']
        self.tick_id = df.loc[:, 'tick id']
        self.player_id = df.loc[:, 'player id']
        self.img_cols = {}
        for img_col_name in img_cols_names:
            self.img_cols[img_col_name] = df.loc[:, img_col_name]

    def __getitem__(self, index):
        if self.tarfile is None:
            self.tarfile = tarfile.open(self.tar_path)
        tarinfo, target = self.df.iloc[index]
        imgs = {}
        for img_col_name, img_col in self.img_cols:
            iob = self.tarfile.extractfile(self.root / tarinfo)
            imgs[img_col_name] = pil_to_tensor(Image.open(iob))
        return imgs

    def __len__(self):
        return len(self.df)