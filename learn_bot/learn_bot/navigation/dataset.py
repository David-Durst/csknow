from torch.utils.data import Dataset
import os
import re
import torch
import tarfile
from PIL import Image
from pathlib import Path


# https://gist.github.com/rwightman/5a7c9232eb57da20afa80cedb8ac2d38
class NavDataset(Dataset):
    root: Path = "trainNavData"

    def __init__(self, df, tar_path):
        self.tar_path = tar_path
        self.tarfile = None  # lazy init in __getitem__
        self.df = df

    def __getitem__(self, index):
        if self.tarfile is None:
            self.tarfile = tarfile.open(self.tar_path)
        tarinfo, target = self.df.iloc[index]
        iob = self.tarfile.extractfile(self.root / tarinfo)
        img = Image.open(iob)
        return img, target

    def __len__(self):
        return len(self.df)