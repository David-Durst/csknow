from learn_bot.latent.dataset import LatentDataset
from learn_bot.latent.transformer_nested_hidden_latent_model import TransformerNestedHiddenLatentModel
from learn_bot.libs.io_transforms import IOColumnTransformers, CUDA_DEVICE_STR, get_untransformed_outputs, \
    CPU_DEVICE_STR
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional
import numpy as np


def off_policy_inference(dataset: LatentDataset, model: TransformerNestedHiddenLatentModel,
                         cts: IOColumnTransformers) -> pd.DataFrame:
    model.eval()
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    result_np: Optional[np.ndarray] = None
    with torch.no_grad():
        with tqdm(total=len(dataloader), disable=False) as pbar:
            for batch, (X, Y) in enumerate(dataloader):
                X, Y = X.to(CUDA_DEVICE_STR), Y.to(CUDA_DEVICE_STR)
                pred = model(X)
                pred_untransformed = get_untransformed_outputs(pred).to(CPU_DEVICE_STR)
                if result_np is not None:
                    result_np = np.concatenate((result_np, pred_untransformed.numpy()))
                else:
                    result_np = pred_untransformed.numpy()
                pbar.update(1)

    return cts.get_untransformed_values_whole_pd(result_np, False)

