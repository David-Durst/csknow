from einops import rearrange

from learn_bot.latent.load_model import LoadedModel
from learn_bot.latent.transformer_nested_hidden_latent_model import TransformerNestedHiddenLatentModel, \
    get_last_attention_output, get_last_embedding_output
from learn_bot.libs.io_transforms import IOColumnTransformers, CUDA_DEVICE_STR, get_untransformed_outputs, \
    CPU_DEVICE_STR
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional
import numpy as np


def off_policy_inference(loaded_model: LoadedModel):
    loaded_model.model.eval()
    loaded_model.model.add_input_logging(3)
    dataloader = DataLoader(loaded_model.cur_dataset, batch_size=1024, shuffle=False)
    result_nps = []
    attention_masks_list = []
    embeddings_list = []
    with torch.no_grad():
        with tqdm(total=len(dataloader), disable=False) as pbar:
            for batch, (X, Y, similarity) in enumerate(dataloader):
                X, Y, similarity = X.to(CUDA_DEVICE_STR), Y.to(CUDA_DEVICE_STR), similarity.to(CUDA_DEVICE_STR)
                temperature = torch.Tensor([1.]).to(CUDA_DEVICE_STR)
                pred = loaded_model.model(X, similarity, temperature)
                pred_untransformed = \
                    rearrange(get_untransformed_outputs(pred), 'b p t d -> b (p t d)').to(CPU_DEVICE_STR)
                result_nps.append(pred_untransformed.numpy())
                attention_masks_list.append(get_last_attention_output().to(CPU_DEVICE_STR))
                embeddings_list.append(get_last_embedding_output().to(CPU_DEVICE_STR))
                pbar.update(1)
    result_np = np.concatenate(result_nps)
    attention_masks = np.concatenate(attention_masks_list)
    embeddings = np.concatenate(embeddings_list)

    loaded_model.cur_inference_df = loaded_model.column_transformers.get_untransformed_values_whole_pd(result_np, False)
    loaded_model.cur_attention_masks = attention_masks
    loaded_model.cur_embeddings = embeddings


