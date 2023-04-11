import pathlib
import time
import torch

def profile_latent_model(model_path: pathlib.Path, batch_size: int, batch: torch.Tensor):
    start = time.time()
    model = torch.jit.load(model_path)
    for i in range(batch_size):
        model(batch[i:i+1, :])
    print((time.time() - start) / batch_size)
    exit(0)