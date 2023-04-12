import pathlib
import time
import torch
import random

aggregate_batch_time = 0.
num_batch_calls = 0
aggregate_per_row_time = 0.
num_per_row_calls = 0


def profile_latent_model(model_path: pathlib.Path, batch_size: int, batch: torch.Tensor):
    global aggregate_batch_time, num_batch_calls, aggregate_per_row_time, num_per_row_calls
    torch.set_num_threads(1)
    model = torch.jit.optimize_for_inference(torch.jit.load(model_path))
    batch_cpu = batch.to("cpu")
    if num_per_row_calls == 0 and num_batch_calls == 0:
        print(model.code)
    if random.uniform(0,1) < 0.5:
        start = time.time()
        model(batch_cpu)
        aggregate_batch_time += (time.time() - start)
        num_batch_calls += 1
        print("CPU batch: " + str(aggregate_batch_time / (batch_size * num_batch_calls)))
    else:
        start = time.time()
        for i in range(batch_size):
            model(batch_cpu[i:i+1, :])
        aggregate_per_row_time += (time.time() - start)
        num_per_row_calls += 1
        print("CPU per line: " + str(aggregate_per_row_time / (batch_size * num_per_row_calls)))