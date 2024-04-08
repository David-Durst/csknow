import pathlib
import sys
import time
import torch
import random

from torch._C._profiler import ProfilerActivity
from torch.autograd.profiler import record_function
from torch.profiler import profile

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


def print_trainable_parameters(model):
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))


def profile_loaded_model(model_path):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    temperature_cpu = torch.Tensor([1.])
    first_row = torch.zeros([1, 1144])
    first_row_similarity = torch.zeros([1, 1])
    cpu_model = torch.jit.load(model_path)
    cpu_model(first_row, first_row_similarity, temperature_cpu)
    cpu_model(first_row, first_row_similarity, temperature_cpu)
    cpu_model(first_row, first_row_similarity, temperature_cpu)
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        with record_function("model_inference"):
            cpu_model(first_row, first_row_similarity, temperature_cpu)
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    #prof.export_chrome_trace('/tmp/profiler_output')


if __name__ == '__main__':
    profile_loaded_model(sys.argv[1])