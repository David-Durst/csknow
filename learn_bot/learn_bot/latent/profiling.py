from pathlib import Path
import sys
import time
import torch
import random

from fvcore.nn import FlopCountAnalysis
from torch._C._profiler import ProfilerActivity
from torch.autograd.profiler import record_function
from torch.profiler import profile

from learn_bot.latent.engagement.column_names import max_enemies
from learn_bot.latent.hyperparameter_options import HyperparameterOptions
from learn_bot.latent.transformer_nested_hidden_latent_model import TransformerNestedHiddenLatentModel

aggregate_batch_time = 0.
num_batch_calls = 0
aggregate_per_row_time = 0.
num_per_row_calls = 0


def profile_latent_model(model_path: Path, batch_size: int, batch: torch.Tensor):
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


def print_flop_analysis(model, x):
    flops = FlopCountAnalysis(model, x)
    print(flops.total())
    print(flops.by_operator())


def load_non_script_module(model_path: Path):
    model_file = torch.load(model_path / "delta_pos_checkpoint.pt")
    hyperparameter_options: HyperparameterOptions = model_file['hyperparameter_options']
    model = TransformerNestedHiddenLatentModel(model_file['column_transformers'], hyperparameter_options.internal_width,
                                               2 * max_enemies, hyperparameter_options.num_input_time_steps,
                                               hyperparameter_options.layers, hyperparameter_options.heads,
                                               hyperparameter_options.control_type,
                                               hyperparameter_options.player_mask_type,
                                               hyperparameter_options.mask_partial_info,
                                               hyperparameter_options.dim_feedforward)
    model.load_state_dict(model_file['model_state_dict'])
    return model


def profile_loaded_model(model_path: Path):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    temperature_cpu = torch.Tensor([1.])
    first_row = torch.zeros([1, 1144])
    first_row_similarity = torch.zeros([1, 1])
    cpu_model = torch.jit.load(model_path / "delta_pos_script_model.pt")
    print_trainable_parameters(cpu_model)
    print_flop_analysis(load_non_script_module(model_path), (first_row, first_row_similarity, temperature_cpu))
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
    profile_loaded_model(Path(sys.argv[1]))