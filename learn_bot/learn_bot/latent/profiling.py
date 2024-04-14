from pathlib import Path
import sys
import time

import pandas as pd
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


def get_trainable_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_flop_analysis(model, x) -> int:
    flops = FlopCountAnalysis(model, x)
    return flops.total()


def load_non_script_module(model_path: Path):
    model_file = torch.load(model_path / "delta_pos_checkpoint.pt")
    hyperparameter_options: HyperparameterOptions = model_file['hyperparameter_options']
    model = TransformerNestedHiddenLatentModel(model_file['column_transformers'], hyperparameter_options.internal_width,
                                               2 * max_enemies, hyperparameter_options.num_input_time_steps,
                                               hyperparameter_options.layers, hyperparameter_options.heads,
                                               hyperparameter_options.control_type,
                                               hyperparameter_options.player_mask_type,
                                               hyperparameter_options.mask_partial_info,
                                               hyperparameter_options.dim_feedforward,
                                               hyperparameter_options.include_dead)
    model.load_state_dict(model_file['model_state_dict'])
    return model


set_threads = False

def profile_loaded_model(model_path: Path, run_pytorch_profiler):
    global set_threads
    if not set_threads:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        set_threads = True
    temperature_cpu = torch.Tensor([1.])
    first_row = torch.zeros([1, 1144])
    first_row_similarity = torch.zeros([1, 1])
    cpu_model = torch.jit.load(model_path / "delta_pos_script_model.pt")
    trainable_parameters = get_trainable_parameters(cpu_model)
    flops = get_flop_analysis(load_non_script_module(model_path), (first_row, first_row_similarity, temperature_cpu))
    if run_pytorch_profiler:
        print(trainable_parameters)
        print(flops)
        cpu_model(first_row, first_row_similarity, temperature_cpu)
        cpu_model(first_row, first_row_similarity, temperature_cpu)
        cpu_model(first_row, first_row_similarity, temperature_cpu)
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            with record_function("model_inference"):
                cpu_model(first_row, first_row_similarity, temperature_cpu)
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        #prof.export_chrome_trace('/tmp/profiler_output')
    return trainable_parameters, flops

models = {
    "L4H1": "04_04_2024__12_01_47_iw_256_bc_20_pr_0_fr_0_b_1024_it_1_ot_3_lr_4e-05_wd_0.0_l_4_h_1_n_20.0_ros_2.0_ct_SimilarityControl_pm_NoMask_mpi_False_om_NoMask_w_None_ws_None_wns_None_dh_None_ifo_False_c_just_human_all",
    "L4H4": "04_02_2024__04_40_51_iw_256_bc_20_pr_0_fr_0_b_1024_it_1_ot_3_lr_4e-05_wd_0.0_l_4_h_4_n_20.0_ros_2.0_ct_SimilarityControl_pm_NoMask_mpi_False_om_NoMask_w_None_ws_None_wns_None_dh_None_ifo_False_c_just_human_all",
    "L1H1": "04_05_2024__01_15_35_iw_256_bc_80_pr_0_fr_0_b_1024_it_1_ot_3_lr_4e-05_wd_0.0_l_1_h_1_n_20.0_ros_2.0_ct_SimilarityControl_pm_NoMask_mpi_False_om_NoMask_w_None_ws_None_wns_None_dh_None_ifo_False_c_just_human_all",
    "L1H4": "04_04_2024__17_17_32_iw_256_bc_40_pr_0_fr_0_b_1024_it_1_ot_3_lr_4e-05_wd_0.0_l_1_h_4_n_20.0_ros_2.0_ct_SimilarityControl_pm_NoMask_mpi_False_om_NoMask_w_None_ws_None_wns_None_dh_None_ifo_False_c_just_human_all",
    "L16H1": "04_07_2024__00_09_55_iw_256_bc_20_pr_0_fr_0_b_1024_it_1_ot_3_lr_4e-05_wd_0.0_l_16_h_1_n_20.0_ros_2.0_ct_SimilarityControl_pm_NoMask_mpi_False_om_NoMask_w_None_ws_None_wns_None_dh_None_ifo_False_c_just_human_all",
    "L16H4": "04_07_2024__00_09_55_iw_256_bc_20_pr_0_fr_0_b_1024_it_1_ot_3_lr_4e-05_wd_0.0_l_16_h_4_n_20.0_ros_2.0_ct_SimilarityControl_pm_NoMask_mpi_False_om_NoMask_w_None_ws_None_wns_None_dh_None_ifo_False_c_just_human_all",
    "L4H1D256": "04_09_2024__22_25_25_iw_256_bc_20_pr_0_fr_0_b_1024_it_1_ot_3_lr_4e-05_wd_0.0_l_4_h_1_n_20.0_ros_2.0_ct_SimilarityControl_pm_NoMask_mpi_False_om_NoMask_w_None_ws_None_wns_None_dh_None_ifo_False_d_256_c_just_human_all",
}

if __name__ == '__main__':
    if len(sys.argv) > 1:
        profile_loaded_model(Path(sys.argv[1]), True)
    else:
        profile_dicts = []
        for k,v in models.items():
            p, f = profile_loaded_model(Path('/home/durst/dev/csknow/learn_bot/learn_bot/latent/checkpoints') / v, False)
            profile_dicts.append({'Trainable Parameters': float(p), 'FLOPs': float(f)})
        profile_df = pd.DataFrame.from_records(profile_dicts, index=list(models.keys()))
        profile_df.to_csv('/home/durst/dev/csknow/learn_bot/learn_bot/latent/analyze/plots/model_profile.csv', float_format='%.1E')
