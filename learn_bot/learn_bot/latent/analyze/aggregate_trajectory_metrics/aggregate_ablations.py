import sys
from dataclasses import dataclass
from typing import List, Dict

import pandas as pd

from learn_bot.latent.analyze.comparison_column_names import similarity_plots_path

@dataclass
class AblationSource:
    directory: str
    name: str

no_attention = "NoAttn"
history = "History"
l1h1 = "L1H1"
l1h4 = "L1H4"
l4h4 = "L4H4"
l16h1 = "L16H1"
l16h4 = "L16H4"
l1h1d256 = "L1H1D256"
l4h1d256 = "L4H1D256"
feature_ablations: Dict[str, AblationSource] = {
    no_attention: AblationSource("mask_4_4_24_learned_layers_4_heads_1", "4_9_24_learned_layers_4_everyone_mask_only"),
    history: AblationSource("perf4_4_4_24_learned_layers_4_heads_1", "4_11_24_learned_layers_4_head_1_time"),
}
size_ablations: Dict[str, AblationSource] = {
    l1h1: AblationSource("perf1_4_4_24_learned_layers_4_heads_1", "4_6_24_learned_layers_1_heads_1"),
    l1h4: AblationSource("perf1_4_4_24_learned_layers_4_heads_1", "4_6_24_learned_layers_1_heads_4"),
    l4h4: AblationSource("4_2_24_learned_push_standard", "4_2_24_learned_push_standard"),
    l16h1: AblationSource("perf2_4_4_24_learned_layers_4_heads_1", "4_8_24_learned_layers_16_heads_1"),
    l16h4: AblationSource("perf2_4_4_24_learned_layers_4_heads_1", "4_7_24_learned_layers_16_heads_4"),
    l4h1d256: AblationSource("perf3_4_4_24_learned_layers_4_heads_1", "4_9_24_learned_layers_4_heads_1_d_256"),
    l1h1d256: AblationSource("perf4_4_4_24_learned_layers_4_heads_1", "4_11_24_learned_layers_1_heads_1_d_256")
}

map_occupancy = "Map Occupancy"
kill_locations = "Kill Locations"
lifetimes = "Lifetimes"
shots_per_kill = "Shots Per Kill"

def collect_one_ablation(name: str, ablation_source: AblationSource):
    result = {}
    plots_path = similarity_plots_path / ablation_source.directory
    map_occupancy_df = pd.read_csv(plots_path / "diff" / "emd_no_filter.txt", index_col=0)
    result[map_occupancy] = map_occupancy_df.loc[ablation_source.name, ' EMD']
    kill_locations_df = pd.read_csv(plots_path / "diff" / "emd_only_kill.txt", index_col=0)
    result[kill_locations] = kill_locations_df.loc[ablation_source.name, ' EMD']
    with open(plots_path / 'lifetimes_no_filter.txt', 'r') as f:
        new_lifetimes_dict = eval(f.read())
    result[lifetimes] = new_lifetimes_dict[ablation_source.name]
    with open(plots_path / 'shots_per_kill_no_filter.txt', 'r') as f:
        new_shots_per_kill_dict = eval(f.read())
    result[shots_per_kill] = new_shots_per_kill_dict[ablation_source.name]
    print(f"{name}, {result}")


def aggregate_ablations():
    aggregation_plots_path = similarity_plots_path / "agg_ablations"
    aggregation_plots_path.mkdir(parents=True, exist_ok=True)

    for name, ablation_source in feature_ablations.items():
        collect_one_ablation(name, ablation_source)

    for name, ablation_source in size_ablations.items():
        collect_one_ablation(name, ablation_source)



if __name__ == "__main__":
    aggregate_ablations()
