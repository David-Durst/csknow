from pathlib import Path
from typing import List

from learn_bot.latent.place_area.column_names import specific_player_place_area_columns

trace_demo_file_name = "trace demo file"
trace_index_name = "trace index"
num_traces_name = "num traces"
trace_one_non_replay_team_name = "trace one non replay team"
trace_one_non_replay_bot_name = "trace one non replay bot"
trace_is_bot_player_names: List[str] = []
for player_columns in specific_player_place_area_columns:
    trace_is_bot_player_names.append(player_columns.trace_is_bot_player)

rollout_aggressive_trace_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / '..' / 'analytics' / 'rollout_outputs' / 'behaviorTreeTeamFeatureStore_8_22_23_aggressive_learned_trace.hdf5'
rollout_aggressive_humanness_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / '..' / 'analytics' / 'rollout_outputs' / 'humannessMetrics_8_22_23_aggressive_learned_trace.hdf5'
rollout_passive_trace_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / '..' / 'analytics' / 'rollout_outputs' / 'behaviorTreeTeamFeatureStore_8_23_23_passive_learned_trace.hdf5'
rollout_passive_humanness_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / '..' / 'analytics' / 'rollout_outputs' / 'humannessMetrics_8_23_23_passive_learned_trace.hdf5'
trace_plots_path = Path(__file__).parent / 'trace_plots'
