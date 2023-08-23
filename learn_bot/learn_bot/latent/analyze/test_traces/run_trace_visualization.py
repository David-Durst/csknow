from learn_bot.latent.analyze.process_trajectory_comparison import set_pd_print_options
from learn_bot.latent.analyze.test_traces.run_trace_creation import *
from learn_bot.latent.analyze.test_traces.column_names import *


def visualize_traces(trace_hdf5_data_path):
    trace_df = load_hdf5_to_pd(trace_hdf5_data_path)
    trace_extra_df = load_hdf5_to_pd(trace_hdf5_data_path, root_key='extra',
                                     cols_to_get=[trace_demo_file_name, trace_index_name, num_traces_name,
                                                  trace_one_non_replay_team_name, trace_one_non_replay_bot_name] + trace_is_bot_player_names)
    print(trace_extra_df)
    x = 1


if __name__ == "__main__":
    set_pd_print_options()

    visualize_traces(rollout_aggressive_trace_hdf5_data_path)
