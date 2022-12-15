import torch
from torch import nn

from learn_bot.engagement_aim.io_transforms import IOColumnTransformers, FUTURE_TICKS, CUR_TICK, PRIOR_TICKS


def row_rollout(model: nn.Module, X: torch.Tensor, all_inputs_column_transformers: IOColumnTransformers,
                network_inputs_column_transformers: IOColumnTransformers):
    num_ticks_to_predict = CUR_TICK + FUTURE_TICKS
    input_range = PRIOR_TICKS

    all_but_oldest_input_range = \
        network_inputs_column_transformers.get_name_ranges_in_time_range(True, False, range(PRIOR_TICKS + 1, -1), False)
    first_output_range = \
        network_inputs_column_transformers.get_name_ranges_in_time_range(True, False, range(0, 1), False)

    start_offset = -1 * PRIOR_TICKS
    # start with just input
    # after first iteration, take next input, shift it back by one, and then get next
    for tick_num in range(num_ticks_to_predict):
        cur_offset = start_offset + num_ticks_to_predict
        input_name_ranges = \
            all_inputs_column_transformers.get_name_ranges_in_time_range(True, False,
                                                                         range(cur_offset, cur_offset + input_range),
                                                                         True)
        tick_X = X[:, input_name_ranges]


    raise NotImplementedError