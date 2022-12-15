import torch
from torch import nn

from learn_bot.engagement_aim.column_names import base_abs_x_pos_column, base_abs_y_pos_column
from learn_bot.engagement_aim.io_transforms import IOColumnTransformers, FUTURE_TICKS, CUR_TICK, PRIOR_TICKS, \
    PRIOR_TICKS_POS
from learn_bot.libs.temporal_column_names import get_temporal_field_str

newest_input_names = [get_temporal_field_str(base_abs_x_pos_column, -1),
                      get_temporal_field_str(base_abs_y_pos_column, -1)]


def row_rollout(model: nn.Module, X: torch.Tensor, all_inputs_column_transformers: IOColumnTransformers,
                network_inputs_column_transformers: IOColumnTransformers):
    num_ticks_to_predict = CUR_TICK + FUTURE_TICKS
    input_range = PRIOR_TICKS_POS

    # no non-temporal data as will get that from X repeatedly
    all_but_newest_input_indices = \
        network_inputs_column_transformers.get_name_ranges_in_time_range(True, False, range(PRIOR_TICKS, -1), False)
    first_output_indices = \
        network_inputs_column_transformers.get_name_ranges_in_time_range(False, False, range(0, 1), False)

    network_col_names_to_ranges = network_inputs_column_transformers.get_name_ranges_dict(True, False)
    newest_input_indices = [network_col_names_to_ranges[col_name].start for col_name in newest_input_names]

    start_offset = PRIOR_TICKS
    # start with just input
    # after first iteration, take next input, shift it back by one, and then get next
    first_input_name_indices = \
        all_inputs_column_transformers.get_name_ranges_in_time_range(True, False,
                                                                     range(start_offset, start_offset + input_range),
                                                                     True)
    tick_X = X[:, first_input_name_indices].clone()
    untransformed_outputs = []
    transformed_outputs = []
    next_input_tick_offset = 0
    for tick_num in range(num_ticks_to_predict):

        # predict and record outputs
        transformed_Y, untransformed_Y = model(tick_X)
        transformed_outputs.append(transformed_Y[:, first_output_indices])
        untransformed_outputs.append(untransformed_Y[:, first_output_indices])

        # shift in next input if not finished
        if tick_num < num_ticks_to_predict:
            # include non-temporal data here
            next_input_name_ranges = \
                all_inputs_column_transformers.get_name_ranges_in_time_range(True, False,
                                                                             range(next_input_tick_offset, next_input_tick_offset + 1),
                                                                             True)
            tick_X = torch.cat([tick_X[:, all_but_newest_input_indices], X[:, next_input_name_ranges]], dim=1)
            tick_X[:, newest_input_indices] = untransformed_Y[:, first_output_indices].detach()

            next_input_tick_offset += 1

    return torch.cat(untransformed_outputs, dim=1), torch.cat(transformed_outputs, dim=1)