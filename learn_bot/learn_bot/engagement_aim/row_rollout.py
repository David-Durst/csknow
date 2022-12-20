import torch
from torch import nn

from learn_bot.engagement_aim.column_names import base_abs_x_pos_column, base_abs_y_pos_column
from learn_bot.engagement_aim.io_transforms import IOColumnTransformers, FUTURE_TICKS, CUR_TICK, PRIOR_TICKS, \
    PRIOR_TICKS_POS
from learn_bot.libs.temporal_column_names import get_temporal_field_str
from dataclasses import dataclass
import random

input_names_x = [get_temporal_field_str(base_abs_x_pos_column, i) for i in range(PRIOR_TICKS, 0)]
input_names_y = [get_temporal_field_str(base_abs_y_pos_column, i) for i in range(PRIOR_TICKS, 0)]
rolling_input_names = input_names_x + input_names_y
newest_input_names = [get_temporal_field_str(base_abs_x_pos_column, -1),
                      get_temporal_field_str(base_abs_y_pos_column, -1)]


class BlendAmount:
    on_policy_pct: float
    off_policy_pct: float

    def __init__(self, on_policy_pct: float):
        self.on_policy_pct = on_policy_pct
        self.off_policy_pct = 1 - on_policy_pct

    def __str__(self):
        return f"(on_policy_pct: {self.on_policy_pct}, off_policy_pct: {self.off_policy_pct})"


def get_on_policy_blend_amount(epoch_num: int, num_epochs: int) -> BlendAmount:
    return BlendAmount(1.)


def get_off_policy_blend_amount(epoch_num: int, num_epochs: int) -> BlendAmount:
    return BlendAmount(0.)


def get_scheduled_sampling_blend_amount(epoch_num: int, num_epochs: int) -> BlendAmount:
    return BlendAmount(epoch_num / num_epochs)


def row_rollout(model: nn.Module, X: torch.Tensor, transformed_Y: torch.tensor, untransformed_Y: torch.tensor,
                all_inputs_column_transformers: IOColumnTransformers,
                network_inputs_column_transformers: IOColumnTransformers, blend_amount: BlendAmount):
    num_ticks_to_predict = CUR_TICK + FUTURE_TICKS
    input_range = PRIOR_TICKS_POS

    # indices to read output from for saving for later
    transformed_first_output_indices, _ = \
        network_inputs_column_transformers.get_name_ranges_in_time_range(False, True, range(0, 1), False)
    untransformed_first_output_indices, _ = \
        network_inputs_column_transformers.get_name_ranges_in_time_range(False, False, range(0, 1), False)

    network_col_names_to_ranges = network_inputs_column_transformers.get_name_ranges_dict(True, False)
    rolling_input_indices = [network_col_names_to_ranges[col_name].start for col_name in rolling_input_names]
    newest_input_indices = [network_col_names_to_ranges[col_name].start for col_name in newest_input_names]


    transformed_outputs = torch.zeros_like(transformed_Y)
    untransformed_outputs = torch.zeros_like(untransformed_Y)
    for tick_num in range(num_ticks_to_predict):
        input_name_indices, inames = \
            all_inputs_column_transformers.get_name_ranges_in_time_range(True, False,
                                                                         range(PRIOR_TICKS + tick_num,
                                                                               PRIOR_TICKS + tick_num + input_range),
                                                                         True)
        true_output_name_indices, _ = \
            network_inputs_column_transformers.get_name_ranges_in_time_range(False, False,
                                                                             range(PRIOR_TICKS + tick_num + input_range,
                                                                                   PRIOR_TICKS + tick_num + input_range + 1),
                                                                             True)
        tick_X = X[:, input_name_indices].clone()
        #off_policy_transformed_Y, off_policy_untransformed_Y = model(tick_X)
        # after first iteration, replace predicted values
        # can get fresh for all other values because they don't change
        # this removes need for shifting
        if tick_num > 0:
            tmp_tick_X = tick_X.clone()
            tick_X[:, rolling_input_indices] = torch.roll(last_rolling_inputs, -1, 1)
            tick_X[:, newest_input_indices] = last_untransformed_output

        last_rolling_inputs = tick_X[:, rolling_input_indices].detach()
        if tick_num > 0: #not torch.equal(last_tick_X
            x = 1
        last_tick_X = tick_X.clone()

        # predict and record outputs
        transformed_pred, untransformed_pred = model(tick_X)
        # note: when blending, need to apply alpha in transformed mode and recompute untransformed
        # actually, random sampling fixes this, no need to blend between values, just pick one
        transformed_output_indices, _ = \
            network_inputs_column_transformers.get_name_ranges_in_time_range(False, True,
                                                                             range(tick_num, tick_num + 1), False)
        transformed_outputs[:, transformed_output_indices] = \
            transformed_pred[:, transformed_first_output_indices]
        untransformed_output_indices, _ = \
            network_inputs_column_transformers.get_name_ranges_in_time_range(False, False,
                                                                             range(tick_num, tick_num + 1), False)
        untransformed_outputs[:, untransformed_output_indices] = \
            untransformed_pred[:, untransformed_first_output_indices]
        if random.uniform(0, 1) < blend_amount.on_policy_pct:
            last_untransformed_output = untransformed_pred[:, untransformed_first_output_indices].detach()
        else:
            last_untransformed_output = untransformed_Y[:, true_output_name_indices].detach()

    return transformed_outputs, untransformed_outputs
    # add extra dimension so can keep x's and y's grouped together
    #transformed_outputs = [o.unsqueeze(-1) for o in transformed_outputs]
    #untransformed_outputs = [o.unsqueeze(-1) for o in untransformed_outputs]
    #return torch.flatten(torch.cat(transformed_outputs, dim=2), 1), \
    #       torch.flatten(torch.cat(untransformed_outputs, dim=2), 1)