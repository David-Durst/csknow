import torch
from torch import nn

from learn_bot.engagement_aim.column_names import base_changed_offset_coordinates, base_recoil_x_column, \
    base_recoil_y_column, base_ticks_since_holding_attack
from learn_bot.engagement_aim.io_transforms import IOColumnTransformers, FUTURE_TICKS, CUR_TICK, PRIOR_TICKS, \
    PRIOR_TICKS_POS, CUDA_DEVICE_STR
from learn_bot.libs.temporal_column_names import get_temporal_field_str
from dataclasses import dataclass
import random

input_ticks_since_holding_attack = \
    [get_temporal_field_str(base_ticks_since_holding_attack, i) for i in range(PRIOR_TICKS, 0)]
input_names_recoil_x = [get_temporal_field_str(base_recoil_x_column, i) for i in range(PRIOR_TICKS, 0)]
input_names_recoil_y = [get_temporal_field_str(base_recoil_y_column, i) for i in range(PRIOR_TICKS, 0)]
input_names_x = [get_temporal_field_str(base_changed_offset_coordinates.attacker_x_view_angle, i)
                 for i in range(PRIOR_TICKS, 0)]
input_names_y = [get_temporal_field_str(base_changed_offset_coordinates.attacker_y_view_angle, i)
                 for i in range(PRIOR_TICKS, 0)]
rolling_input_names = input_ticks_since_holding_attack + input_names_recoil_x + input_names_recoil_y + \
                      input_names_x + input_names_y
newest_ticks_since_last_holding_attack_name = get_temporal_field_str(base_ticks_since_holding_attack, -1)
second_newest_ticks_since_last_holding_attack_name = get_temporal_field_str(base_ticks_since_holding_attack, -2)
newest_input_names = [
    get_temporal_field_str(base_recoil_x_column, -1),
    get_temporal_field_str(base_recoil_y_column, -1),
    get_temporal_field_str(base_changed_offset_coordinates.attacker_x_view_angle, -1),
    get_temporal_field_str(base_changed_offset_coordinates.attacker_y_view_angle, -1)
]


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


zero_tensor = torch.tensor(0).to(CUDA_DEVICE_STR)
hundred_tensor = torch.tensor(100).to(CUDA_DEVICE_STR)

ROW_ROLLOUT_DEBUG = True

def row_rollout(model: nn.Module, X: torch.Tensor, transformed_Y: torch.tensor, untransformed_Y: torch.tensor,
                all_inputs_column_transformers: IOColumnTransformers,
                network_inputs_column_transformers: IOColumnTransformers, blend_amount: BlendAmount):
    num_ticks_to_predict = CUR_TICK + FUTURE_TICKS
    input_range = PRIOR_TICKS_POS

    # indices to read output from for saving for later
    transformed_first_output_indices, _ = \
        network_inputs_column_transformers.get_name_ranges_in_time_range(False, True, range(0, 1), False, True)
    untransformed_first_output_indices, _ = \
        network_inputs_column_transformers.get_name_ranges_in_time_range(False, False, range(0, 1), False, True)
    # removing cat so can get just those used in on-policy
    untransformed_first_output_indices_non_cat, _ = \
        network_inputs_column_transformers.get_name_ranges_in_time_range(False, False, range(0, 1), False, False)
    untransformed_first_output_indices_cat, _ = \
        network_inputs_column_transformers.get_name_ranges_in_time_range(False, False, range(0, 1), False, True)
    untransformed_first_output_indices_cat_only = \
        untransformed_first_output_indices_cat[len(untransformed_first_output_indices_non_cat):]

    network_col_names_to_ranges = network_inputs_column_transformers.get_name_ranges_dict(True, False)
    rolling_input_indices = [network_col_names_to_ranges[col_name].start for col_name in rolling_input_names]
    newest_input_indices = [network_col_names_to_ranges[col_name].start for col_name in newest_input_names]
    newest_holding_attack_input_indices = \
        network_col_names_to_ranges[newest_ticks_since_last_holding_attack_name].start
    second_newest_holding_attack_input_indices = \
        network_col_names_to_ranges[second_newest_ticks_since_last_holding_attack_name].start


    transformed_outputs = torch.zeros_like(transformed_Y)
    untransformed_outputs = torch.zeros_like(untransformed_Y)
    for tick_num in range(num_ticks_to_predict):
        input_name_indices, inames = \
            all_inputs_column_transformers.get_name_ranges_in_time_range(True, False,
                                                                         range(PRIOR_TICKS + tick_num,
                                                                               PRIOR_TICKS + tick_num + input_range),
                                                                         True, True)
        true_output_name_indices, true_output_names = \
            network_inputs_column_transformers.get_name_ranges_in_time_range(False, False,
                                                                             range(PRIOR_TICKS + tick_num + input_range,
                                                                                   PRIOR_TICKS + tick_num + input_range + 1),
                                                                             True, False)
        tick_X = X[:, input_name_indices].clone()
        if ROW_ROLLOUT_DEBUG:
            ticks_since_holding_attack_test = network_inputs_column_transformers.\
                get_untransformed_values_like(tick_X[0], True, base_ticks_since_holding_attack)
            scaled_recoil_x_test = network_inputs_column_transformers. \
                get_untransformed_values_like(tick_X[0], True, base_recoil_x_column)
            scaled_recoil_y_test = network_inputs_column_transformers. \
                get_untransformed_values_like(tick_X[0], True, base_recoil_y_column)
        #off_policy_transformed_Y, off_policy_untransformed_Y = model(tick_X)
        # after first iteration, replace predicted values
        # can get fresh for all other values because they don't change
        # this removes need for shifting
        if tick_num > 0:
            tmp_tick_X = tick_X.clone()
            saved_newest_ticks_since_holding_attack = X[:, [newest_holding_attack_input_indices]].clone()
            tick_X[:, rolling_input_indices] = torch.roll(last_rolling_inputs, -1, 1)
            tick_X[:, newest_input_indices] = last_untransformed_output
            # since ticks since fire is rolling but not set directly, need to update regardless of on or off policy
            if last_firing_output is None:
                tick_X[:, [newest_holding_attack_input_indices]] = \
                    saved_newest_ticks_since_holding_attack
            else:
                tick_X[:, [newest_holding_attack_input_indices]] = \
                    torch.where(last_firing_output >= 0.5, zero_tensor,
                                torch.min(hundred_tensor, tick_X[:, [second_newest_holding_attack_input_indices]] + 1))
        if ROW_ROLLOUT_DEBUG:
            updated_ticks_since_holding_attack_test = network_inputs_column_transformers. \
                get_untransformed_values_like(tick_X[0], True, base_ticks_since_holding_attack)
            updated_scaled_recoil_x_test = network_inputs_column_transformers. \
                get_untransformed_values_like(tick_X[0], True, base_recoil_x_column)
            updated_scaled_recoil_y_test = network_inputs_column_transformers. \
                get_untransformed_values_like(tick_X[0], True, base_recoil_y_column)

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
                                                                             range(tick_num, tick_num + 1), False,
                                                                             True)
        transformed_outputs[:, transformed_output_indices] = \
            transformed_pred[:, transformed_first_output_indices]
        untransformed_output_indices, _ = \
            network_inputs_column_transformers.get_name_ranges_in_time_range(False, False,
                                                                             range(tick_num, tick_num + 1), False,
                                                                             True)
        untransformed_outputs[:, untransformed_output_indices] = \
            untransformed_pred[:, untransformed_first_output_indices]
        retransformed_pred = network_inputs_column_transformers.transform_columns(False, untransformed_pred, tick_X)
        reuntransformed_pred = network_inputs_column_transformers.untransform_columns(False, retransformed_pred, tick_X)
        # only set to true if on policy, then take as signal to update ticks since last firing
        last_firing_output = None
        if random.uniform(0, 1) < blend_amount.on_policy_pct:
            last_untransformed_output = untransformed_pred[:, untransformed_first_output_indices_non_cat].detach()
            last_firing_output = untransformed_pred[:, untransformed_first_output_indices_cat_only].detach()
        else:
            last_untransformed_output = untransformed_Y[:, true_output_name_indices].detach()

    test_transform = network_inputs_column_transformers.transform_columns(False, untransformed_outputs, X)
    test_transform_rev = network_inputs_column_transformers.untransform_columns(False, test_transform, X)
    return network_inputs_column_transformers.transform_columns(False, untransformed_outputs, X), untransformed_outputs
    # add extra dimension so can keep x's and y's grouped together
    #transformed_outputs = [o.unsqueeze(-1) for o in transformed_outputs]
    #untransformed_outputs = [o.unsqueeze(-1) for o in untransformed_outputs]
    #return torch.flatten(torch.cat(transformed_outputs, dim=2), 1), \
    #       torch.flatten(torch.cat(untransformed_outputs, dim=2), 1)