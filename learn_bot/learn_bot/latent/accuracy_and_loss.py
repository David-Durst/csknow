from typing import Callable

import torch
from einops import rearrange, repeat, pack
from torch.utils.tensorboard import SummaryWriter

from learn_bot.latent.dataset import *
from learn_bot.latent.order.column_names import num_radial_ticks
from learn_bot.latent.place_area.pos_abs_from_delta_grid_or_radial import get_delta_indices_from_grid, \
    get_delta_pos_from_radial
from learn_bot.latent.transformer_nested_hidden_latent_model import TransformerNestedHiddenLatentModel
from learn_bot.libs.io_transforms import IOColumnTransformers, ColumnTransformerType, CPU_DEVICE_STR, \
    get_untransformed_outputs, get_transformed_outputs, CUDA_DEVICE_STR
from math import sqrt
from torch import nn

# https://stackoverflow.com/questions/65192475/pytorch-logsoftmax-vs-softmax-for-crossentropyloss
# no need to do softmax for classification output
cross_entropy_loss_fn: Optional[nn.CrossEntropyLoss] = None
#huber_loss_fn = nn.HuberLoss(reduction='none')
mse_loss_fn = nn.MSELoss(reduction='none')

class LatentLosses:
    cat_loss: torch.Tensor
    cat_accumulator: float
    float_loss: torch.Tensor
    float_accumulator: float
    duplicate_last_cat_loss: torch.Tensor
    duplicate_last_cat_accumulator: float
    duplicate_last_float_loss: torch.Tensor
    duplicate_last_float_accumulator: float
    total_loss: torch.Tensor
    total_accumulator: float

    def __init__(self):
        self.cat_loss = torch.zeros([1]).to(CUDA_DEVICE_STR)
        self.cat_accumulator = 0.
        self.float_loss = torch.zeros([1]).to(CUDA_DEVICE_STR)
        self.float_accumulator = 0.
        self.duplicate_last_cat_loss = torch.zeros([1]).to(CUDA_DEVICE_STR)
        self.duplicate_last_cat_accumulator = 0.
        self.duplicate_last_float_loss = torch.zeros([1]).to(CUDA_DEVICE_STR)
        self.duplicate_last_float_accumulator = 0.
        self.total_loss = torch.zeros([1]).to(CUDA_DEVICE_STR)
        self.total_accumulator = 0.

    def __iadd__(self, other):
        self.cat_accumulator += other.cat_loss.item()
        self.float_accumulator += other.float_loss.item()
        self.duplicate_last_cat_accumulator += other.duplicate_last_cat_loss.item()
        self.duplicate_last_float_accumulator += other.duplicate_last_float_loss.item()
        self.total_accumulator += other.total_loss.item()
        return self

    def __itruediv__(self, other):
        self.cat_accumulator /= other
        self.float_accumulator /= other
        self.duplicate_last_cat_accumulator /= other
        self.duplicate_last_float_accumulator /= other
        self.total_accumulator /= other
        return self

    def add_scalars(self, writer: SummaryWriter, prefix: str, total_epoch_num: int):
        writer.add_scalar(prefix + '/loss/cat', self.cat_accumulator, total_epoch_num)
        writer.add_scalar(prefix + '/loss/float', self.float_accumulator, total_epoch_num)
        writer.add_scalar(prefix + '/loss/repeated cat', self.duplicate_last_cat_accumulator, total_epoch_num)
        writer.add_scalar(prefix + '/loss/repeated float', self.duplicate_last_float_accumulator, total_epoch_num)
        writer.add_scalar(prefix + '/loss/total', self.total_accumulator, total_epoch_num)


# https://discuss.pytorch.org/t/how-to-combine-multiple-criterions-to-a-loss-function/348/4
def compute_loss(model: TransformerNestedHiddenLatentModel, pred, Y, X_orig: Optional, X_rollout: Optional,
                 duplicated_last, num_players, weight_loss: bool) -> LatentLosses:
    global cross_entropy_loss_fn
    pred_transformed = get_transformed_outputs(pred)

    losses = LatentLosses()

    # merging time steps so can do filtering before cross entropy loss
    Y_per_player_time_step = rearrange(Y, "b (p t d) -> (b p t) d", p=num_players, t=num_radial_ticks)
    pred_transformed_per_player_time_step = rearrange(pred_transformed, "b p t d -> (b p t) d",
                                                      p=num_players, t=num_radial_ticks)
    duplicated_last_per_player_time_step = repeat(duplicated_last, "b -> (b repeat)", repeat=num_players * num_radial_ticks)
    valid_rows = Y_per_player_time_step.sum(axis=1) > 0.1
    valid_Y_transformed = Y_per_player_time_step[valid_rows]
    valid_pred_transformed = pred_transformed_per_player_time_step[valid_rows]
    valid_duplicated = duplicated_last_per_player_time_step[valid_rows]

    if X_orig is not None:
        # predictions for multiple time steps in future, but X is just cur time step
        X_orig_cur_pos_per_players = rearrange(X_orig[:, model.players_cur_pos_columns], "b (p d) -> (b p) d",
                                               p=num_players, d=model.num_dim)
        X_rollout_cur_pos_per_players = rearrange(X_rollout[:, model.players_cur_pos_columns], "b (p d) -> (b p) d",
                                                  p=num_players, d=model.num_dim)
        valid_rows_one_time_step = rearrange(valid_rows, '(bp t) -> bp t', t=num_radial_ticks)[:, 0]
        valid_duplicated_one_time_step = rearrange(valid_duplicated, '(bp t) -> bp t', t=num_radial_ticks)[:, 0]
        valid_X_orig_cur_pos_per_players = X_orig_cur_pos_per_players[valid_rows_one_time_step]
        valid_X_rollout_cur_pos_per_players = X_rollout_cur_pos_per_players[valid_rows_one_time_step]
    #cat_loss = cross_entropy_loss_fn(valid_pred_transformed, valid_Y_transformed)
    #if torch.isnan(cat_loss).any():
    #    print('bad loss')
    #losses.cat_loss += cat_loss
    losses_to_cat = []
    losses_to_float = []

    # weight stopping the most
    weights = torch.tensor([50. if i == 0 else 1. for i in range(valid_pred_transformed.shape[1])],
                           device=valid_Y_transformed.device)
    if weight_loss:
        weight_sum = torch.sum(weights)
    else:
        weight_sum = 1.
    if cross_entropy_loss_fn is None:
        if weight_loss:
            cross_entropy_loss_fn = nn.CrossEntropyLoss(reduction='none', weight=weights)
            #unweighted_cross_entropy_loss_fn = nn.CrossEntropyLoss(reduction='none')
        else:
            cross_entropy_loss_fn = nn.CrossEntropyLoss(reduction='none')

    if valid_Y_transformed[~valid_duplicated].shape[0] > 0:
        cat_loss = cross_entropy_loss_fn(valid_pred_transformed[~valid_duplicated],
                                         valid_Y_transformed[~valid_duplicated])# / weight_sum
        #unweighted_cat_loss = unweighted_cross_entropy_loss_fn(valid_pred_transformed[~valid_duplicated],
        #                                 valid_Y_transformed[~valid_duplicated])
        if torch.isnan(cat_loss).any():
            print('bad cat loss')
        losses.cat_loss = torch.mean(cat_loss)
        losses_to_cat.append(cat_loss)
        if X_orig is not None:
            # mse + sum across x/y/z and sqrt == L2 distance
            float_loss = mse_loss_fn(valid_X_rollout_cur_pos_per_players[~valid_duplicated_one_time_step],
                                     valid_X_orig_cur_pos_per_players[~valid_duplicated_one_time_step]) / 5.
            # repeat the L2 distance for each time step that has a categorical prediction)
            float_loss = rearrange(
                repeat(torch.sqrt(torch.sum(float_loss, axis=1, keepdim=True)), 'b 1 -> b t', t=num_radial_ticks),
                'b t -> (b t)')
            if torch.isnan(float_loss).any():
                print('bad huber loss')
            losses.float_loss = torch.mean(float_loss)
            losses_to_float.append(float_loss)
    if valid_Y_transformed[valid_duplicated].shape[0] > 0.:
        duplicated_last_cat_loss = cross_entropy_loss_fn(valid_pred_transformed[valid_duplicated],
                                                         valid_Y_transformed[valid_duplicated]) / weight_sum
        if torch.isnan(duplicated_last_cat_loss).any():
            print('bad cat loss')
        losses.duplicate_last_cat_loss = torch.mean(duplicated_last_cat_loss)
        losses_to_cat.append(duplicated_last_cat_loss)
        if X_orig is not None:
            duplicated_last_float_loss = mse_loss_fn(valid_X_rollout_cur_pos_per_players[valid_duplicated_one_time_step],
                                                     valid_X_orig_cur_pos_per_players[~valid_duplicated_one_time_step]) / 5.
            duplicated_last_float_loss = rearrange(
                repeat(torch.sqrt(torch.sum(duplicated_last_float_loss, axis=1)), 'b 1 -> b t', t=num_radial_ticks),
                'b t -> (b t)')
            if torch.isnan(duplicated_last_float_loss).any():
                print('bad float loss')
            losses.duplicated_last_float_loss = torch.mean(duplicated_last_float_loss)
            losses_to_float.append(duplicated_last_float_loss)
    #total_loss = cross_entropy_loss_fn(valid_pred_transformed, valid_Y_transformed)
    #if torch.isnan(total_loss).any():
    #    print('bad loss')
    #losses.total_loss = total_loss
    if X_orig is None:
        losses.total_loss = torch.mean(pack(losses_to_cat, '*')[0])
    else:
        cat_loss_packed = pack(losses_to_cat, '*')[0]
        float_loss_packed = pack(losses_to_float, '*')[0]
        losses.total_loss = torch.mean(cat_loss_packed * float_loss_packed)

    return losses

duplicated_name_str = 'duplicated'

def compute_accuracy_and_delta_diff(pred, Y, duplicated_last, accuracy, delta_diff_xy, delta_diff_xyz,
                                    valids_per_accuracy_column, num_players, column_transformers: IOColumnTransformers,
                                    stature_to_speed):
    pred_untransformed = get_untransformed_outputs(pred)

    name = column_transformers.output_types.categorical_distribution_first_sub_cols[0]

    # keeping time steps flattened since just summing across all at end
    Y_per_player = rearrange(Y, "b (p t d) -> b (p t) d", p=num_players, t=num_radial_ticks)
    Y_label_per_player = torch.argmax(Y_per_player, -1)
    pred_untransformed_per_player = rearrange(pred_untransformed, "b p t d -> b (p t) d",
                                              p=num_players, t=num_radial_ticks)
    pred_untransformed_label_per_player = torch.argmax(pred_untransformed_per_player, -1)
    accuracy_per_player = (Y_label_per_player == pred_untransformed_label_per_player).type(torch.float)
    Y_valid_per_player_row = Y_per_player.sum(axis=2)
    masked_accuracy_per_player = accuracy_per_player * Y_valid_per_player_row

    # compute delta diffs
    #Y_delta_indices = get_delta_indices_from_grid(Y_label_per_player)
    #pred_untransformed_delta_indices = get_delta_indices_from_grid(pred_untransformed_label_per_player)
    Y_delta_pos = get_delta_pos_from_radial(Y_label_per_player, stature_to_speed)
    pred_untransformed_delta_pos = \
        get_delta_pos_from_radial(pred_untransformed_label_per_player, stature_to_speed)
    delta_diff_xy_per_player = \
        torch.sqrt(torch.sum(torch.pow(Y_delta_pos.delta_pos - pred_untransformed_delta_pos.delta_pos, 2), dim=-1))
    #torch.sqrt(
    #    torch.pow(Y_delta_indices.x_index - pred_untransformed_delta_indices.x_index, 2) +
    #    torch.pow(Y_delta_indices.y_index - pred_untransformed_delta_indices.y_index, 2)
    #)
    masked_delta_diff_xy = delta_diff_xy_per_player * Y_valid_per_player_row
    # 45 is reasonable jump height, so use that as z distance
    Y_delta_pos_with_z = Y_delta_pos.delta_pos.clone()
    Y_delta_pos_with_z[:, :, 2] = Y_delta_pos.z_jump_index * 45.
    pred_untransformed_delta_pos_with_z = pred_untransformed_delta_pos.delta_pos.clone()
    pred_untransformed_delta_pos_with_z[:, :, 2] = pred_untransformed_delta_pos.z_jump_index * 45.
    delta_diff_xyz_per_player = \
        torch.sqrt(torch.sum(torch.pow(Y_delta_pos_with_z - pred_untransformed_delta_pos_with_z, 2), dim=-1))
    #    torch.sqrt(
    #    torch.pow(Y_delta_indices.x_index - pred_untransformed_delta_indices.x_index, 2) +
    #    torch.pow(Y_delta_indices.y_index - pred_untransformed_delta_indices.y_index, 2) +
    #    torch.pow(Y_delta_indices.z_jump_index - pred_untransformed_delta_indices.z_jump_index, 2)
    #)
    masked_delta_diff_xyz = delta_diff_xyz_per_player * Y_valid_per_player_row

    if name not in accuracy:
        accuracy[name] = torch.zeros([1]).to(CUDA_DEVICE_STR)
        delta_diff_xy[name] = torch.zeros([1]).to(CUDA_DEVICE_STR)
        delta_diff_xyz[name] = torch.zeros([1]).to(CUDA_DEVICE_STR)
        valids_per_accuracy_column[name] = torch.zeros([1]).to(CUDA_DEVICE_STR)
        accuracy[name + duplicated_name_str] = torch.zeros([1]).to(CUDA_DEVICE_STR)
        delta_diff_xy[name + duplicated_name_str] = torch.zeros([1]).to(CUDA_DEVICE_STR)
        delta_diff_xyz[name + duplicated_name_str] = torch.zeros([1]).to(CUDA_DEVICE_STR)
        valids_per_accuracy_column[name + duplicated_name_str] = torch.zeros([1]).to(CUDA_DEVICE_STR)
    accuracy[name] += masked_accuracy_per_player[~duplicated_last].sum()
    delta_diff_xy[name] += masked_delta_diff_xy[~duplicated_last].sum()
    delta_diff_xyz[name] += masked_delta_diff_xyz[~duplicated_last].sum()
    valids_per_accuracy_column[name] += Y_valid_per_player_row[~duplicated_last].sum()
    accuracy[name + duplicated_name_str] += masked_accuracy_per_player[duplicated_last].sum()
    delta_diff_xy[name + duplicated_name_str] += masked_delta_diff_xy[duplicated_last].sum()
    delta_diff_xyz[name + duplicated_name_str] += masked_delta_diff_xyz[duplicated_last].sum()
    valids_per_accuracy_column[name + duplicated_name_str] += Y_valid_per_player_row[duplicated_last].sum()


def finish_accuracy_and_delta_diff(accuracy, delta_diff_xy, delta_diff_xyz, valids_per_accuracy_column,
                                   column_transformers: IOColumnTransformers):
    accuracy_string = ""
    for name, unadjusted_r in zip(column_transformers.output_types.column_names(True),
                                  column_transformers.get_name_ranges(False, False)):
        if name not in accuracy:
            continue
        elif name in column_transformers.output_types.column_names_all_categorical_columns():
            if valids_per_accuracy_column[name] > 0:
                accuracy_string += f"{name}: {accuracy[name]} % cat top 1 acc {delta_diff_xy[name]} delta diff xy " \
                                   f"{delta_diff_xyz[name]} delta diff xyz"
            else:
                accuracy_string += f'''{name}: no valids % cat top 1 acc'''
        accuracy_string += "; "
    return accuracy_string
