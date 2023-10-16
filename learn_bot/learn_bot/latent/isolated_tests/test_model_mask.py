import torch
from einops import rearrange
from torch.utils.data import DataLoader

from learn_bot.latent.load_model import load_model_file
from learn_bot.latent.place_area.load_data import LoadDataResult
from learn_bot.latent.place_area.simulator import limit_to_every_nth_row
from learn_bot.latent.train import load_data_options
from learn_bot.libs.io_transforms import CPU_DEVICE_STR, get_transformed_outputs

temperature = torch.Tensor([1.])
model_names = [
    "10_15_2023__21_33_23_iw_128_bc_25_pr_0_fr_0_b_1024_it_1_lr_4e-05_wd_0.0_l_2_h_4_n_20.0_ros_2.0_m_NoMask_w_None_dh_None_c_just_human_all",
    "10_15_2023__21_33_23_iw_128_bc_25_pr_0_fr_0_b_1024_it_1_lr_4e-05_wd_0.0_l_2_h_4_n_20.0_ros_2.0_m_EveryoneFullMask_w_None_dh_None_c_just_human_all",
    "10_15_2023__21_33_23_iw_128_bc_25_pr_0_fr_0_b_1024_it_1_lr_4e-05_wd_0.0_l_2_h_4_n_20.0_ros_2.0_m_TeammateFullMask_w_None_dh_None_c_just_human_all"
]


def test_model_mask(load_data_result: LoadDataResult, model_name: str):
    print(f"{model_name}")
    loaded_model = load_model_file(load_data_result, use_test_data_only=True, model_name_override=model_name)
    dataloader = DataLoader(loaded_model.cur_dataset, batch_size=256, shuffle=False)
    loaded_model.model.eval()
    loaded_model.model.to(CPU_DEVICE_STR)
    with torch.no_grad():
        for batch, (X, Y, similarity) in enumerate(dataloader):
            X_orig = X[[0]]
            X_mod = X_orig.clone()
            X_mod[0, loaded_model.model.players_pos_columns[0]] += 100.
            similarity_one = similarity[[0]]
            pred_orig = get_transformed_outputs(loaded_model.model(X_orig, similarity_one, temperature))
            pred_mod = get_transformed_outputs(loaded_model.model(X_mod, similarity_one, temperature))
            equal_tokens = torch.sum(pred_orig == pred_mod)
            num_players = loaded_model.model.num_players
            tokens_per_player = pred_orig.shape[2] * pred_orig.shape[3]
            equal_players = equal_tokens // tokens_per_player
            masked_all_players = equal_players == num_players - 1
            masked_teammates = (equal_players == num_players // 2) and \
                               torch.equal(pred_orig[0, num_players // 2, 0], pred_mod[0, num_players // 2, 0])
            masked_enemies = (equal_players == num_players // 2) and \
                               torch.equal(pred_orig[0, 0, 0], pred_mod[0, 0, 0])
            print(f'masked all players {masked_all_players}, masked teammates {masked_teammates}, '
                  f'masked enemies {masked_enemies}')
            break


if __name__ == "__main__":
    load_data_options.custom_limit_fn = limit_to_every_nth_row
    load_data_result = LoadDataResult(load_data_options)

    for model_name in model_names:
        test_model_mask(load_data_result, model_name)
