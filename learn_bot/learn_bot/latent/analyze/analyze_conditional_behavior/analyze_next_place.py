from typing import List, Dict

from tqdm import tqdm

from learn_bot.latent.analyze.compare_trajectories.process_trajectory_comparison import get_hdf5_to_test_round_ids
from learn_bot.latent.analyze.plot_trajectory_heatmap.compute_teamwork_metrics import TeamPlaces
from learn_bot.latent.engagement.column_names import round_id_column
from learn_bot.latent.load_model import load_model_file
from learn_bot.latent.place_area.column_names import specific_player_place_area_columns
from learn_bot.latent.place_area.load_data import LoadDataResult
from learn_bot.latent.place_area.simulation.constants import place_names, place_name_to_index
from learn_bot.latent.vis import run_vis_checkpoint

t_on_a_site_places = set([place_name_to_index[s] for s in ["BombsiteA", "ExtendedA"]])
ct_a_site_from_spawn_long = set([place_name_to_index[s] for s in ["LongA", "CTSpawn", "UnderA"]])
ct_under_a_site_place = place_name_to_index["UnderA"]

t_away_cat_a_site_places = set([place_name_to_index[s] for s in ["BombsiteA", "ARamp", "Ramp"]])
ct_a_site_from_spawn_cat = set([place_name_to_index[s] for s in ["ExtendedA", "Short", "ShortStairs", "UnderA", "CTSpawn"]])
ct_extended_a_site_place = place_name_to_index["ExtendedA"]

t_on_b_site_places = set([place_name_to_index[s] for s in ["BombsiteB", "UpperTunnel"]])
ct_outside_b_site_place = place_name_to_index["BDoors"]

def analyze_next_place():
    load_data_options = run_vis_checkpoint.load_data_options
    load_data_result = LoadDataResult(load_data_options)
    loaded_model = load_model_file(load_data_result)
    hdf5_to_round_ids = get_hdf5_to_test_round_ids(push_only=True)[0]

    a_mistake_opportunities: set = set()
    a_mistakes: set = set()
    b_mistake_opportunities: set = set()
    with tqdm(total=len(loaded_model.dataset.data_hdf5s), disable=False) as pbar:
        for dataset_index, hdf5_wrapper in enumerate(loaded_model.dataset.data_hdf5s):
            #if i > 3:
            #    continue
            hdf5_key = str(hdf5_wrapper.hdf5_path.name)
            if hdf5_key not in hdf5_to_round_ids:
                #print(f'skipping {hdf5_key}')
                continue

            loaded_model.cur_hdf5_index = dataset_index
            loaded_model.load_cur_dataset_only(include_outputs=False)

            # get the data per hdf5
            id_df = loaded_model.get_cur_id_df()
            vis_df = loaded_model.get_cur_vis_df()
            dataset = loaded_model.cur_dataset.X

            # get trajectory identifies
            trajectory_ids = hdf5_to_round_ids[hdf5_key]
            trajectory_id_col = id_df[round_id_column]

            for trajectory_id in trajectory_ids:
                # get data per trajectory
                trajectory_np = dataset[trajectory_id_col == trajectory_id]
                trajectory_id_df = id_df[trajectory_id_col == trajectory_id]
                trajectory_vis_df = vis_df[trajectory_id_col == trajectory_id]

                # build the per tick team places
                team_to_tick_to_place: Dict[bool, List[TeamPlaces]] = {True: [], False: []}
                for ct_team in [True, False]:
                    if ct_team:
                        team_range = range(0, loaded_model.model.num_players_per_team)
                    else:
                        team_range = range(loaded_model.model.num_players_per_team, loaded_model.model.num_players)
                    planted_a_np = trajectory_np[:, loaded_model.model.c4_planted_columns[0]] > 0.5
                    place_columns = [specific_player_place_area_columns[i].place_index for i in team_range]
                    places_np = trajectory_vis_df.loc[:, place_columns].to_numpy()

                    for i, tick_places in enumerate(places_np.tolist()):
                        team_place = TeamPlaces(ct_team, planted_a_np[i],
                                                # filter out places of dead players
                                                [int(i) for i in tick_places if i < len(place_names)])
                        team_to_tick_to_place[ct_team].append(team_place)

                for j in range(len(trajectory_np) - 1):
                    cur_t_place = team_to_tick_to_place[False][j]
                    next_t_place = team_to_tick_to_place[False][j+1]
                    cur_ct_place = team_to_tick_to_place[True][j]

                    # cur_t_only_on_a = set(cur_t_place.places).issubset(t_on_a_site_places)
                    next_t_under_a = ct_under_a_site_place in next_t_place.places
                    # cur_ct_attack_a_spawn_long = set(cur_ct_place.places).issubset(ct_a_site_from_spawn_long)
                    cur_t_away_cat_a = set(cur_t_place.places).issubset(t_away_cat_a_site_places)
                    next_t_extended_a = ct_extended_a_site_place in next_t_place.places
                    cur_ct_attack_a_spawn_cat = set(cur_ct_place.places).isdisjoint(t_away_cat_a_site_places) and \
                                                ct_under_a_site_place in cur_ct_place.places

                    #cur_t_only_on_a = set(cur_t_place.places).issubset(t_on_a_site_places)
                    #next_t_under_a = ct_under_a_site_place in next_t_place.places
                    #cur_ct_under_a = ct_under_a_site_place in cur_ct_place.places

                    #if cur_t_only_on_a and cur_ct_under_a:
                    #    a_mistake_opportunities.add((dataset_index, hdf5_key, trajectory_id))
                    #if cur_t_only_on_a and cur_ct_under_a and next_t_under_a:
                    #    a_mistakes.add((dataset_index, hdf5_key, trajectory_id))
                    if cur_t_away_cat_a and cur_ct_attack_a_spawn_cat:
                        a_mistake_opportunities.add((dataset_index, hdf5_key, trajectory_id))
                    if cur_t_away_cat_a and next_t_under_a and cur_ct_attack_a_spawn_cat:
                        a_mistakes.add((dataset_index, hdf5_key, trajectory_id))

                    cur_t_only_on_b = set(cur_t_place.places).issubset(t_on_b_site_places)
                    cur_ct_outside_b = ct_outside_b_site_place in cur_ct_place.places

                    if cur_t_only_on_b and cur_ct_outside_b:
                        b_mistake_opportunities.add((dataset_index, hdf5_key, trajectory_id))

            pbar.update(1)
    print('opportunities')
    #print(len(a_mistake_opportunities))
    print(a_mistake_opportunities)
    print('mistakes')
    print(a_mistakes)
    print("")
    #print(b_mistake_opportunities)
    #print('mistakes')
    #print(a_mistakes)


if __name__ == '__main__':
    analyze_next_place()