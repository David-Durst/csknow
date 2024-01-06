from dataclasses import dataclass
import time
from pathlib import Path
from typing import List, Callable, Optional, Dict

import pandas as pd

from learn_bot.latent.analyze.comparison_column_names import predicted_trace_batch_col, \
    best_fit_ground_truth_round_id_col, predicted_round_id_col, best_match_id_col, metric_type_col, \
    all_human_vs_human_28_similarity_hdf5_data_path
from learn_bot.latent.engagement.column_names import game_id_column, round_id_column, tick_id_column
from learn_bot.latent.place_area.column_names import hdf5_id_columns, test_success_col, get_similarity_column, \
    vis_only_columns, default_similarity_columns, get_tick_similarity_column
from learn_bot.latent.place_area.create_test_data import create_zeros_train_data, create_similarity_data
from learn_bot.latent.place_area.push_save_label import PushSaveRoundLabels
from learn_bot.latent.train_paths import default_save_push_round_labels_path
from learn_bot.libs.hdf5_to_pd import load_hdf5_to_pd
from learn_bot.libs.hdf5_wrapper import HDF5Wrapper, PDWrapper
from learn_bot.libs.multi_hdf5_wrapper import MultiHDF5Wrapper, HDF5SourceOptions

just_test_comment = "just_test"
just_bot_comment = "just_bot"
bot_and_human_comment = "bot_and_human"
just_small_human_comment = "just_small_human"
just_human_comment = "just_human"
comment_28 = "_28"
all_comment = "_all"
human_with_bot_nav_added_comment = "human_with_added_bot_nav"
limited_comment = "_limited"
curriculum_comment = "_curriculum"

all_train_latent_team_hdf5_dir_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'analytics' / 'all_train_outputs'
human_28_latent_team_hdf5_dir_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'analytics' / 'all_train_outputs' / 'behaviorTreeTeamFeatureStore_28.hdf5'
human_latent_team_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'analytics' / 'csv_outputs' / 'behaviorTreeTeamFeatureStore.hdf5'
small_latent_team_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'analytics' / 'csv_outputs' / 'smallBehaviorTreeTeamFeatureStore.parquet'
manual_latent_team_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'analytics' / 'manual_outputs' / 'behaviorTreeTeamFeatureStore.hdf5'
rollout_latent_team_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'analytics' / 'rollout_outputs' / 'behaviorTreeTeamFeatureStore.hdf5'

SimilarityFn = Callable[[pd.DataFrame], pd.Series]


@dataclass
class LoadDataOptions:
    use_manual_data: bool
    use_rollout_data: bool
    use_synthetic_data: bool
    use_small_human_data: bool
    use_human_28_data: bool
    use_all_human_data: bool
    add_manual_to_all_human_data: bool
    limit_manual_data_to_no_enemies_nav: bool
    # set these for similarity columns
    # set these and limit_by_similarity if want to filter all human based on small human
    load_similarity_data: bool = False
    #hand_labeled_push_round_ids: Optional[List[List[int]]] = None
    #similarity_dfs: Optional[List[pd.DataFrame]] = None
    # limit or add feature based on matches
    limit_manual_data_to_only_enemies_no_nav: bool = False
    # train test split file name
    train_test_split_file_name: Optional[str] = None
    # custom rollout data
    custom_rollout_extension: Optional[str] = None
    # custom limit fn to be used for all human data
    custom_limit_fn: Optional[SimilarityFn] = None
    # similarity analysis
    # typically fix similarity by making no matches 0.5 and taking label if available
    # when doing analysis, mark no match as -1 and always take label of a different round (leave one out)
    similarity_analysis: bool = False


class LoadDataResult:
    diff_train_test: bool
    dataset_comment: str
    multi_hdf5_wrapper: MultiHDF5Wrapper

    def __init__(self, load_data_options: LoadDataOptions):
        self.diff_train_test = True
        force_test_data = None
        hdf5_sources: List[HDF5SourceOptions] = []
        custom_limit_fns = []
        duplicate_last_hdf5_equal_to_rest = False
        if load_data_options.use_manual_data:
            self.dataset_comment = just_bot_comment
            manual_data = HDF5Wrapper(manual_latent_team_hdf5_data_path, hdf5_id_columns)
            if load_data_options.limit_manual_data_to_no_enemies_nav:
                manual_data.limit((manual_data.id_df[test_success_col] == 1.) & (manual_data.id_df[game_id_column] == 1))
            elif load_data_options.limit_manual_data_to_only_enemies_no_nav:
                manual_data.limit((manual_data.id_df[test_success_col] == 1.) & (manual_data.id_df[game_id_column] == 0))
            else:
                manual_data.limit(manual_data.id_df[test_success_col] == 1.)
            hdf5_sources.append(manual_data)
        elif load_data_options.use_rollout_data:
            self.dataset_comment = "rollout"
            hdf5_path = rollout_latent_team_hdf5_data_path
            if load_data_options.custom_rollout_extension is not None:
                if '*' in load_data_options.custom_rollout_extension:
                    hdf5_sources += [hdf5_file for hdf5_file in
                                     hdf5_path.parent.glob('behaviorTreeTeamFeatureStore' +
                                                           load_data_options.custom_rollout_extension + '.hdf5')]
                else:
                    hdf5_path = hdf5_path.parent / (hdf5_path.stem + load_data_options.custom_rollout_extension +
                                                    hdf5_path.suffix)
                    rollout_data = HDF5Wrapper(hdf5_path, hdf5_id_columns, vis_cols=vis_only_columns)
                    hdf5_sources.append(rollout_data)
            else:
                raise Exception('rollout data always requires a custom extension now, not using default path')
            self.diff_train_test = False
        elif load_data_options.use_synthetic_data:
            self.dataset_comment = just_test_comment
            base_data = load_hdf5_to_pd(manual_latent_team_hdf5_data_path, rows_to_get=[1])
            #tmp_synthetic_data_df = create_left_right_train_data(base_data)
            #synthetic_data = PDWrapper('train', synthetic_data_df, hdf5_id_columns)
            #force_test_data_df = create_left_right_test_data(base_data)
            #force_test_data = PDWrapper('test', force_test_data_df, hdf5_id_columns)
            synthetic_data_df = create_zeros_train_data(base_data)
            similarity_data = create_similarity_data()
            synthetic_data = PDWrapper('train', synthetic_data_df, hdf5_id_columns)
            force_test_data_df = create_zeros_train_data(base_data)
            force_test_data = PDWrapper('test', force_test_data_df, hdf5_id_columns)
            for c in similarity_data.columns:
                synthetic_data.add_extra_column(c, similarity_data.loc[:, c])
                force_test_data.add_extra_column(c, similarity_data.loc[:, c])
            self.diff_train_test = False
            hdf5_sources.append(synthetic_data)
        elif load_data_options.use_small_human_data:
            self.dataset_comment = just_human_comment + limited_comment + "_unfilitered"
            human_data = HDF5Wrapper(human_latent_team_hdf5_data_path, ['id', round_id_column, test_success_col])
            #with open(good_retake_rounds_path, "r") as f:
            #    good_retake_rounds = eval(f.read())
            #human_data.limit(human_data.id_df[round_id_column].isin(good_retake_rounds))
            hdf5_sources.append(human_data)
        elif load_data_options.use_human_28_data:
            self.dataset_comment = just_human_comment + comment_28
            hdf5_sources.append(human_28_latent_team_hdf5_dir_path)
        elif load_data_options.use_all_human_data:
            self.dataset_comment = just_human_comment + all_comment
            hdf5_sources.append(all_train_latent_team_hdf5_dir_path)
            # NEED WAY TO RESTRICT TO GOOD ROUNDS
            if load_data_options.add_manual_to_all_human_data:
                self.dataset_comment = human_with_bot_nav_added_comment
                manual_data = HDF5Wrapper(manual_latent_team_hdf5_data_path, hdf5_id_columns)
                if load_data_options.limit_manual_data_to_no_enemies_nav:
                    manual_data.limit((manual_data.id_df[test_success_col] == 1.) & (manual_data.id_df[game_id_column] == 1))
                else:
                    manual_data.limit(manual_data.id_df[test_success_col] == 1.)
                hdf5_sources.append(manual_data)
                duplicate_last_hdf5_equal_to_rest = True
        else:
            raise Exception("Must call LoadDataResult with something True")
        self.multi_hdf5_wrapper = MultiHDF5Wrapper(hdf5_sources, hdf5_id_columns, diff_train_test=self.diff_train_test,
                                                   force_test_hdf5=force_test_data,
                                                   duplicate_last_hdf5_equal_to_rest=duplicate_last_hdf5_equal_to_rest,
                                                   train_test_split_file_name=load_data_options.train_test_split_file_name,
                                                   vis_cols=vis_only_columns)
        # load similarity
        if not load_data_options.use_synthetic_data and load_data_options.load_similarity_data:
            start_similarity_time = time.time()
            similarity_dfs = [load_hdf5_to_pd(all_human_vs_human_28_similarity_hdf5_data_path)]
            hand_labeled_push_round_ids = [PushSaveRoundLabels(default_save_push_round_labels_path)]
            for i in range(len(similarity_dfs)):
                self.load_similarity_columns_from_hand_labeled_push_rounds(
                    similarity_dfs[i], hand_labeled_push_round_ids[i],
                    i, load_data_options.similarity_analysis)
                self.generate_tick_similarity_columns(i)
            print(f"similarity load time {time.time() - start_similarity_time}")
        else:
            self.fill_empty_similarity_columns()
        if load_data_options.use_all_human_data and load_data_options.custom_limit_fn is not None:
            for _ in self.multi_hdf5_wrapper.hdf5_wrappers:
                custom_limit_fns.append(load_data_options.custom_limit_fn)
            self.limit(custom_limit_fns)
        self.multi_hdf5_wrapper.train_test_split_by_col(force_test_data)

    def limit(self, limit_fns: List[Optional[SimilarityFn]]):
        # note: need to redo train test split if applying limit later and want to use train-test splits with
        # updated filter
        for i in range(len(limit_fns)):
            if limit_fns[i] is None:
                continue
            self.multi_hdf5_wrapper.hdf5_wrappers[i].limit(limit_fns[i](self.multi_hdf5_wrapper.hdf5_wrappers[i].id_df))

    def add_column(self, add_column_fns: List[Optional[SimilarityFn]], column_name: str):
        # note: need to redo train test split if applying limit later and want to use train-test splits with
        # updated filter
        for i in range(len(add_column_fns)):
            if add_column_fns[i] is None:
                continue
            self.multi_hdf5_wrapper.hdf5_wrappers[i].add_extra_column(column_name,
                                                                      add_column_fns[i](self.multi_hdf5_wrapper.hdf5_wrappers[i].id_df))

    def load_similarity_columns_from_hand_labeled_push_rounds(self, similarity_df: pd.DataFrame,
                                                              hand_labeled_push_rounds: PushSaveRoundLabels,
                                                              similarity_index: int, similarity_analysis: bool):
        similarity_col = get_similarity_column(similarity_index)
        # build dataframe from hand-labeled round id to similarity score: 1 if push, 0 if save,
        # float for start push and end save
        push_round_ids_and_percents: List[Dict] = []
        for push_round_id, push_save_round_data in hand_labeled_push_rounds.round_id_to_data.items():
            push_round_ids_and_percents.append({
                best_fit_ground_truth_round_id_col: push_round_id,
                similarity_col: push_save_round_data.to_float_label()
            })
        push_round_ids_and_percents_df = pd.DataFrame.from_records(push_round_ids_and_percents)

        if similarity_analysis:
            best_match_similarity_df = similarity_df[((similarity_df[metric_type_col] == b'Unconstrained DTW') |
                                                      (similarity_df[metric_type_col] == b'Slope Constrained DTW'))]
        else:
            best_match_similarity_df = similarity_df[(similarity_df[best_match_id_col] == 0) &
                                                     ((similarity_df[metric_type_col] == b'Unconstrained DTW') |
                                                      (similarity_df[metric_type_col] == b'Slope Constrained DTW'))]
        # don't worry about merging reording index, verified that same length data, but merge does it's own sort and
        # index according to sort
        # unmerged_tmp = best_match_similarity_df.copy()
        best_match_similarity_df = \
            best_match_similarity_df.merge(push_round_ids_and_percents_df, how='inner', on=best_fit_ground_truth_round_id_col)
        # unsorted_tmp = best_match_similarity_df.copy()
        best_match_similarity_df.sort_values([predicted_trace_batch_col, predicted_round_id_col, metric_type_col,
                                              best_match_id_col], inplace=True)

        #for idx, row in best_match_similarity_df.iterrows():
        #    hdf5_filename = row[predicted_trace_batch_col].decode('utf-8')
        #    if hdf5_filename not in hdf5_to_push_round_ids:
        #        hdf5_to_push_round_ids[hdf5_filename] = []
        #    if row[best_fit_ground_truth_round_id_col] in hand_labeled_push_rounds.round_id_to_data:
        #        hdf5_to_push_round_ids[hdf5_filename].append(row[predicted_round_id_col])

        similarity_fns: List[Optional[SimilarityFn]] = []
        total_rounds = 0
        num_rounds_not_matched = 0
        for i, hdf5_wrapper in enumerate(self.multi_hdf5_wrapper.hdf5_wrappers):
            # get best match similarity for this hdf5
            hdf5_round_id_and_similarity = \
                best_match_similarity_df[best_match_similarity_df[predicted_trace_batch_col].str.decode('utf-8') ==
                                         str(hdf5_wrapper.hdf5_path.name)].loc[:, [predicted_round_id_col, best_fit_ground_truth_round_id_col,
                                                                                   similarity_col, metric_type_col, best_match_id_col]] # for debugging

            # compute similarity per round
            hdf5_round_id_to_similarity_dict = {}
            # only to use with analysis, where can skip a predicted round id because it equals ground truth round id
            predicted_round_ids_matched = []
            for _, row in hdf5_round_id_and_similarity.iterrows():
                # since Slope constrained comes first, skip if already present to prevent overwrite
                predicted_round_id = row[predicted_round_id_col]
                if predicted_round_id not in hdf5_round_id_to_similarity_dict:
                    # if doing analysis, want match that isn't the same (and only looking at same hdf5 file, so can just check round id)
                    if similarity_analysis and predicted_round_id == row[best_fit_ground_truth_round_id_col]:
                        continue
                    hdf5_round_id_to_similarity_dict[predicted_round_id] = row[similarity_col]
                    predicted_round_ids_matched.append(predicted_round_id)

            # fill in similarity for rounds that lack a label
            round_ids_in_hdf5 = hdf5_wrapper.id_df[round_id_column].unique()
            similarity_round_ids = hdf5_round_id_and_similarity[predicted_round_id_col].unique()
            if similarity_analysis:
                round_ids_not_matched = [r for r in round_ids_in_hdf5 if r not in predicted_round_ids_matched]
            else:
                round_ids_not_matched = [r for r in round_ids_in_hdf5 if r not in similarity_round_ids]
            #print(f"{hdf5_wrapper.hdf5_path.name}: not matched round ids: {round_ids_not_matched}")
            total_rounds += len(round_ids_in_hdf5)
            num_rounds_not_matched += len(round_ids_not_matched)
            for r in round_ids_not_matched:
                if similarity_analysis:
                    hdf5_round_id_to_similarity_dict[r] = -1.
                else:
                    hdf5_round_id_to_similarity_dict[r] = 0.5

            # make functions for generating round-based similarity data per tick
            # https://stackoverflow.com/questions/2295290/what-do-lambda-function-closures-capture
            similarity_fns.append(lambda df, round_id_to_similarity_dict=hdf5_round_id_to_similarity_dict: df[round_id_column].map(round_id_to_similarity_dict))

        #print(f"num non matched rounds: {num_rounds_not_matched} / {total_rounds}, {num_rounds_not_matched / total_rounds:.3f}")
        self.add_column(similarity_fns, get_similarity_column(similarity_index))

    def generate_tick_similarity_columns(self, similarity_index: int):
        similarity_col = get_similarity_column(similarity_index)
        tick_similarity_col = get_tick_similarity_column(similarity_index)

        for i, hdf5_wrapper in enumerate(self.multi_hdf5_wrapper.hdf5_wrappers):
            id_df = hdf5_wrapper.id_df.copy()
            round_ids_similarity_first_last_tick_id_df = \
                id_df.groupby(round_id_column, as_index=False).agg(
                    first_tick=(tick_id_column, 'first'),
                    last_tick=(tick_id_column, 'last'),
                    num_ticks=(tick_id_column, 'count')
                )
            for _, round_row in round_ids_similarity_first_last_tick_id_df.iterrows():
                if round_row['num_ticks'] != 1 + round_row['last_tick'] - round_row['first_tick']:
                    print('missing tick in round')
            id_df = id_df.merge(round_ids_similarity_first_last_tick_id_df, on=round_id_column)
            fraction_of_round = (id_df[tick_id_column] - id_df['first_tick']) / \
                                (1 + id_df['last_tick'] - id_df['first_tick'])
            hdf5_wrapper.id_df[tick_similarity_col] = fraction_of_round <= id_df[similarity_col]

    def fill_empty_similarity_columns(self):
        for similarity_index in range(default_similarity_columns):
            similarity_fns: List[Optional[SimilarityFn]] = []
            for i, hdf5_wrapper in enumerate(self.multi_hdf5_wrapper.hdf5_wrappers):
                similarity_fns.append(lambda df: df[round_id_column] != df[round_id_column])
            self.add_column(similarity_fns, get_similarity_column(similarity_index))
            self.add_column(similarity_fns, get_tick_similarity_column(similarity_index))
