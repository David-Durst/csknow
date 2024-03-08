import glob
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pypandoc
from matplotlib import pyplot as plt
from scikit_posthocs import posthoc_dunn
from scipy import stats

from learn_bot.latent.analyze.color_lib import default_bar_color
from learn_bot.latent.analyze.user_responses.build_trueskill import build_trueskill
from learn_bot.latent.analyze.user_responses.user_constants import rank_map, answer_key, file_name_col, rank_col, \
    years_exp_col, example_col, high_skill_col, user_study_plots, bot_types, rank_reverse_map
from learn_bot.libs.pd_printing import set_pd_print_options

anonymized_user_responses = Path("/home/durst/anonymized_user_response")


def process_user_responses():
    answers: List[Dict] = []
    anonymized_user_responses.mkdir(parents=True, exist_ok=True)

    for i, f in enumerate(glob.glob("/home/durst/user_responses/*")):
        if 'isaac' in f:
            continue

        print(f)
        file_txt = pypandoc.convert_file(f, 'plain')
        anonymized_file_txt = file_txt.replace("durst@stanford.edu", "<anonymized>")
        with open(anonymized_user_responses / f"response_{i}.txt", 'w') as g:
            g.write(anonymized_file_txt)
        assert 'durst' not in anonymized_file_txt.lower()
        assert 'david' not in anonymized_file_txt.lower()
        assert 'stanford' not in anonymized_file_txt.lower()

        if 'tc_zhang' in f:
            rank_answer = re.search(r'Experience Question 1:\W*_*([a-zA-Z0-9]+)_*.*\n', file_txt)  #
            years_dev_answer = re.search(r'\nExperience Question 2:\W*_*([0-9]+).*\n', file_txt)  #
        else:
            rank_answer = re.search(r'\nExperience Answer 1:\W*_*([a-zA-Z0-9]+)_*.*\n', file_txt)  #
            years_dev_answer = re.search(r'\nExperience Answer 2:\W*_*([0-9]+).*\n', file_txt)  #
        rank = rank_map[rank_answer.group(1)]
        years_dev = int(years_dev_answer.group(1))

        def get_example_n_anwser(n: int) -> str:
            if 'tc_zhang' in f:
                return r'\nExample ' + str(n + 1) + r':\W*_*([\d, ]+).*\n'
            else:
                return r'\nExample ' + str(n+1) + r' Movement Answer:\W*_*([\d, ]+).*\n'

        for i in range(8):
            example_answer = re.search(get_example_n_anwser(i), file_txt)  #
            example_ints = [int(i) for i in example_answer.group(1).split(',')[:4]]
            answer = {rank_col: rank, years_exp_col: years_dev, example_col: i + 1, file_name_col: Path(f).name}
            for humanness_index, example_int in enumerate(example_ints):
                answer[answer_key[i][example_int - 1]] = humanness_index + 1
            answers.append(answer)

    df = pd.DataFrame.from_records(answers)
    df.sort_values(['File', 'Example'], ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f'num responses: {len(df) // 8}')
    print(f'human most human: {sum(df["human"] > 1.5) / len(df["human"])}')
    user_study_plots.mkdir(parents=True, exist_ok=True)

    df[high_skill_col] = df[rank_col] >= rank_map['SMFC']

    significance_tests(df, user_study_plots / 'significance.txt')

    plot_mean_by_rank(df)
    plot_count_by_rank(df)

    set_pd_print_options()
    print(df.describe())

    ratings = build_trueskill(df)
    ratings.plot(user_study_plots / 'trueskill.pdf')

    low_skill_df = df[~df[high_skill_col]]
    low_skill_ratings = build_trueskill(low_skill_df)
    print(len(low_skill_df))
    low_skill_ratings.plot(user_study_plots / 'low_skill_trueskill.pdf', 'Low Skill')

    high_skill_df = df[df[high_skill_col]]
    high_skill_ratings = build_trueskill(high_skill_df)
    print(len(high_skill_df))
    high_skill_ratings.plot(user_study_plots / 'high_skill_trueskill.pdf', 'High Skill')


def plot_mean_by_rank(df: pd.DataFrame):
    fig_length = 6
    fig = plt.figure(figsize=(fig_length * 3, fig_length), constrained_layout=True)
    ax = fig.subplots()

    mean_by_high_skill_df = df.groupby(high_skill_col, as_index=False).mean(numeric_only=True)

    mean_by_high_skill_df.plot.bar(x=high_skill_col, y=bot_types, ax=ax)

    plt.savefig(user_study_plots / 'mean_by_high_skill.pdf')


def plot_count_by_rank(df: pd.DataFrame):
    df = df.copy()
    fig = plt.figure(figsize=(3.3, 3.3), constrained_layout=True)
    ax = fig.subplots()

    num_per_rank = df.groupby(rank_col, as_index=False).count()
    num_per_rank[years_exp_col] /= 8

    num_per_rank.plot.bar(x=rank_col, y=years_exp_col, ax=ax, rot=0, color=default_bar_color)


    old_x_tick_labels = ax.get_xticklabels()
    x_tick_labels = [rank_reverse_map[int(i.get_text())] for i in old_x_tick_labels]
    ax.set_xticklabels(x_tick_labels)

    ax.get_legend().remove()

    ax.set_title('Number of Users By Rank', fontsize=8)
    ax.set_xlabel('Rank', fontsize=8)
    ax.set_ylabel('Users', fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    # remove right/top spine
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # remove veritcal grid lines, make horizontal dotted
    ax.yaxis.grid(True, color='#EEEEEE', dashes=[4, 1])
    ax.xaxis.grid(False)

    plt.savefig(user_study_plots / 'count_by_rank.pdf')


def significance_tests(df: pd.DataFrame, plot_path: Path):
    ratings_df = df.loc[:, ['human', 'learned', 'hand-crafted', 'default']]
    melted_ratings_df = ratings_df.melt()
    dunn = posthoc_dunn(melted_ratings_df, val_col='value', group_col='variable', p_adjust='holm')
    ratings_2d = [list(df['human']), list(df['learned']), list(df['hand-crafted']), list(df['default'])]
    kruskal = stats.kruskal(*ratings_2d)
    with plot_path.open("w") as f:
        f.write(f"kruskal: {set(kruskal)}\n")
        f.write(f"dunn: {str(dunn)}")


if __name__ == '__main__':
    process_user_responses()