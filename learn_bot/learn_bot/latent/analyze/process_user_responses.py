import glob
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pypandoc
from matplotlib import pyplot as plt

from learn_bot.libs.pd_printing import set_pd_print_options

rank_map = {
    'NA': 0,
    'S1': 1,
    'S2': 2,
    'S3': 3,
    'S4': 4,
    'SE': 5,
    'SEM': 6,
    'GN1': 7,
    'GN2': 8,
    'GN3': 9,
    'GNM': 10,
    'MG1': 11,
    'MG2': 12,
    'MGE': 13,
    'DMG': 14,
    'LE': 15,
    'LEM': 16,
    'SMFC': 17,
    'GE': 18,
    'Global': 18
}

bot_types = ['human', 'learned', 'hand-crafted', 'default']

answer_key = [
    ['human', 'default', 'learned', 'hand-crafted'],
    ['hand-crafted', 'human', 'learned', 'default'],
    ['learned', 'human', 'default', 'hand-crafted'],
    ['default', 'hand-crafted', 'learned', 'human'],
    ['human', 'default', 'learned', 'hand-crafted'],
    ['default', 'hand-crafted', 'human', 'learned'],
    ['learned', 'human', 'hand-crafted', 'default'],
    ['hand-crafted', 'learned', 'human', 'default']
]

file_name_col = 'File'
rank_col = 'Rank'
years_exp_col = 'Years of Development Experience'
example_col = 'Example'
user_study_plots = Path(__file__).parent / 'user_plots'

def process_user_responses():
    answers: List[Dict] = []
    for f in glob.glob("/home/durst/user_responses/*"):
        print(f)
        file_txt = pypandoc.convert_file(f, 'plain')
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
            answer = {rank_col: rank, years_exp_col: years_dev, example_col: i+1, file_name_col: Path(f).name}
            for humanness_index, example_int in enumerate(example_ints):
                answer[answer_key[i][example_int-1]] = humanness_index+1
            answers.append(answer)

    df = pd.DataFrame.from_records(answers)
    print(f'num responses: {len(df) // 8}')
    user_study_plots.mkdir(parents=True, exist_ok=True)

    fig_length = 6
    fig = plt.figure(figsize=(fig_length * 4, fig_length), constrained_layout=True)
    axs = fig.subplots(1, 4, squeeze=False)

    mean_by_exp = df.groupby(years_exp_col, as_index=False).mean(numeric_only=True)
    upper_iqr_by_exp = df.groupby(years_exp_col, as_index=False).quantile(0.75, numeric_only=True)
    lower_iqr_by_exp = df.groupby(years_exp_col, as_index=False).quantile(0.75, numeric_only=True)

    for i in range(4):
        mean_by_exp.plot.line(x=years_exp_col, y=bot_types[i], ax=axs[0, i])

    plt.savefig(user_study_plots / 'mean_iqr_by_experience.pdf')

    fig_length = 6
    fig = plt.figure(figsize=(fig_length * 4, fig_length), constrained_layout=True)
    axs = fig.subplots(1, 4, squeeze=False)


    mean_by_rank = df.groupby(rank_col, as_index=False).mean(numeric_only=True)
    upper_iqr_by_rank = df.groupby(rank_col, as_index=False).quantile(0.75, numeric_only=True)
    lower_iqr_by_rank = df.groupby(rank_col, as_index=False).quantile(0.75, numeric_only=True)

    for i in range(4):
        mean_by_rank.plot.line(x=rank_col, y=bot_types[i], ax=axs[0, i])

    plt.savefig(user_study_plots / 'median_iqr_by_rank.pdf')

    num_per_rank = df.groupby(rank_col, as_index=False).count()
    num_per_rank[years_exp_col] /= 8

    print(num_per_rank)
    set_pd_print_options()
    print(df.describe())


if __name__ == '__main__':
    process_user_responses()