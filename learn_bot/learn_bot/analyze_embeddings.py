from torch import nn
from learn_bot.model import *
from learn_bot.dataset import BotDatasetArgs, BotDataset
from joblib import load
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict
from time import sleep
import pandas as pd
import numpy as np
import re
import sys
from io import StringIO
import seaborn as sns

nn_args: NNArgs = load(Path(__file__).parent / '..' / 'model' / 'nn_args.joblib')
device = "cpu"
model = NeuralNetwork(nn_args).to(device)
model.load_state_dict(torch.load(Path(__file__).parent / '..' / 'model' / 'model.pt', map_location=torch.device('cpu')))
skills_df: pd.DataFrame = pd.read_csv(Path(__file__).parent / '..' / 'data' / 'train_skills.csv')
skills_df = skills_df.replace({'player_id': nn_args.player_id_to_ix})
player_embeddings_df = pd.DataFrame(model.embeddings.weight.detach().clone().numpy())
player_embeddings_df.columns = ['embedding ' + str(c) for c in player_embeddings_df.columns]
skills_and_embeddings_df = pd.concat([skills_df, player_embeddings_df], axis=1).reindex(skills_df.index)

rows = skills_df.columns[2:]
columns = player_embeddings_df.columns
fig, ax = plt.subplots(nrows=len(rows), ncols=len(columns), figsize=(len(columns)*8, len(rows)*8))

for r, skill_col_name in enumerate(rows):
    for c, embedding_col_name in enumerate(columns):
        z = pd.qcut(skills_and_embeddings_df.loc[:, skill_col_name], 10, duplicates='drop')
        ax_df = skills_and_embeddings_df.copy()
        if len(ax_df.loc[:, skill_col_name].unique()) > 10:
            ax_df.loc[:, skill_col_name] = pd.qcut(ax_df.loc[:, skill_col_name], 10, duplicates='drop')
        ax_df.loc[:, embedding_col_name] = pd.qcut(ax_df.loc[:, embedding_col_name], 10)
        ax_df = ax_df.loc[:, [embedding_col_name, skill_col_name]]
        ax_crosstab = pd.crosstab(ax_df.loc[:, embedding_col_name], ax_df.loc[:, skill_col_name])
        sns.heatmap(ax_crosstab, annot=True, ax=ax[r][c])
        ax[r][c].set_xlabel(embedding_col_name, fontsize=14)
        ax[r][c].set_xlabel(skill_col_name, fontsize=14)
        ax[r][c].set_title(embedding_col_name + ' vs ' + skill_col_name, fontsize=14)


plt.suptitle("Skill Parameters vs Embeddings", fontsize=30)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig(Path(__file__).parent / '..' / 'plots' / 'embeddings_scatters.png')
