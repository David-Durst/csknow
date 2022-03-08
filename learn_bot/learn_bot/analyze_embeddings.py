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

nn_args: NNArgs = load(Path(__file__).parent / '..' / 'model' / 'nn_args.joblib')
device = "cpu"
model = NeuralNetwork(nn_args).to(device)
model.load_state_dict(torch.load(Path(__file__).parent / '..' / 'model' / 'model.pt', map_location=torch.device('cpu')))
skills_df: pd.DataFrame = pd.read_csv(Path(__file__).parent / '..' / 'data' / 'train_skills.csv')
skills_df = skills_df.replace({'player_id': nn_args.player_id_to_ix})
player_embeddings_df = pd.DataFrame(model.embeddings.weight.detach().clone().numpy())
player_embeddings_df.columns = ['embedding ' + str(c) for c in player_embeddings_df.columns]
skills_and_embeddings_df = pd.concat([skills_df, player_embeddings_df], axis=1).reindex(skills_df.index)

rows = len(skills_df.columns[1:])
columns = len(player_embeddings_df.columns)
fig, ax = plt.subplots(nrows=len(rows), ncols=len(columns), figsize=(len(columns)*8, len(rows)*8))

for skill_col_name in rows:
    for embedding_col_name in columns:

        if len(column_series.unique()) == 2:
            for column_value in column_series.unique():
                filtered_series = column_series[column_series == column_value]


        for player_index in range(len(skills_df)):
            x = 1

plt.suptitle("Skill Parameters vs Embeddings", fontsize=30)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig(plot_folder + 'hist_' + name.lower().replace(' ', '_') + '.png')
