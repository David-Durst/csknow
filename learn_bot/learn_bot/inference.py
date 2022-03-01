from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from learn_bot.model import *
from joblib import load
from pathlib import Path
from typing import Dict
import pandas as pd
import re

# load model and arguments
args: NNArgs = load(Path(__file__).parent / '..' / 'model' / 'args.joblib')
device = "cpu"
model = NeuralNetwork(args).to(device)
model.load_state_dict(Path(__file__).parent / '..' / 'model' / 'model.pt')

def infer(inference_data: Dict):
    inference_df = pd.from_dict(inference_data)
    # hack, will fix later when actually storing player ids
    inference_df['player_id'] = 0
    dataloader = DataLoader(inference_df, batch_size=len(inference_df))

    with torch.no_grad():
        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            nav_target_idx = pred[:, args.output_ranges[0]].argmax(1)
            nav_targets = [feature_name for feature_name in args.output_ct.get_feature_names_out()
                           if "nav target" in feature_name]
            nav_target_name = nav_targets[nav_target_idx]
            return int(re.search(r'\d+', nav_target_name).group())


