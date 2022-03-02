from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from learn_bot.model import *
from learn_bot.dataset import BotDatasetArgs, BotDataset
from joblib import load
from pathlib import Path
from typing import Dict
import pandas as pd
import re

# load model and arguments
dataset_args: BotDatasetArgs = load(Path(__file__).parent / '..' / 'model' / 'dataset_args.joblib')
nn_args: NNArgs = load(Path(__file__).parent / '..' / 'model' / 'nn_args.joblib')
device = "cpu"
model = NeuralNetwork(nn_args).to(device)
model.load_state_dict(torch.load(Path(__file__).parent / '..' / 'model' / 'model.pt', map_location=torch.device('cpu')))

def infer(inference_dict: Dict):
    print([inference_dict])
    inference_df = pd.DataFrame.from_dict([inference_dict])
    # hack, will fix later when actually storing player ids
    inference_df['source player id'] = 0
    inference_data = BotDataset(inference_df, dataset_args, True)
    dataloader = DataLoader(inference_data, batch_size=len(inference_df))

    with torch.no_grad():
        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            nav_target_idx = pred[:, nn_args.output_ranges[0]].argmax(1)
            nav_targets = [feature_name for feature_name in nn_args.output_ct.get_feature_names_out()
                           if "nav target" in feature_name]
            nav_target_name = nav_targets[nav_target_idx]
            return int(re.search(r'\d+', nav_target_name).group())

#x = {'id': 1, 'tick id': 30, 'round id': 0, 'source player id': 1, 'team': 0, 'cur nav area': 7, 'cur pos x': 128, 'cur pos y': 0, 'cur pos z': 0, 'cur nav 0 friends': 0, 'cur nav 0 enemies': 0, 'cur nav 1 friends': 0, 'cur nav 1 enemies': 0, 'cur nav 2 friends': 0, 'cur nav 2 enemies': 0, 'cur nav 3 friends': 0, 'cur nav 3 enemies': 0, 'cur nav 4 friends': 0, 'cur nav 4 enemies': 0, 'cur nav 5 friends': 0, 'cur nav 5 enemies': 0, 'cur nav 6 friends': 0, 'cur nav 6 enemies': 1, 'cur nav 7 friends': 1, 'cur nav 7 enemies': 0, 'cur nav 8 friends': 0, 'cur nav 8 enemies': 0, 'cur nav 9 friends': 0, 'cur nav 9 enemies': 0, 'cur nav 10 friends': 0, 'cur nav 10 enemies': 0, 'last nav area': 7, 'last pos x': 128, 'last pos y': 0, 'last pos z': 0, 'last nav 0 friends': 0, 'last nav 0 enemies': 0, 'last nav 1 friends': 0, 'last nav 1 enemies': 0, 'last nav 2 friends': 0, 'last nav 2 enemies': 0, 'last nav 3 friends': 0, 'last nav 3 enemies': 0, 'last nav 4 friends': 0, 'last nav 4 enemies': 0, 'last nav 5 friends': 0, 'last nav 5 enemies': 0, 'last nav 6 friends': 0, 'last nav 6 enemies': 1, 'last nav 7 friends': 1, 'last nav 7 enemies': 0, 'last nav 8 friends': 0, 'last nav 8 enemies': 0, 'last nav 9 friends': 0, 'last nav 9 enemies': 0, 'last nav 10 friends': 0, 'last nav 10 enemies': 0, 'old nav area': 7, 'old pos x': 128, 'old pos y': 0, 'old pos z': 0, 'old nav 0 friends': 0, 'old nav 0 enemies': 0, 'old nav 1 friends': 0, 'old nav 1 enemies': 0, 'old nav 2 friends': 0, 'old nav 2 enemies': 0, 'old nav 3 friends': 0, 'old nav 3 enemies': 0, 'old nav 4 friends': 0, 'old nav 4 enemies': 0, 'old nav 5 friends': 0, 'old nav 5 enemies': 0, 'old nav 6 friends': 0, 'old nav 6 enemies': 1, 'old nav 7 friends': 1, 'old nav 7 enemies': 0, 'old nav 8 friends': 0, 'old nav 8 enemies': 0, 'old nav 9 friends': 0, 'old nav 9 enemies': 0, 'old nav 10 friends': 0, 'old nav 10 enemies': 0, 'delta x': 0, 'delta y': 0, 'shoot next': 0, 'crouch next': 0, 'nav target': 7}
#infer(x)