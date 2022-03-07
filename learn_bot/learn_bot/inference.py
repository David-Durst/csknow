from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from learn_bot.model import *
from learn_bot.dataset import BotDatasetArgs, BotDataset
from joblib import load
from pathlib import Path
from typing import Dict
from time import sleep
import pandas as pd
import re
import sys
from io import StringIO

# load model and arguments
dataset_args: BotDatasetArgs = load(Path(__file__).parent / '..' / 'model' / 'dataset_args.joblib')
nn_args: NNArgs = load(Path(__file__).parent / '..' / 'model' / 'nn_args.joblib')
device = "cpu"
model = NeuralNetwork(nn_args).to(device)
model.load_state_dict(torch.load(Path(__file__).parent / '..' / 'model' / 'model.pt', map_location=torch.device('cpu')))

#def infer(inference_dict: Dict):
#    print([inference_dict])
#    print('hi')
def infer(inference_df: pd.DataFrame):
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
            nav_target_names = [nav_targets[idx] for idx in nav_target_idx]
            return [int(re.search(r'\d+', nav_target_name).group()) for nav_target_name in nav_target_names]


#x = {'id': 1, 'tick id': 30, 'round id': 0, 'source player id': 1, 'team': 0, 'cur nav area': 7, 'cur pos x': 128, 'cur pos y': 0, 'cur pos z': 0, 'cur nav 0 friends': 0, 'cur nav 0 enemies': 0, 'cur nav 1 friends': 0, 'cur nav 1 enemies': 0, 'cur nav 2 friends': 0, 'cur nav 2 enemies': 0, 'cur nav 3 friends': 0, 'cur nav 3 enemies': 0, 'cur nav 4 friends': 0, 'cur nav 4 enemies': 0, 'cur nav 5 friends': 0, 'cur nav 5 enemies': 0, 'cur nav 6 friends': 0, 'cur nav 6 enemies': 1, 'cur nav 7 friends': 1, 'cur nav 7 enemies': 0, 'cur nav 8 friends': 0, 'cur nav 8 enemies': 0, 'cur nav 9 friends': 0, 'cur nav 9 enemies': 0, 'cur nav 10 friends': 0, 'cur nav 10 enemies': 0, 'last nav area': 7, 'last pos x': 128, 'last pos y': 0, 'last pos z': 0, 'last nav 0 friends': 0, 'last nav 0 enemies': 0, 'last nav 1 friends': 0, 'last nav 1 enemies': 0, 'last nav 2 friends': 0, 'last nav 2 enemies': 0, 'last nav 3 friends': 0, 'last nav 3 enemies': 0, 'last nav 4 friends': 0, 'last nav 4 enemies': 0, 'last nav 5 friends': 0, 'last nav 5 enemies': 0, 'last nav 6 friends': 0, 'last nav 6 enemies': 1, 'last nav 7 friends': 1, 'last nav 7 enemies': 0, 'last nav 8 friends': 0, 'last nav 8 enemies': 0, 'last nav 9 friends': 0, 'last nav 9 enemies': 0, 'last nav 10 friends': 0, 'last nav 10 enemies': 0, 'old nav area': 7, 'old pos x': 128, 'old pos y': 0, 'old pos z': 0, 'old nav 0 friends': 0, 'old nav 0 enemies': 0, 'old nav 1 friends': 0, 'old nav 1 enemies': 0, 'old nav 2 friends': 0, 'old nav 2 enemies': 0, 'old nav 3 friends': 0, 'old nav 3 enemies': 0, 'old nav 4 friends': 0, 'old nav 4 enemies': 0, 'old nav 5 friends': 0, 'old nav 5 enemies': 0, 'old nav 6 friends': 0, 'old nav 6 enemies': 1, 'old nav 7 friends': 1, 'old nav 7 enemies': 0, 'old nav 8 friends': 0, 'old nav 8 enemies': 0, 'old nav 9 friends': 0, 'old nav 9 enemies': 0, 'old nav 10 friends': 0, 'old nav 10 enemies': 0, 'delta x': 0, 'delta y': 0, 'shoot next': 0, 'crouch next': 0, 'nav target': 7}
#x = {'id': 0, 'tick id': 0, 'round id': 0, 'source player id': 0, 'source player name': '', 'demo name': '', 'team': 3, 'cur nav area': 7, 'cur pos x': 128.0, 'cur pos y': 0.0, 'cur pos z': -63.96875, 'cur nav 0 friends': 0, 'cur nav 0 enemies': 0, 'cur nav 1 friends': 0, 'cur nav 1 enemies': 0, 'cur nav 2 friends': 0, 'cur nav 2 enemies': 0, 'cur nav 3 friends': 0, 'cur nav 3 enemies': 0, 'cur nav 4 friends': 0, 'cur nav 4 enemies': 0, 'cur nav 5 friends': 0, 'cur nav 5 enemies': 0, 'cur nav 6 friends': 0, 'cur nav 6 enemies': 1, 'cur nav 7 friends': 1, 'cur nav 7 enemies': 0, 'cur nav 8 friends': 0, 'cur nav 8 enemies': 0, 'cur nav 9 friends': 0, 'cur nav 9 enemies': 0, 'cur nav 10 friends': 0, 'cur nav 10 enemies': 0, 'last nav area': 7, 'last pos x': 128.0, 'last pos y': 0.0, 'last pos z': -63.96875, 'last nav 0 friends': 0, 'last nav 0 enemies': 0, 'last nav 1 friends': 0, 'last nav 1 enemies': 0, 'last nav 2 friends': 0, 'last nav 2 enemies': 0, 'last nav 3 friends': 0, 'last nav 3 enemies': 0, 'last nav 4 friends': 0, 'last nav 4 enemies': 0, 'last nav 5 friends': 0, 'last nav 5 enemies': 0, 'last nav 6 friends': 0, 'last nav 6 enemies': 1, 'last nav 7 friends': 1, 'last nav 7 enemies': 0, 'last nav 8 friends': 0, 'last nav 8 enemies': 0, 'last nav 9 friends': 0, 'last nav 9 enemies': 0, 'last nav 10 friends': 0, 'last nav 10 enemies': 0, 'old nav area': 7, 'old pos x': 128.0, 'old pos y': 0.0, 'old pos z': -63.96875, 'old nav 0 friends': 0, 'old nav 0 enemies': 0, 'old nav 1 friends': 0, 'old nav 1 enemies': 0, 'old nav 2 friends': 0, 'old nav 2 enemies': 0, 'old nav 3 friends': 0, 'old nav 3 enemies': 0, 'old nav 4 friends': 0, 'old nav 4 enemies': 0, 'old nav 5 friends': 0, 'old nav 5 enemies': 0, 'old nav 6 friends': 0, 'old nav 6 enemies': 1, 'old nav 7 friends': 1, 'old nav 7 enemies': 0, 'old nav 8 friends': 0, 'old nav 8 enemies': 0, 'old nav 9 friends': 0, 'old nav 9 enemies': 0, 'old nav 10 friends': 0, 'old nav 10 enemies': 0, 'delta x': 0.0, 'delta y': 0.0, 'shoot next': 0, 'crouch next': 0, 'nav target': 0}
#x_df = pd.DataFrame.from_dict([x, x])
#infer(x_df)
#infer(x)
x_str = """
id,tick id,round id,source player id,source player name,demo name,team,cur nav area,cur pos x,cur pos y,cur pos z,cur nav 0 friends,cur nav 0 enemies,cur nav 1 friends,cur nav 1 enemies,cur nav 2 friends,cur nav 2 enemies,cur nav 3 friends,cur nav 3 enemies,cur nav 4 friends,cur nav 4 enemies,cur nav 5 friends,cur nav 5 enemies,cur nav 6 friends,cur nav 6 enemies,cur nav 7 friends,cur nav 7 enemies,cur nav 8 friends,cur nav 8 enemies,cur nav 9 friends,cur nav 9 enemies,cur nav 10 friends,cur nav 10 enemies,last nav area,last pos x,last pos y,last pos z,last nav 0 friends,last nav 0 enemies,last nav 1 friends,last nav 1 enemies,last nav 2 friends,last nav 2 enemies,last nav 3 friends,last nav 3 enemies,last nav 4 friends,last nav 4 enemies,last nav 5 friends,last nav 5 enemies,last nav 6 friends,last nav 6 enemies,last nav 7 friends,last nav 7 enemies,last nav 8 friends,last nav 8 enemies,last nav 9 friends,last nav 9 enemies,last nav 10 friends,last nav 10 enemies,old nav area,old pos x,old pos y,old pos z,old nav 0 friends,old nav 0 enemies,old nav 1 friends,old nav 1 enemies,old nav 2 friends,old nav 2 enemies,old nav 3 friends,old nav 3 enemies,old nav 4 friends,old nav 4 enemies,old nav 5 friends,old nav 5 enemies,old nav 6 friends,old nav 6 enemies,old nav 7 friends,old nav 7 enemies,old nav 8 friends,old nav 8 enemies,old nav 9 friends,old nav 9 enemies,old nav 10 friends,old nav 10 enemies,delta x,delta y,shoot next,crouch next,nav target
0,0,425,0,Will,live,3,7,128,0,-63.9688,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,7,128,0,-63.9688,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,7,128,0,-63.9688,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0
"""
def test_str():
    inference_df = pd.read_csv(StringIO(x_str))
    targets = infer(inference_df)
    print(",".join([str(target) for target in targets]))
#test_str()

if __name__ == "__main__":
    dataPath = Path(sys.argv[1])
    pythonToCppFilePath = dataPath / "python_to_cpp.csv"
    tmpPythonToCppFilePath = dataPath / "python_to_cpp.csv.tmp.write";
    cppToPythonFilePath = dataPath / "cpp_to_python.csv"
    tmpCppToPythonFilePath = dataPath / "cpp_to_python.csv.tmp.read";

    while True:
        if cppToPythonFilePath.exists():
            cppToPythonFilePath.rename(tmpCppToPythonFilePath)
            inference_df = pd.read_csv(tmpCppToPythonFilePath)
            if len(inference_df) > 1:
                print("BADDD")
                print(inference_df)
            targets = infer(inference_df)
            with tmpPythonToCppFilePath.open("w") as f:
                f.write(",".join([str(target) for target in targets]) + "\n")
            tmpPythonToCppFilePath.rename(pythonToCppFilePath)
        sleep(0.01)

