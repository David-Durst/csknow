from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from joblib import load
from pathlib import Path


input_ct = load(Path(__file__).parent / '..' / 'model' / 'input_ct.joblib')
output_ct = load(Path(__file__).parent / '..' / 'model' / 'output_ct.joblib')
torch.save(model.state_dict(), Path(__file__).parent / '..' / 'model' / 'model.pt')
