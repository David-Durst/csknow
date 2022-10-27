import pandas as pd
from pathlib import Path
from learn_bot.libs.df_grouping import train_test_split_by_col
from learn_bot.navigation.dataset import NavDataset

csv_outputs_path = Path(__file__).parent / '..' / '..' / '..' / 'analytics' / 'csv_outputs'

non_img_df = pd.read_csv(csv_outputs_path / 'trainNav.csv')

train_test_split = train_test_split_by_col(non_img_df, 'trajectory id')
train_df = train_test_split.train_df
test_df = train_test_split.test_df

train_dataset = NavDataset(train_df, csv_outputs_path / 'trainNavData.tar')
train_dataset.__getitem__(0)
test_dataset = NavDataset(test_df)
