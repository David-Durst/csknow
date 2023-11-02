import pandas as pd

from learn_bot.latent.place_area.load_data import LoadDataResult


def get_nearest_neighbors(load_data_result: LoadDataResult, num_matches: int = 10) -> pd.DataFrame:
