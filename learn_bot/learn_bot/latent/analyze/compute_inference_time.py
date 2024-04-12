import sys
from pathlib import Path

import pandas as pd
from scipy.stats import iqr


def compute_inference_time(log_path: Path, model_name: str):
    inference_df = pd.read_csv(log_path)
    result_path = log_path.parent / 'inference_result.csv'
    if not result_path.exists():
        with open(result_path, 'w') as f:
            f.write('model,median,iqr,1,10,90,99\n')
    with open(result_path, 'a') as f:
        f.write(f"{model_name}, {inference_df.median()}, {iqr(inference_df)}, {inference_df.quantile(0.01)}, "
                f"{inference_df.quantile(0.1)}, {inference_df.quantile(0.9)}, {inference_df.quantile(0.99)}\n")


if __name__ == '__main__':
    compute_inference_time(Path(sys.argv[1]), sys.argv[2])