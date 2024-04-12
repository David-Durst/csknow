import sys
from pathlib import Path

import pandas as pd
from scipy.stats import iqr


def compute_inference_time(log_path: Path, model_name: str):
    unscaled_inference_df = pd.read_csv(log_path)
    # scale ms up to first digit
    inference_df = unscaled_inference_df * 1000
    result_path = log_path.parent / 'inference_result.csv'
    if not result_path.exists():
        with open(result_path, 'w') as f:
            f.write('model,median,iqr,1,10,90,99\n')
    with open(result_path, 'a') as f:
        f.write(f"{model_name}, {inference_df.median()[0]:.2f}, {iqr(inference_df)}, {inference_df.quantile(0.01)[0]:.2f}, "
                f"{inference_df.quantile(0.1)[0]:.2f}, {inference_df.quantile(0.9)[0]:.2f}, {inference_df.quantile(0.99)[0]:.2f}\n")


if __name__ == '__main__':
    compute_inference_time(Path(sys.argv[1]), sys.argv[2])