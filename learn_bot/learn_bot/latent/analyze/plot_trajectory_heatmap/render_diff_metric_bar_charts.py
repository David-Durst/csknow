import glob
import os
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

fig_length = 10


def render_diff_metric_bar_charts(plots_path: Path):
    metric_files = glob.glob(str(plots_path / 'diff' / '*.txt'))
    metric_files.sort(key=lambda x: os.path.getmtime(x))

    fig = None
    axs = None

    ax_index = 0
    for metric_file in metric_files:
        data_subset = Path(metric_file).stem
        metric_file_df = pd.read_csv(metric_file)
        metrics = metric_file_df.columns[1:]
        metric_file_df['Data'] = metric_file_df['Title']#.str.slice(stop=12)
        if ax_index == 0:
            fig = plt.figure(figsize=(fig_length * len(metrics), fig_length * len(metric_files)), constrained_layout=True)
            axs = fig.subplots(len(metric_files), len(metrics), squeeze=False)
            #plt.xticks(rotation=45)

        for col_index, col in enumerate(metrics):
            metric_file_df.plot(x='Data', y=col, kind='bar', rot=45, legend=False, ax=axs[ax_index, col_index])
            #plt.xticks(rotation=45)
            axs[ax_index, col_index].set_title(data_subset + col)

        ax_index += 1

    plt.savefig(plots_path / 'diff' / 'emd_bar_plots.png')
    plt.close(fig)


if __name__ == '__main__':
    render_diff_metric_bar_charts(Path("/home/durst/dev/csknow/learn_bot/learn_bot/latent/analyze/similarity_plots/1_6_24_learned_similarity"))