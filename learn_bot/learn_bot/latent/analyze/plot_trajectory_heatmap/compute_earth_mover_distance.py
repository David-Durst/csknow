from pathlib import Path
from typing import List, Dict

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from learn_bot.latent.analyze.plot_trajectory_heatmap.build_heatmap_plots import get_title_to_buffers, ImageBuffers
from learn_bot.latent.analyze.plot_trajectory_heatmap.filter_trajectories import TrajectoryFilterOptions


# https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/RUBNER/emd.htm
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html

def compute_one_earth_mover_distance(dist_a_points: ImageBuffers, dist_b_buffers: ImageBuffers) -> float:
    for ct_team in [True, False]:
        # convert the xy points to lists
        d = cdist(Y1, Y2)
        assignment = linear_sum_assignment(d)
        print(d[assignment].sum() / n)


def compute_trajectory_earth_mover_distances(titles: List[str], diff_indices: List[int], plots_path: Path,
                                             trajectory_filter_options: TrajectoryFilterOptions):
    title_to_buffers = get_title_to_buffers()

    titles_to_emd: Dict[str, float] = {}

    for i, title in enumerate(titles[1:]):
        diff_title = titles[diff_indices[i]]
        titles_to_emd[f'{title} vs {diff_title}'] = \
            compute_one_earth_mover_distance(title_to_buffers[title], title_to_buffers[diff_title])

    with open(plots_path / 'diff' / ('emd_' + str(trajectory_filter_options) + '.txt'), 'w') as f:
        for titles, emd in titles_to_emd.items():
            f.write(f'{titles}: {emd}\n')
