from pathlib import Path
from typing import List, Dict

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from learn_bot.latent.analyze.plot_trajectory_heatmap.build_heatmap_plots import get_title_to_buffers, ImageBuffers
from learn_bot.latent.analyze.plot_trajectory_heatmap.filter_trajectories import TrajectoryFilterOptions


# https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/RUBNER/emd.htm
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html

def compute_one_earth_mover_distance(dist_a: ImageBuffers, dist_b: ImageBuffers) -> float:
    emd = 0
    for ct_team in [True, False]:
        # merge points from different trajectories
        a_points = np.concatenate(dist_a.get_points(ct_team))
        b_points = np.concatenate(dist_b.get_points(ct_team))

        # compute distances between each a point and b point
        d = cdist(a_points, b_points)
        # find the minimum cost flow between points
        assignment = linear_sum_assignment(d)
        # scale by flow (aka smaller number of poinits
        emd += (d[assignment].sum() / min(len(a_points), len(b_points)))
    return emd


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
