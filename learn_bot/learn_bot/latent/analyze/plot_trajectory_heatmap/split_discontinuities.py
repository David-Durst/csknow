from typing import List, Optional

import numpy as np


# given a list of constraints, identify where valid regions should be split
# each index is relative to number of valid points before it, not total points
# as will filter out invalid points before plotting
def identify_discontinuity_indices_in_valid_points(constraint: np.ndarray) -> List[int]:
    # identify number of valids up to each point in constraint array
    constraint_int = constraint.astype(np.int)
    valids_up_to_point = np.cumsum(constraint_int) - 1

    # identify discontinuity in valids - region where true and was false before and had a prior valid (so not start)
    prior_constraint = np.roll(constraint, 1)
    discontinuities = constraint & (constraint != prior_constraint) & (valids_up_to_point > 0)

    # convert overall indices to just indices in valids
    overall_discontinuity_indices = np.nonzero(discontinuities)
    return valids_up_to_point[overall_discontinuity_indices]


# split the list of valid points by the discontinuities
def split_list_by_valid_points(l: List, valid_discontinuities: Optional[List[int]]) -> List[List]:
    if valid_discontinuities is None:
        return [l]
    result = []

    prior_discontinuity = 0
    for discontinuity in valid_discontinuities:
        # if this particular list doesn't contain all possible valids (aka one player died early)
        if discontinuity > len(l):
            result.append(l[prior_discontinuity:len(l)])
        else:
            result.append(l[prior_discontinuity:discontinuity])
        prior_discontinuity = discontinuity
    # handle last region if this player lived entire rounds
    result.append(l[prior_discontinuity:])

    assert sum(len(l_inner) for l_inner in result) == len(l)
    return result