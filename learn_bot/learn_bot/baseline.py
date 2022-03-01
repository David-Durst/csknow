from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import ClassifierMixin
import numpy as np
from typing import List
from torch import Tensor

class BaselineBotModel:
    names: List[str]
    models: List[ClassifierMixin]
    output_range_starts: List[int]
    output_range_ends: List[int]

    def __init__(self, X: Tensor, Y: Tensor, output_names: List[str],
                  output_ranges: List[slice]):
        self.models = []
        for i, output_name in enumerate(output_names):
            # skip columns that have 1 value
            if output_ranges[i].start + 1 == output_ranges[i].stop:
                self.models.append([])
            else:
                self.models.append(KNeighborsClassifier(n_neighbors=3)
                                   .fit(X.numpy(), Y.numpy()[:, output_ranges[i]]))
        self.names = output_names
        self.output_ranges = output_ranges

    def score(self, X: Tensor, Y: Tensor):
        for i, name in enumerate(self.names):
            if self.output_ranges[i].start + 1 == self.output_ranges[i].stop:
                print(f'{name} test accuracy 1.')
            else:
                output_subset = Y.numpy()[:, self.output_ranges[i]]
                print(f'{name} test accuracy {self.models[i].score(X.numpy(), output_subset)}')


