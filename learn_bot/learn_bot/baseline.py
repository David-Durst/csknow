from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import ClassifierMixin
import numpy as np
from typing import List
from torch import Tensor

class BaselineBotModel:
    names: List[str]
    models: List[ClassifierMixin]
    Y_range_starts: List[int]
    Y_range_ends: List[int]

    def __init__(self, X: Tensor, Y: Tensor, output_names: List[str],
                  Y_range_starts: List[int], Y_range_ends: List[int]):
        self.models = []
        for i, output_name in enumerate(output_names):
            self.models.append(KNeighborsClassifier(n_neighbors=3)
                          .fit(X.numpy(), Y.numpy()[:, Y_range_starts[i]:Y_range_ends[i]]))
        self.names = output_names
        self.Y_range_starts = Y_range_starts
        self.Y_range_ends = Y_range_ends

    def score(self, X: Tensor, Y: Tensor):
        for i, name in enumerate(self.names):
            Y_subset = Y.numpy()[:, self.Y_range_starts[i]:self.Y_range_ends[i]]
            print(f'{name} test accuracy {self.models[i].score(X.numpy(), Y_subset)}')


