from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
import numpy as np
from typing import List
from torch import Tensor

class BaselineBotModel:
    names: List[str]
    models: List[ClassifierMixin]
    output_range_starts: List[int]
    output_range_ends: List[int]
    label_encoders: List[LabelEncoder]

    def __init__(self, X: Tensor, Y: Tensor, output_names: List[str],
                  output_ranges: List[slice]):
        self.models = []
        self.label_encoders = []
        for i, output_name in enumerate(output_names):
            # skip columns that have 1 value
            if output_ranges[i].start + 1 == output_ranges[i].stop:
                self.models.append([])
            else:
                #self.models.append(KNeighborsClassifier(n_neighbors=3)
                #.fit(X.numpy(), Y.numpy())
                self.label_encoders.append(LabelEncoder())
                Y_np = Y.numpy()[:, output_ranges[i]]
                self.label_encoders[i].fit(Y_np)
                self.models.append(LogisticRegression()
                                   .fit(X.numpy(), self.label_encoders[i].transform(Y_np)))
        self.names = output_names
        self.output_ranges = output_ranges

    def score(self, X: Tensor, Y: Tensor):
        for i, name in enumerate(self.names):
            if self.output_ranges[i].start + 1 == self.output_ranges[i].stop:
                print(f'{name} test accuracy 1.')
            else:
                output_subset = Y.numpy()[:, self.output_ranges[i]]
                #print(f'{name} test accuracy {self.models[i].score(X.numpy(), output_subset)}')
                print(f'{name} test accuracy {self.models[i].score(X.numpy(), self.label_encoders[i].transform(output_subset))}')


