from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, QuantileTransformer
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
import pandas as pd
from typing import List, Dict
from enum import Enum
from functools import cache
from abc import abstractmethod
from torch.nn import functional as F
import torch

class ColumnTransformerType(Enum):
    FLOAT_STANDARD = 0
    FLOAT_MIN_MAX = 1
    FLOAT_NON_LINEAR = 2
    CATEGORICAL = 3
    BOOLEAN = 4
    BOOKKEEPING_PASSTHROUGH = 5

class ColumnTypes:
    float_standard_cols: List[str]
    float_min_max_cols: List[str]
    float_non_linear_cols: List[str]
    categorical_cols: List[str]
    num_cats_per_col: Dict[str, int]
    boolean_cols: List[str]
    bookkeeping_passthrough_cols: List[str]

    def __init__(self, float_standard_cols: List[str] = [], float_min_max_cols: List[str] = [],
                 float_non_linear_cols: List[str] = [], categorical_cols: List[str] = [],
                 num_cats_per_col: List[int] = [],
                 boolean_cols: List[str] = [], bookkeeping_passthrough_cols: List[str] = []):
        self.float_standard_cols = float_standard_cols
        self.float_min_max_cols = float_min_max_cols
        self.float_non_linear_cols = float_non_linear_cols
        self.categorical_cols = categorical_cols
        self.num_cats_per_col = {}
        for cat, num_cats in zip(categorical_cols, num_cats_per_col):
            self.num_cats_per_col[cat] = num_cats
        self.boolean_cols = boolean_cols
        self.bookkeeping_passthrough_cols = bookkeeping_passthrough_cols

    #caching values
    column_types_ = None
    all_cols_ = None

    def column_types(self) -> List[ColumnTransformerType]:
        if self.column_types_ is None:
            self.column_types_ = []
            for _ in self.float_standard_cols:
                self.column_types_.append(ColumnTransformerType.FLOAT_STANDARD)
            for _ in self.float_min_max_cols:
                self.column_types_.append(ColumnTransformerType.FLOAT_MIN_MAX)
            for _ in self.float_non_linear_cols:
                self.column_types_.append(ColumnTransformerType.FLOAT_NON_LINEAR)
            for _ in self.categorical_cols:
                self.column_types_.append(ColumnTransformerType.CATEGORICAL)
            for _ in self.boolean_cols:
                self.column_types_.append(ColumnTransformerType.BOOLEAN)
            for _ in self.bookkeeping_passthrough_cols:
                self.column_types_.append(ColumnTransformerType.BOOKKEEPING_PASSTHROUGH)
        return self.column_types_

    def column_names(self) -> List[str]:
        if self.all_cols_ is None:
            self.all_cols_ = self.float_standard_cols + self.float_min_max_cols + self.float_non_linear_cols + \
                             self.categorical_cols + self.boolean_cols + self.bookkeeping_passthrough_cols
        return self.all_cols_


class PTColumnTransformer:
    pt_ct_type: ColumnTransformerType

    @abstractmethod
    def convert(self, value):
        pass

    @abstractmethod
    def inverse(self, value):
        pass


@dataclass
class PTMeanStdColumnTransformer(PTColumnTransformer):
    # FLOAT_STANDARD data
    mean: float
    standard_deviation: float

    pt_ct_type: ColumnTransformerType = ColumnTransformerType.FLOAT_STANDARD

    def convert(self, value):
        return (value - self.mean) / self.standard_deviation

    def inverse(self, value):
        return (value * self.standard_deviation) + self.mean


@dataclass
class PTOneHotColumnTransformer(PTColumnTransformer):
    # CATEGORICAL data
    num_classes: int

    pt_ct_type: ColumnTransformerType = ColumnTransformerType.CATEGORICAL

    def convert(self, value: torch.Tensor):
        one_hot_result =  F.one_hot(value.to(torch.int64), num_classes=self.num_classes)
        one_hot_float_result = one_hot_result.to(value.dtype)
        return torch.flatten(one_hot_float_result, start_dim=1)

    def inverse(self, value):
        NotImplementedError


class IOColumnTransformers:
    input_types: ColumnTypes
    output_types: ColumnTypes

    input_ct: ColumnTransformer
    output_ct: ColumnTransformer

    input_ct_pts: List[PTColumnTransformer]
    output_ct_pts: List[PTColumnTransformer]

    def __init__(self, input_types: ColumnTypes, output_types: ColumnTypes, all_data_df: pd.DataFrame):
        self.input_types = input_types
        self.output_types = output_types

        self.input_ct = self.create_column_transformer(self.input_types)
        self.output_ct = self.create_column_transformer(self.output_types)

        # remember: fit Y is ignored for this fitting as not supervised learning
        x = all_data_df.loc[:, self.input_types.column_names()[0:1]]
        self.input_ct.fit(all_data_df.loc[:, self.input_types.column_names()])
        self.output_ct.fit(all_data_df.loc[:, self.output_types.column_names()])

        self.input_ct_pts = self.create_pytorch_column_transformers(self.input_types, self.input_ct)
        self.output_ct_pts = self.create_pytorch_column_transformers(self.output_types, self.output_ct)

    # https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
    def create_column_transformer(self, types: ColumnTypes):
        transformers = []
        if types.float_standard_cols:
            transformers.append(('standard-scaler', StandardScaler(), types.float_standard_cols))
        if types.float_min_max_cols:
            transformers.append(('zero-to-one-min-max', MinMaxScaler(), types.float_min_max_cols))
        if types.float_non_linear_cols:
            transformers.append(('zero-to-one-non-linear', QuantileTransformer(), types.float_non_linear_cols))
        if types.categorical_cols or types.boolean_cols or types.bookkeeping_passthrough_cols:
            transformers.append(('pass', 'passthrough', types.bookkeeping_passthrough_cols + types.boolean_cols +
                                 types.categorical_cols))
        return ColumnTransformer(transformers=transformers, sparse_threshold=0)

    def create_pytorch_column_transformers(self, types: ColumnTypes, ct: ColumnTransformer) -> List[PTColumnTransformer]:
        result: List[PTColumnTransformer] = []
        standard_scaler_ct = ct.named_transformers_['standard-scaler']
        for name, type in zip(types.column_names(), types.column_types()):
            if type == ColumnTransformerType.FLOAT_STANDARD:
                col_index = list(standard_scaler_ct.feature_names_in_).index(name)
                result.append(PTMeanStdColumnTransformer(standard_scaler_ct.mean_[col_index].item(),
                                                         standard_scaler_ct.scale_[col_index].item()))
            if type == ColumnTransformerType.CATEGORICAL:
                result.append(PTOneHotColumnTransformer(types.num_cats_per_col[name]))
            else:
                NotImplementedError
        return result

    def get_name_ranges(self, input: bool) -> List[range]:
        result: List[range] = []
        cur_start: int = 0
        cts: List[PTColumnTransformer] = self.input_ct_pts if input else self.output_ct_pts
        for ct in cts:
            if ct.pt_ct_type == ColumnTransformerType.FLOAT_STANDARD:
                result.append(range(cur_start, cur_start + 1))
            elif ct.pt_ct_type == ColumnTransformerType.CATEGORICAL:
                result.append(range(cur_start, cur_start + ct.num_classes))
            else:
                NotImplementedError
            cur_start = result[-1].stop
        return result
