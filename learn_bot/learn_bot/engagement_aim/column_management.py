from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, QuantileTransformer
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
import pandas as pd
from typing import List, Dict
from enum import Enum
from functools import cache

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
    boolean_cols: List[str]
    bookkeeping_passthrough_cols: List[str]

    def __init__(self, float_standard_cols: List[str] = [], float_min_max_cols: List[str] = [],
                 float_non_linear_cols: List[str] = [], categorical_cols: List[str] = [],
                 boolean_cols: List[str] = [], bookkeeping_passthrough_cols: List[str] = []):
        self.float_standard_cols = float_standard_cols
        self.float_min_max_cols = float_min_max_cols
        self.float_non_linear_cols = float_non_linear_cols
        self.categorical_cols = categorical_cols
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


@dataclass
class PTColumnTransformer:
    pt_ct_type: ColumnTransformerType

    # FLOAT_STANDARD data
    mean: float
    standard_deviation: float

    # CATEGORICAL data
    num_cols: int

    def convert(self, value):
        if self.pt_ct_type == ColumnTransformerType.FLOAT_STANDARD:
            return (value - self.mean) / self.standard_deviation
        elif self.pt_ct_type == ColumnTransformerType.CATEGORICAL:
            NotImplementedError
        else:
            NotImplementedError

    def inverse(self, value):
        if self.pt_ct_type == ColumnTransformerType.FLOAT_STANDARD:
            return (value * self.standard_deviation) + self.mean
        elif self.pt_ct_type == ColumnTransformerType.CATEGORICAL:
            NotImplementedError
        else:
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
        if types.categorical_cols:
            transformers.append(
                ('one-hot', OneHotEncoder(), types.categorical_cols))
        if types.boolean_cols or types.bookkeeping_passthrough_cols:
            transformers.append(('pass', 'passthrough', types.bookkeeping_passthrough_cols + types.boolean_cols))
        return ColumnTransformer(transformers=transformers, sparse_threshold=0)

    def create_pytorch_column_transformers(self, types: ColumnTypes, ct: ColumnTransformer) -> List[PTColumnTransformer]:
        result: List[PTColumnTransformer] = []
        standard_scaler_ct = ct.named_transformers_['standard-scaler']
        standard_scaler_ct = ct.named_transformers_['standard-scaler']
        for name, type in zip(types.column_names(), types.column_types()):
            if type == ColumnTransformerType.FLOAT_STANDARD:
                col_index = list(standard_scaler_ct.feature_names_in_).index(name)
                result.append(PTColumnTransformer(type, standard_scaler_ct.mean_[col_index].item(),
                                                  standard_scaler_ct.scale_[col_index].item()))
            else:
                NotImplementedError
        return result

    def get_name_range(self, name: str, input: bool) -> range:
        ct = self.input_ct if input else self.output_ct
        name_indices = [i for i, col_name in enumerate(ct.get_feature_names_out()) if name in col_name]
        if name_indices:
            return range(min(name_indices), max(name_indices) + 1)
        else:
            return range(0, 0)

    def get_name_ranges(self, input: bool) -> List[range]:
        types = self.input_types if input else self.output_types
        return [self.get_name_range(name, input) for name in types.column_names()]


def get_params(types: ColumnTypes, ct: ColumnTransformer) -> str:
    results = []
    if types.boolean_cols or types.bookkeeping_passthrough_cols:
        NotImplementedError
    if types.categorical_cols:
        NotImplementedError
    if types.float_standard_cols:
        for i, col_name in enumerate(types.float_standard_cols):
            transformer = ct.named_transformers_['standard-scaler']
            results.append(f'''standard-scaler;{col_name};{transformer.scale_[i]};{transformer.mean_[i]}''')
    if types.float_min_max_cols:
        NotImplementedError
    if types.float_non_linear_cols:
        NotImplementedError
    return ",".join(results)
