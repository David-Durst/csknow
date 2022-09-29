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

@dataclass(frozen=True)
class ColumnTypes:
    float_standard_cols: List[str]
    float_min_max_cols: List[str]
    float_non_linear_cols: List[str]
    categorical_cols: List[str]
    boolean_cols: List[str]
    bookkeeping_passthrough_cols: List[str]

    @cache
    def columnNameToType(self) -> Dict[str, ColumnTransformerType]:
        result: Dict[str, ColumnTransformerType] = {}
        for col in self.float_standard_cols:
            result[col] = ColumnTransformerType.FLOAT_STANDARD
        for col in self.float_min_max_cols:
            result[col] = ColumnTransformerType.FLOAT_MIN_MAX
        for col in self.float_non_linear_cols:
            result[col] = ColumnTransformerType.FLOAT_NON_LINEAR
        for col in self.categorical_cols:
            result[col] = ColumnTransformerType.CATEGORICAL
        for col in self.boolean_cols:
            result[col] = ColumnTransformerType.BOOLEAN
        for col in self.bookkeeping_passthrough_cols:
            result[col] = ColumnTransformerType.BOOKKEEPING_PASSTHROUGH
        return result

    @cache
    def get_all_columns(self) -> List[str]:
        return self.float_standard_cols + self.float_min_max_cols + self.float_non_linear_cols + \
               self.categorical_cols + self.boolean_cols + self.bookkeeping_passthrough_cols


@dataclass
class PTColumnTransformer:
    pt_ct_type: ColumnTransformerType

    # FLOAT_STANDARD data
    mean: float
    standard_deviation: float

    def convert(self, value):
        if self.pt_ct_type == ColumnTransformerType.FLOAT_STANDARD:
            return (value - self.mean) / self.standard_deviation
        else:
            NotImplementedError

    def inverse(self, value):
        if self.pt_ct_type == ColumnTransformerType.FLOAT_STANDARD:
            return (value * self.standard_deviation) + self.mean
        else:
            NotImplementedError


@dataclass
class IOColumnTransformers:
    input_types: ColumnTypes
    output_types: ColumnTypes

    input_ct: ColumnTransformer
    output_ct: ColumnTransformer

    input_ct_pts: Dict[str, PTColumnTransformer]
    output_ct_pts: Dict[str, PTColumnTransformer]

    def __init__(self, input_types: ColumnTypes, output_types: ColumnTypes, all_data_df: pd.DataFrame):
        self.input_types = input_types
        self.output_types = output_types

        self.input_ct = self.create_column_transformer(self.input_types)
        self.output_ct = self.create_column_transformer(self.output_types)

        # remember: fit Y is ignored for this fitting as not supervised learning
        self.input_ct.fit(all_data_df.loc[:, self.input_types.get_all_columns()])
        self.output_ct.fit(all_data_df.loc[:, self.output_types.get_all_columns()])

        self.create_pytorch_column_transformers(self.input_types, self.input_ct, self.input_ct_pts)
        self.create_pytorch_column_transformers(self.output_types, self.output_ct, self.output_ct_pts)

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

    def create_pytorch_column_transformers(self, types: ColumnTypes, ct: ColumnTransformer, ct_pts: Dict[str, PTColumnTransformer]):
        standard_scaler_ct = ct.named_transformers_['standard-scaler']
        for col, type in types.columnNameToType:
            if type == ColumnTransformerType.FLOAT_STANDARD:
                col_index = standard_scaler_ct.feature_names_in_.index(col)
                ct_pts[col] = PTColumnTransformer(type, standard_scaler_ct.mean_[col_index].item(),
                                                  standard_scaler_ct.scale_[col_index].item())
            else:
                NotImplementedError

    def get_output_name_range(self, name: str) -> range:
        name_indices = [i for i, col_name in enumerate(self.output_ct.get_feature_names_out()) if name in col_name]
        if name_indices:
            return range(min(name_indices), max(name_indices) + 1)
        else:
            return range(0, 0)

    def get_output_name_ranges(self) -> List[range]:
        return [self.get_output_name_range(name) for name in self.output_types.get_all_columns()]


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
