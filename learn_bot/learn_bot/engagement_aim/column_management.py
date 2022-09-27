from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, QuantileTransformer
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
import pandas as pd
from typing import List


@dataclass
class ColumnTypes:
    float_standard_cols: List[str]
    float_min_max_cols: List[str]
    float_non_linear_cols: List[str]
    categorical_cols: List[str]
    boolean_cols: List[str]
    bookkeeping_passthrough_cols: List[str]

    def get_all_columns(self) -> List[str]:
        return self.float_standard_cols + self.float_min_max_cols + self.float_non_linear_cols + \
               self.categorical_cols + self.boolean_cols + self.bookkeeping_passthrough_cols


@dataclass
class IOColumnTransformers:
    input_types: ColumnTypes
    output_types: ColumnTypes

    input_ct: ColumnTransformer
    output_ct: ColumnTransformer

    def __init__(self, input_types: ColumnTypes, output_types: ColumnTypes, all_data_df: pd.DataFrame):
        self.input_types = input_types
        self.output_types = output_types

        self.input_ct = self.create_column_transformer(self.input_types)
        self.output_ct = self.create_column_transformer(self.output_types)

        # remember: fit Y is ignored for this fitting as not supervised learning
        self.input_ct.fit(all_data_df.loc[:, self.input_types.get_all_columns()])
        self.output_ct.fit(all_data_df.loc[:, self.output_types.get_all_columns()])

    # https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
    def create_column_transformer(self, types: ColumnTypes):
        transformers = []
        if types.boolean_cols or types.bookkeeping_passthrough_cols:
            transformers.append(('pass', 'passthrough', types.bookkeeping_passthrough_cols + types.boolean_cols))
        if types.categorical_cols:
            transformers.append(
                ('one-hot', OneHotEncoder(), types.categorical_cols))
        if types.float_standard_cols:
            transformers.append(('standard-scaler', StandardScaler(), types.float_standard_cols))
        if types.float_min_max_cols:
            transformers.append(('zero-to-one-min-max', MinMaxScaler(), types.float_min_max_cols))
        if types.float_non_linear_cols:
            transformers.append(('zero-to-one-non-linear', QuantileTransformer(), types.float_non_linear_cols))
        return ColumnTransformer(transformers=transformers, sparse_threshold=0)

    def get_output_name_range(self, name: str) -> range:
        name_indices = [i for i, col_name in enumerate(self.output_ct.get_feature_names_out()) if name in col_name]
        if name_indices:
            return range(min(name_indices), max(name_indices) + 1)
        else:
            return range(0, 0)

    def get_output_name_ranges(self) -> List[range]:
        return [self.get_output_name_range(name) for name in self.output_types.get_all_columns()]
