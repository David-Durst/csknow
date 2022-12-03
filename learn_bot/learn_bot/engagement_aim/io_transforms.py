from dataclasses import dataclass
import pandas as pd
from typing import List, Dict, Set, Tuple
from enum import Enum
from abc import abstractmethod, ABC
from torch.nn import functional as F
import torch

CPU_DEVICE_STR = "cpu"
CUDA_DEVICE_STR = "cuda"

PRIOR_TICKS = -12
PRIOR_TICKS_POS = -1 * PRIOR_TICKS
FUTURE_TICKS = 6
CUR_TICK = 1


class ColumnTransformerType(Enum):
    FLOAT_STANDARD = 0
    FLOAT_DELTA = 1
    CATEGORICAL = 2


@dataclass
class DeltaColumn:
    relative_col: str
    reference_col: str


def split_delta_columns(delta_ref_columns: List[DeltaColumn]) -> Tuple[List[str], List[str]]:
    return [drc.relative_col for drc in delta_ref_columns], [drc.reference_col for drc in delta_ref_columns]


ALL_TYPES: Set[ColumnTransformerType] = {ColumnTransformerType.FLOAT_STANDARD,
                                         ColumnTransformerType.FLOAT_DELTA,
                                         ColumnTransformerType.CATEGORICAL}


class ColumnTypes:
    float_standard_cols: List[str]
    float_delta_cols: List[DeltaColumn]
    categorical_cols: List[str]
    num_cats_per_col: Dict[str, int]

    def __init__(self, float_standard_cols: List[str] = [], float_delta_cols: List[DeltaColumn] = [],
                 categorical_cols: List[str] = [], num_cats_per_col: List[int] = []):
        self.float_standard_cols = float_standard_cols
        self.float_delta_cols = float_delta_cols
        self.categorical_cols = categorical_cols
        self.num_cats_per_col = {}
        for cat, num_cats in zip(categorical_cols, num_cats_per_col):
            self.num_cats_per_col[cat] = num_cats

    # caching values
    column_types_ = None
    all_cols_ = None

    def column_types(self) -> List[ColumnTransformerType]:
        if self.column_types_ is None:
            self.column_types_ = []
            for _ in self.float_standard_cols:
                self.column_types_.append(ColumnTransformerType.FLOAT_STANDARD)
            for _ in self.float_delta_cols:
                self.column_types_.append(ColumnTransformerType.FLOAT_DELTA)
            for _ in self.categorical_cols:
                self.column_types_.append(ColumnTransformerType.CATEGORICAL)
        return self.column_types_

    def column_names(self) -> List[str]:
        if self.all_cols_ is None:
            relative_cols, _ = split_delta_columns(self.float_delta_cols)
            self.all_cols_ = self.float_standard_cols + relative_cols + self.categorical_cols
        return self.all_cols_


class PTColumnTransformer(ABC):
    pt_ct_type: ColumnTransformerType

    @abstractmethod
    def convert(self, value):
        pass

    @abstractmethod
    def inverse(self, value):
        pass

    @abstractmethod
    def delta_convert(self, relative_value: torch.Tensor, reference_value: torch.Tensor):
        pass

    @abstractmethod
    def delta_inverse(self, relative_value: torch.Tensor, reference_value: torch.Tensor):
        pass


class PTMeanStdColumnTransformer(PTColumnTransformer):
    # FLOAT_STANDARD data
    cpu_means: torch.Tensor
    cpu_standard_deviations: torch.Tensor
    means: torch.Tensor
    standard_deviations: torch.Tensor

    pt_ct_type: ColumnTransformerType = ColumnTransformerType.FLOAT_STANDARD

    def __init__(self, means: torch.Tensor, standard_deviations: torch.Tensor):
        self.cpu_means = means.view(1,-1)
        self.cpu_standard_deviations = standard_deviations.view(1,-1)
        # done so columsn that all equal mean are 0, 253 is sentinel value,
        # doesn't matter what value, sub by mean will make it equal 0.
        # NOTE TO SELF: IT DOES MATTER, TRAINING VALUES WON'T BE 0, AND LARGE STD DEV PREVENTS THEM FROM CONVERGING
        # DUE TO BAD LOSS
        self.cpu_standard_deviations[self.cpu_standard_deviations == 0.] = \
            torch.finfo(standard_deviations.dtype).smallest_normal
        self.means = self.cpu_means.to(CUDA_DEVICE_STR)
        self.standard_deviations = self.cpu_standard_deviations.to(CUDA_DEVICE_STR)

    def convert(self, value: torch.Tensor):
        if value.device.type == CPU_DEVICE_STR:
            return (value - self.cpu_means) / self.cpu_standard_deviations
        else:
            return (value - self.means) / self.standard_deviations

    def inverse(self, value: torch.Tensor):
        if value.device.type == CPU_DEVICE_STR:
            return (value * self.cpu_standard_deviations) + self.cpu_means
        else:
            return (value * self.standard_deviations) + self.means

    def delta_convert(self, relative_value: torch.Tensor, reference_value: torch.Tensor):
        raise NotImplementedError

    def delta_inverse(self, relative_value: torch.Tensor, reference_value: torch.Tensor):
        raise NotImplementedError


class PTDeltaMeanStdColumnTransformer(PTColumnTransformer):
    # FLOAT_STANDARD data
    cpu_delta_means: torch.Tensor
    cpu_delta_standard_deviations: torch.Tensor
    delta_means: torch.Tensor
    delta_standard_deviations: torch.Tensor

    pt_ct_type: ColumnTransformerType = ColumnTransformerType.FLOAT_STANDARD

    def __init__(self, delta_means: torch.Tensor, delta_standard_deviations: torch.Tensor):
        self.delta_cpu_means = delta_means.view(1,-1)
        self.delta_cpu_standard_deviations = delta_standard_deviations.view(1,-1)
        # done so columsn that all equal mean are 0, 253 is sentinel value,
        # doesn't matter what value, sub by mean will make it equal 0.
        # NOTE TO SELF: IT DOES MATTER, TRAINING VALUES WON'T BE 0, AND LARGE STD DEV PREVENTS THEM FROM CONVERGING
        # DUE TO BAD LOSS
        self.delta_cpu_standard_deviations[self.delta_cpu_standard_deviations == 0.] = \
            torch.finfo(delta_standard_deviations.dtype).smallest_normal
        self.delta_means = self.delta_cpu_means.to(CUDA_DEVICE_STR)
        self.delta_standard_deviations = self.delta_cpu_standard_deviations.to(CUDA_DEVICE_STR)

    def convert(self, offset_value: torch.Tensor):
        raise NotImplementedError

    def inverse(self, offset_value: torch.Tensor):
        raise NotImplementedError

    def delta_convert(self, relative_value: torch.Tensor, reference_value: torch.Tensor):
        if relative_value.device.type == CPU_DEVICE_STR:
            return (relative_value - reference_value - self.cpu_delta_means) / self.cpu_delta_standard_deviations
        else:
            return (relative_value - reference_value - self.delta_means) / self.delta_standard_deviations

    def delta_inverse(self, relative_value: torch.Tensor, reference_value: torch.Tensor):
        if relative_value.device.type == CPU_DEVICE_STR:
            return (relative_value * self.cpu_delta_standard_deviations) + self.cpu_delta_means + reference_value
        else:
            return (relative_value * self.delta_standard_deviations) + self.delta_means + reference_value


@dataclass
class PTOneHotColumnTransformer(PTColumnTransformer):
    # CATEGORICAL data
    num_classes: int

    pt_ct_type: ColumnTransformerType = ColumnTransformerType.CATEGORICAL

    def convert(self, value: torch.Tensor):
        one_hot_result = F.one_hot(value.to(torch.int64), num_classes=self.num_classes)
        one_hot_float_result = one_hot_result.to(value.dtype)
        return torch.flatten(one_hot_float_result, start_dim=1)

    def inverse(self, value: torch.Tensor):
        return torch.argmax(value, -1, keepdim=True)

    def delta_convert(self, relative_value: torch.Tensor, reference_value: torch.Tensor):
        raise NotImplementedError

    def delta_inverse(self, relative_value: torch.Tensor, reference_value: torch.Tensor):
        raise NotImplementedError


class IOColumnTransformers:
    input_types: ColumnTypes
    output_types: ColumnTypes

    input_ct_pts: List[PTColumnTransformer]
    output_ct_pts: List[PTColumnTransformer]

    def __init__(self, input_types: ColumnTypes, output_types: ColumnTypes, all_data_df: pd.DataFrame):
        self.input_types = input_types
        self.output_types = output_types

        self.input_ct_pts = self.create_pytorch_column_transformers(self.input_types, all_data_df)
        self.output_ct_pts = self.create_pytorch_column_transformers(self.output_types, all_data_df)

    def create_pytorch_column_transformers(self, types: ColumnTypes, all_data_df: pd.DataFrame) -> \
            List[PTColumnTransformer]:
        result: List[PTColumnTransformer] = []
        if types.float_standard_cols:
            result.append(PTMeanStdColumnTransformer(
                torch.Tensor(all_data_df.loc[:, types.float_standard_cols].mean()),
                torch.Tensor(all_data_df.loc[:, types.float_standard_cols].std()),
            ))
        if types.float_delta_cols:
            relative_cols, reference_cols = split_delta_columns(types.float_delta_cols)
            relative_df = all_data_df.loc[:, relative_cols]
            reference_df = all_data_df.loc[:, relative_cols]
            delta_df = relative_df - reference_df
            result.append(PTDeltaMeanStdColumnTransformer(
                torch.Tensor(delta_df.mean()),
                torch.Tensor(delta_df.std()),
            ))
        for name in types.categorical_cols:
            result.append(PTOneHotColumnTransformer(types.num_cats_per_col[name]))
        return result

    def get_name_ranges(self, input: bool, transformed: bool, types: Set[ColumnTransformerType] = ALL_TYPES) -> List[range]:
        result: List[range] = []
        cur_start: int = 0

        column_types: ColumnTypes = self.input_types if input else self.output_types
        cts: List[PTColumnTransformer] = self.input_ct_pts if input else self.output_ct_pts

        for _ in column_types.float_standard_cols:
            if ColumnTransformerType.FLOAT_STANDARD in types:
                result.append(range(cur_start, cur_start + 1))
            cur_start += 1

        for _ in column_types.float_delta_cols:
            if ColumnTransformerType.FLOAT_DELTA in types:
                result.append(range(cur_start, cur_start + 1))
            cur_start += 1

        for ct in cts:
            if ct.pt_ct_type == ColumnTransformerType.CATEGORICAL:
                if transformed:
                    if ColumnTransformerType.CATEGORICAL in types:
                        result.append(range(cur_start, cur_start + ct.num_classes))
                    cur_start += ct.num_classes
                else:
                    if ColumnTransformerType.CATEGORICAL in types:
                        result.append(range(cur_start, cur_start + 1))
                    cur_start += 1

        return result

    # given the locations of columns in an input tensor that are used as reference for producing delta columns
    def get_input_delta_reference_positions(self, types: ColumnTypes) -> List[int]:
        result: List[int] = []

        _, reference_cols = split_delta_columns(types.float_delta_cols)

        cur_col_index = 0
        # input has baseline to compare to
        for reference_col in reference_cols:
            if reference_col in self.input_types.float_standard_cols:
                result.append(self.input_types.float_standard_cols.index(reference_col))
            cur_col_index += 1

        return result

    def transform_columns(self, input: bool, x: torch.Tensor, x_input: torch.Tensor) -> torch.Tensor:
        uncat_result: List[torch.Tensor] = []

        ct_pts = self.input_ct_pts if input else self.output_ct_pts
        types = self.input_types if input else self.output_types

        ct_offset = 0
        x_float_standard_name_ranges = self.get_name_ranges(input, False, {ColumnTransformerType.FLOAT_STANDARD})
        if x_float_standard_name_ranges:
            x_floats = x[:, x_float_standard_name_ranges[0].start:x_float_standard_name_ranges[-1].stop]
            uncat_result.append(ct_pts[0].convert(x_floats))
            ct_offset += 1

        x_float_delta_name_ranges = self.get_name_ranges(input, False, {ColumnTransformerType.FLOAT_DELTA})
        if x_float_delta_name_ranges:
            x_relative_floats = x[:, x_float_delta_name_ranges[0].start:x_float_delta_name_ranges[-1].stop]
            x_float_delta_reference_positions = self.get_input_delta_reference_positions(types)
            x_reference_floats = x_input[:, x_float_delta_reference_positions]
            uncat_result.append(ct_pts[ct_offset].delta_convert(x_relative_floats, x_reference_floats))
            ct_offset += 1

        x_categorical_name_ranges = self.get_name_ranges(input, False, {ColumnTransformerType.CATEGORICAL})
        for i, categorical_name_range in enumerate(x_categorical_name_ranges):
            uncat_result.append(ct_pts[i+ct_offset].convert(x[:, categorical_name_range]))

        return torch.cat(uncat_result, dim=1)

    def untransform_columns(self, input: bool, x: torch.Tensor, x_input: torch.Tensor) -> torch.Tensor:
        uncat_result: List[torch.Tensor] = []

        ct_pts = self.input_ct_pts if input else self.output_ct_pts
        types = self.input_types if input else self.output_types

        ct_offset = 0
        x_float_name_ranges = self.get_name_ranges(input, True, {ColumnTransformerType.FLOAT_STANDARD})
        if x_float_name_ranges:
            x_floats = x[:, x_float_name_ranges[0].start:x_float_name_ranges[-1].stop]
            uncat_result.append(ct_pts[0].inverse(x_floats))
            ct_offset += 1

        x_float_delta_name_ranges = self.get_name_ranges(input, False, {ColumnTransformerType.FLOAT_DELTA})
        if x_float_delta_name_ranges:
            x_relative_floats = x[:, x_float_delta_name_ranges[0].start:x_float_delta_name_ranges[-1].stop]
            x_float_delta_reference_positions = self.get_input_delta_reference_positions(types)
            x_reference_floats = x_input[:, x_float_delta_reference_positions]
            uncat_result.append(ct_pts[ct_offset].delta_inverse(x_relative_floats, x_reference_floats))
            ct_offset += 1

        x_categorical_name_ranges = self.get_name_ranges(input, True, {ColumnTransformerType.CATEGORICAL})
        for i, categorical_name_range in enumerate(x_categorical_name_ranges):
            uncat_result.append(ct_pts[i+ct_offset].inverse(x[:, categorical_name_range]))

        return torch.cat(uncat_result, dim=1)

    def get_untransformed_value(self, x: torch.Tensor, col_name: str, input: bool) -> float:
        col_names = self.input_types.column_names() if input else self.output_types.column_names()
        col_ranges = self.get_name_ranges(input, False)
        col_index = 0
        for i, col_name_ in enumerate(col_names):
            if col_name_ == col_name:
                col_index = i
                break

        if input:
            return x[col_ranges[col_index].start].item()
        else:
            return x[1:, col_ranges[col_index].start].item()

    def set_untransformed_input_value(self, x: torch.Tensor, col_name: str, value: float):
        col_names = self.input_types.column_names() if input else self.output_types.column_names()
        col_ranges = self.get_name_ranges(True, False)
        col_index = 0
        for i, col_name_ in enumerate(col_names):
            if col_name_ == col_name:
                col_index = i
                break

        x[col_ranges[col_index].start] = value


def get_transformed_outputs(x: torch.Tensor) -> torch.Tensor:
    return x[:, 0, :].squeeze()


def get_untransformed_outputs(x: torch.Tensor):
    return x[:, 1, :].squeeze()
