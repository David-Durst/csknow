import re
from dataclasses import dataclass
from functools import cache

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, FrozenSet, Union
from enum import Enum
from abc import abstractmethod, ABC
from torch.nn import functional as F
import torch

CPU_DEVICE_STR = "cpu"
CUDA_DEVICE_STR = "cuda"

PRIOR_TICKS = -13
PRIOR_TICKS_POS = -1 * PRIOR_TICKS
FUTURE_TICKS = 13
CUR_TICK = 1

ModelOutput = Tuple[torch.Tensor, torch.Tensor]


class ColumnTransformerType(Enum):
    FLOAT_STANDARD = 0
    FLOAT_DELTA = 1
    # -180 to 180
    FLOAT_180_ANGLE = 2
    FLOAT_180_ANGLE_DELTA = 3
    # -90 to 90
    FLOAT_90_ANGLE = 4
    FLOAT_90_ANGLE_DELTA = 5
    CATEGORICAL = 6


@dataclass
class DeltaColumn:
    relative_col: str
    reference_col: str
    target_col: str


def split_delta_columns(delta_ref_columns: List[DeltaColumn]) -> Tuple[List[str], List[str]]:
    return [drc.relative_col for drc in delta_ref_columns], [drc.reference_col for drc in delta_ref_columns]


def target_delta_columns(delta_ref_columns: List[DeltaColumn]) -> Tuple[List[str], List[str]]:
    return [drc.target_col for drc in delta_ref_columns]


ALL_TYPES: FrozenSet[ColumnTransformerType] = frozenset({ColumnTransformerType.FLOAT_STANDARD,
                                                         ColumnTransformerType.FLOAT_DELTA,
                                                         ColumnTransformerType.FLOAT_180_ANGLE,
                                                         ColumnTransformerType.FLOAT_180_ANGLE_DELTA,
                                                         ColumnTransformerType.FLOAT_90_ANGLE,
                                                         ColumnTransformerType.FLOAT_90_ANGLE_DELTA,
                                                         ColumnTransformerType.CATEGORICAL})


@dataclass(frozen=True)
class ColumnTimeOffset:
    temporal: bool
    offset: int

    def offset_valid_for_range(self, time_offset_range: range, include_non_temporal: bool):
        if include_non_temporal:
            return not self.temporal or self.offset in time_offset_range
        else:
            return self.temporal and self.offset in time_offset_range


class ColumnTypes:
    float_standard_cols: List[str]
    float_delta_cols: List[DeltaColumn]
    float_180_angle_cols: List[str]
    float_180_angle_delta_cols: List[DeltaColumn]
    float_90_angle_cols: List[str]
    float_90_angle_delta_cols: List[DeltaColumn]
    categorical_cols: List[str]
    num_cats_per_col: Dict[str, int]
    # need to use same mean and std dev for angular columns so they can be compared in loss
    float_angular_standard_cols: List[str]
    float_angular_delta_cols: List[DeltaColumn]
    float_180_wrap_cols: List[str]

    def __init__(self, float_standard_cols: List[str] = [], float_delta_cols: List[DeltaColumn] = [],
                 float_180_angle_cols: List[str] = [], float_180_angle_delta_cols: List[DeltaColumn] = [],
                 float_90_angle_cols: List[str] = [], float_90_angle_delta_cols: List[DeltaColumn] = [],
                 categorical_cols: List[str] = [], num_cats_per_col: List[int] = [],
                 # these are for computing a single mean/std dev across all float_standard/float_delta
                 # cols that are angular
                 float_angular_standard_cols: List[str] = [], float_angular_delta_cols: List[DeltaColumn] = [],
                 # these are standard or delta columns that wrap around at 180/-180 when computing loss
                 float_180_wrap_cols: List[str] = []):
        self.float_standard_cols = float_standard_cols
        self.float_delta_cols = float_delta_cols
        self.float_180_angle_cols = float_180_angle_cols
        self.float_180_angle_delta_cols = float_180_angle_delta_cols
        self.float_90_angle_cols = float_90_angle_cols
        self.float_90_angle_delta_cols = float_90_angle_delta_cols
        self.categorical_cols = categorical_cols
        self.num_cats_per_col = {}
        for cat, num_cats in zip(categorical_cols, num_cats_per_col):
            self.num_cats_per_col[cat] = num_cats
        self.float_angular_standard_cols = float_angular_standard_cols
        self.float_angular_delta_cols = float_angular_delta_cols
        self.float_180_wrap_cols = float_180_wrap_cols
        self.compute_time_offsets()

    col_time_offsets: Dict[str, ColumnTimeOffset]

    def compute_time_offsets(self):
        self.col_time_offsets = {}
        for col_name in self.column_names():
            if '(t)' in col_name:
                self.col_time_offsets[col_name] = ColumnTimeOffset(True, 0)
            elif '(t+' in col_name:
                self.col_time_offsets[col_name] = ColumnTimeOffset(True,
                                                                   int(re.search(r'\(t\+(\d+)\)', col_name).group(1)))
            elif '(t-' in col_name:
                self.col_time_offsets[col_name] = ColumnTimeOffset(True, -1 *
                                                                   int(re.search(r'\(t\-(\d+)\)', col_name).group(1)))
            else:
                self.col_time_offsets[col_name] = ColumnTimeOffset(False, 0)

    # caching values
    column_types_ = None
    all_cols_ = None
    delta_float_column_names_ = None
    delta_180_angle_column_names_ = None
    delta_90_angle_column_names_ = None
    delta_float_target_column_names_ = None

    def column_types(self) -> List[ColumnTransformerType]:
        if self.column_types_ is None:
            self.column_types_ = []
            for _ in self.float_standard_cols:
                self.column_types_.append(ColumnTransformerType.FLOAT_STANDARD)
            for _ in self.float_delta_cols:
                self.column_types_.append(ColumnTransformerType.FLOAT_DELTA)
            for _ in self.float_180_angle_cols:
                self.column_types_.append(ColumnTransformerType.FLOAT_180_ANGLE)
            for _ in self.float_180_angle_delta_cols:
                self.column_types_.append(ColumnTransformerType.FLOAT_180_ANGLE_DELTA)
            for _ in self.float_90_angle_cols:
                self.column_types_.append(ColumnTransformerType.FLOAT_90_ANGLE)
            for _ in self.float_90_angle_delta_cols:
                self.column_types_.append(ColumnTransformerType.FLOAT_90_ANGLE_DELTA)
            for _ in self.categorical_cols:
                self.column_types_.append(ColumnTransformerType.CATEGORICAL)
        return self.column_types_

    def column_names(self, relative_cols_first=False) -> List[str]:
        if relative_cols_first:
            relative_cols, _ = split_delta_columns(self.float_delta_cols)
            relative_180_cols, _ = split_delta_columns(self.float_180_angle_delta_cols)
            relative_90_cols, _ = split_delta_columns(self.float_90_angle_delta_cols)
            return relative_cols + relative_180_cols + relative_90_cols + \
                   self.float_180_angle_cols + self.float_90_angle_cols + self.categorical_cols
        if self.all_cols_ is None:
            relative_cols, _ = split_delta_columns(self.float_delta_cols)
            relative_180_cols, _ = split_delta_columns(self.float_180_angle_delta_cols)
            relative_90_cols, _ = split_delta_columns(self.float_90_angle_delta_cols)
            self.all_cols_ = self.float_standard_cols + relative_cols + \
                             self.float_180_angle_cols + relative_180_cols + \
                             self.float_90_angle_cols + relative_90_cols + \
                             self.categorical_cols
        return self.all_cols_

    def delta_float_column_names(self) -> List[str]:
        if self.delta_float_column_names_ is None:
            self.delta_float_column_names_, _ = split_delta_columns(self.float_delta_cols)
        return self.delta_float_column_names_

    def delta_180_angle_column_names(self) -> List[str]:
        if self.delta_180_angle_column_names_ is None:
            self.delta_180_angle_column_names_, _ = split_delta_columns(self.float_180_angle_delta_cols)
        return self.delta_180_angle_column_names_

    def delta_90_angle_column_names(self) -> List[str]:
        if self.delta_90_angle_column_names_ is None:
            self.delta_90_angle_column_names_, _ = split_delta_columns(self.float_90_angle_delta_cols)
        return self.delta_90_angle_column_names_

    def delta_float_target_column_names(self) -> List[str]:
        if self.delta_float_target_column_names_ is None:
            self.delta_float_target_column_names_ = target_delta_columns(self.float_delta_cols +
                                                                         self.float_180_angle_delta_cols +
                                                                         self.float_90_angle_delta_cols)
        return self.delta_float_target_column_names_

    def sin_cos_encoded_angles(self) -> List[str]:
        return self.float_180_angle_cols + self.delta_180_angle_column_names() + \
               self.float_90_angle_cols + self.delta_90_angle_column_names()


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
    def delta_inverse(self, delta_value: torch.Tensor, reference_value: torch.Tensor):
        pass


class PTMeanStdColumnTransformer(PTColumnTransformer):
    # FLOAT_STANDARD data
    cpu_means: torch.Tensor
    cpu_standard_deviations: torch.Tensor
    means: torch.Tensor
    standard_deviations: torch.Tensor

    pt_ct_type: ColumnTransformerType = ColumnTransformerType.FLOAT_STANDARD

    def __init__(self, means: torch.Tensor, standard_deviations: torch.Tensor):
        self.cpu_means = means.view(1, -1)
        self.cpu_standard_deviations = standard_deviations.view(1, -1)
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

    def delta_inverse(self, delta_value: torch.Tensor, reference_value: torch.Tensor):
        raise NotImplementedError


class PTDeltaMeanStdColumnTransformer(PTColumnTransformer):
    # FLOAT_STANDARD data
    cpu_delta_means: torch.Tensor
    cpu_delta_standard_deviations: torch.Tensor
    delta_means: torch.Tensor
    delta_standard_deviations: torch.Tensor

    pt_ct_type: ColumnTransformerType = ColumnTransformerType.FLOAT_STANDARD

    def __init__(self, delta_means: torch.Tensor, delta_standard_deviations: torch.Tensor):
        self.cpu_delta_means = delta_means.view(1, -1)
        self.cpu_delta_standard_deviations = delta_standard_deviations.view(1, -1)
        # done so columsn that all equal mean are 0, 253 is sentinel value,
        # doesn't matter what value, sub by mean will make it equal 0.
        # NOTE TO SELF: IT DOES MATTER, TRAINING VALUES WON'T BE 0, AND LARGE STD DEV PREVENTS THEM FROM CONVERGING
        # DUE TO BAD LOSS
        self.cpu_delta_standard_deviations[self.cpu_delta_standard_deviations == 0.] = \
            torch.finfo(delta_standard_deviations.dtype).smallest_normal
        self.delta_means = self.cpu_delta_means.to(CUDA_DEVICE_STR)
        self.delta_standard_deviations = self.cpu_delta_standard_deviations.to(CUDA_DEVICE_STR)

    def convert(self, offset_value: torch.Tensor):
        raise NotImplementedError

    def inverse(self, offset_value: torch.Tensor):
        raise NotImplementedError

    def delta_convert(self, relative_value: torch.Tensor, reference_value: torch.Tensor):
        delta_value = relative_value - reference_value
        # make delta values relative to each other rather than reference value
        delta_value[:, 1:] -= torch.roll(delta_value, 1, 1)[:, 1:]
        if relative_value.device.type == CPU_DEVICE_STR:
            return (delta_value - self.cpu_delta_means) / self.cpu_delta_standard_deviations
        else:
            return (delta_value - self.delta_means) / self.delta_standard_deviations

    def delta_inverse(self, delta_value: torch.Tensor, reference_value: torch.Tensor):
        if delta_value.device.type == CPU_DEVICE_STR:
            return (delta_value * self.cpu_delta_standard_deviations) + self.cpu_delta_means + reference_value
        else:
            return (delta_value * self.delta_standard_deviations) + self.delta_means + reference_value


class PT180AngleColumnTransformer(PTMeanStdColumnTransformer):
    pt_ct_type: ColumnTransformerType = ColumnTransformerType.FLOAT_180_ANGLE

    def __init__(self, num_cols: int, sin_mean: float, sin_std: float, cos_mean: float, cos_std: float):
        means = torch.tensor([sin_mean, cos_mean] * num_cols)
        stds = torch.tensor([sin_std, cos_std] * num_cols)
        super().__init__(means, stds)


    def convert(self, value: torch.Tensor):
        sin_value = torch.sin(torch.deg2rad(value))
        cos_value = torch.cos(torch.deg2rad(value))
        stack_value = torch.stack([sin_value, cos_value], dim=2)
        return super().convert(torch.flatten(stack_value, start_dim=1))

    def inverse(self, value: torch.Tensor):
        value = super().inverse(value)
        value = torch.unflatten(value, dim=1, sizes=(-1, 2))
        return torch.rad2deg(torch.atan2(value[:, :, 0], value[:, :, 1]))

    def delta_convert(self, relative_value: torch.Tensor, reference_value: torch.Tensor):
        raise NotImplementedError

    def delta_inverse(self, delta_value: torch.Tensor, reference_value: torch.Tensor):
        raise NotImplementedError


class PTDelta180AngleColumnTransformer(PT180AngleColumnTransformer):
    pt_ct_type: ColumnTransformerType = ColumnTransformerType.FLOAT_180_ANGLE_DELTA

    def __init__(self, num_cols: int, sin_mean: float, sin_std: float, cos_mean: float, cos_std: float):
        super().__init__(num_cols, sin_mean, sin_std, cos_mean, cos_std)

    def convert(self, value: torch.Tensor):
        raise NotImplementedError

    def inverse(self, value: torch.Tensor):
        raise NotImplementedError

    def delta_convert(self, relative_value: torch.Tensor, reference_value: torch.Tensor):
        delta_value = relative_value - reference_value
        # make delta values relative to each other rather than reference value
        delta_value[:, 1:] -= torch.roll(delta_value, 1, 1)[:, 1:]
        return super().convert(delta_value)

    def delta_inverse(self, delta_value: torch.Tensor, reference_value: torch.Tensor):
        angular_delta_value = super().inverse(delta_value)
        return angular_delta_value + reference_value


class PT90AngleColumnTransformer(PT180AngleColumnTransformer):
    pt_ct_type: ColumnTransformerType = ColumnTransformerType.FLOAT_90_ANGLE

    def __init__(self, num_cols: int, sin_mean: float, sin_std: float, cos_mean: float, cos_std: float):
        super().__init__(num_cols, sin_mean, sin_std, cos_mean, cos_std)

    def inverse(self, value: torch.Tensor):
        value = PTMeanStdColumnTransformer.inverse(self, value)
        value = torch.unflatten(value, dim=1, sizes=(-1, 2))
        return torch.rad2deg(torch.atan(value[:, :, 0] / value[:, :, 1]))

    def delta_convert(self, relative_value: torch.Tensor, reference_value: torch.Tensor):
        raise NotImplementedError

    def delta_inverse(self, delta_value: torch.Tensor, reference_value: torch.Tensor):
        raise NotImplementedError


class PTDelta90AngleColumnTransformer(PT90AngleColumnTransformer):
    pt_ct_type: ColumnTransformerType = ColumnTransformerType.FLOAT_90_ANGLE_DELTA

    def __init__(self, num_cols: int, sin_mean: float, sin_std: float, cos_mean: float, cos_std: float):
        super().__init__(num_cols, sin_mean, sin_std, cos_mean, cos_std)

    def convert(self, value: torch.Tensor):
        raise NotImplementedError

    def inverse(self, value: torch.Tensor):
        raise NotImplementedError

    def delta_convert(self, relative_value: torch.Tensor, reference_value: torch.Tensor):
        delta_value = relative_value - reference_value
        # make delta values relative to each other rather than reference value
        delta_value[:, 1:] -= torch.roll(delta_value, 1, 1)[:, 1:]
        return super().convert(delta_value)

    def delta_inverse(self, delta_value: torch.Tensor, reference_value: torch.Tensor):
        delta_value = super().inverse(delta_value)
        return delta_value + reference_value


@dataclass
class PTOneHotColumnTransformer(PTColumnTransformer):
    # CATEGORICAL data
    col_name: str
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

    angular_mean: float
    angular_std: float

    angular_sin_mean: float
    angular_sin_std: float
    angular_cos_mean: float
    angular_cos_std: float

    def __init__(self, input_types: ColumnTypes, output_types: ColumnTypes, all_data_df: pd.DataFrame):
        self.input_types = input_types
        self.output_types = output_types

        self.compute_angular_mean_std(all_data_df)

        self.input_ct_pts = self.create_pytorch_column_transformers(self.input_types, all_data_df)
        self.output_ct_pts = self.create_pytorch_column_transformers(self.output_types, all_data_df)

    def compute_angular_mean_std(self, all_data_df: pd.DataFrame):
        angular_standard_cols = self.input_types.float_angular_standard_cols + \
                                self.output_types.float_angular_standard_cols
        angular_relative_cols, _ = split_delta_columns(self.input_types.float_angular_delta_cols +
                                                       self.output_types.float_angular_delta_cols)
        angular_angle_encoded = angular_standard_cols + angular_relative_cols
        angular_sin_cos_encoded = self.input_types.sin_cos_encoded_angles() + self.output_types.sin_cos_encoded_angles()
        if angular_angle_encoded:
            self.angular_mean = np.mean(all_data_df.loc[:, angular_angle_encoded].to_numpy()).item()
            self.angular_std = np.std(all_data_df.loc[:, angular_angle_encoded].to_numpy()).item()
        if angular_sin_cos_encoded:
            self.angular_sin_mean = np.mean(np.sin(np.deg2rad(all_data_df.loc[:, angular_sin_cos_encoded].to_numpy()))).item()
            self.angular_sin_std = np.std(np.sin(np.deg2rad(all_data_df.loc[:, angular_sin_cos_encoded].to_numpy()))).item()
            self.angular_cos_mean = np.mean(np.cos(np.deg2rad(all_data_df.loc[:, angular_sin_cos_encoded].to_numpy()))).item()
            self.angular_cos_std = np.std(np.cos(np.deg2rad(all_data_df.loc[:, angular_sin_cos_encoded].to_numpy()))).item()


    def compute_mean_per_column(self, all_cols: List[str], angular_cols: List[str], all_data_df: pd.DataFrame) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        means = all_data_df.loc[:, all_cols].mean()
        stds = all_data_df.loc[:, all_cols].std()
        for angular_col in angular_cols:
            means[angular_col] = self.angular_mean
            stds[angular_col] = self.angular_std
        return torch.Tensor(means), torch.Tensor(stds)

    def create_pytorch_column_transformers(self, types: ColumnTypes, all_data_df: pd.DataFrame) -> \
            List[PTColumnTransformer]:
        result: List[PTColumnTransformer] = []
        if types.float_standard_cols:
            means, stds = self.compute_mean_per_column(types.float_standard_cols, types.float_angular_standard_cols,
                                                       all_data_df)
            result.append(PTMeanStdColumnTransformer(means, stds))
        if types.float_delta_cols:
            relative_cols, reference_cols = split_delta_columns(types.float_delta_cols)
            delta_np = all_data_df.loc[:, relative_cols].to_numpy() - \
                       all_data_df.loc[:, reference_cols].to_numpy()
            angular_relative_cols, _ = split_delta_columns(types.float_angular_delta_cols)
            delta_df = pd.DataFrame(delta_np, columns=relative_cols)
            means, stds = self.compute_mean_per_column(relative_cols, angular_relative_cols, delta_df)
            result.append(PTDeltaMeanStdColumnTransformer(means, stds))
        if types.float_180_angle_cols:
            result.append(PT180AngleColumnTransformer(len(types.float_180_angle_cols),
                                                      self.angular_sin_mean, self.angular_sin_std,
                                                      self.angular_cos_mean, self.angular_cos_std))
        if types.float_180_angle_delta_cols:
            result.append(PTDelta180AngleColumnTransformer(len(types.float_180_angle_delta_cols),
                                                           self.angular_sin_mean, self.angular_sin_std,
                                                           self.angular_cos_mean, self.angular_cos_std))
        if types.float_90_angle_cols:
            result.append(PT90AngleColumnTransformer(len(types.float_90_angle_cols),
                                                     self.angular_sin_mean, self.angular_sin_std,
                                                     self.angular_cos_mean, self.angular_cos_std))
        if types.float_90_angle_delta_cols:
            result.append(PTDelta90AngleColumnTransformer(len(types.float_90_angle_delta_cols),
                                                          self.angular_sin_mean, self.angular_sin_std,
                                                          self.angular_cos_mean, self.angular_cos_std))
        for name in types.categorical_cols:
            result.append(PTOneHotColumnTransformer(name, types.num_cats_per_col[name]))
        return result

    @cache
    def get_name_ranges(self, input: bool, transformed: bool, types: frozenset[ColumnTransformerType] = ALL_TYPES,
                        only_wrap_cols: bool = False) -> List[range]:
        result: List[range] = []
        cur_start: int = 0

        column_types: ColumnTypes = self.input_types if input else self.output_types
        cts: List[PTColumnTransformer] = self.input_ct_pts if input else self.output_ct_pts

        angle_columns = 2 if transformed else 1

        for col_name in column_types.float_standard_cols:
            if ColumnTransformerType.FLOAT_STANDARD in types and \
                    (not only_wrap_cols or col_name in column_types.float_180_wrap_cols):
                result.append(range(cur_start, cur_start + 1))
            cur_start += 1

        for col_name in column_types.float_delta_cols:
            if ColumnTransformerType.FLOAT_DELTA in types and \
                    (not only_wrap_cols or col_name in column_types.float_180_wrap_cols):
                result.append(range(cur_start, cur_start + 1))
            cur_start += 1

        for _ in column_types.float_180_angle_cols:
            if ColumnTransformerType.FLOAT_180_ANGLE in types and \
                    (not only_wrap_cols or col_name in column_types.float_180_wrap_cols):
                result.append(range(cur_start, cur_start + angle_columns))
            cur_start += angle_columns

        for _ in column_types.float_180_angle_delta_cols:
            if ColumnTransformerType.FLOAT_180_ANGLE_DELTA in types and \
                    (not only_wrap_cols or col_name in column_types.float_180_wrap_cols):
                result.append(range(cur_start, cur_start + angle_columns))
            cur_start += angle_columns

        for _ in column_types.float_90_angle_cols:
            if ColumnTransformerType.FLOAT_90_ANGLE in types and \
                    (not only_wrap_cols or col_name in column_types.float_180_wrap_cols):
                result.append(range(cur_start, cur_start + angle_columns))
            cur_start += angle_columns

        for _ in column_types.float_90_angle_delta_cols:
            if ColumnTransformerType.FLOAT_90_ANGLE_DELTA in types and \
                    (not only_wrap_cols or col_name in column_types.float_180_wrap_cols):
                result.append(range(cur_start, cur_start + angle_columns))
            cur_start += angle_columns

        for ct in cts:
            if ct.pt_ct_type == ColumnTransformerType.CATEGORICAL:
                if transformed:
                    if ColumnTransformerType.CATEGORICAL in types and not only_wrap_cols:
                        result.append(range(cur_start, cur_start + ct.num_classes))
                    cur_start += ct.num_classes
                else:
                    if ColumnTransformerType.CATEGORICAL in types and not only_wrap_cols:
                        result.append(range(cur_start, cur_start + 1))
                    cur_start += 1

        return result

    @cache
    def get_name_ranges_dict(self, input: bool, transformed: bool) -> Dict[str, range]:
        result: Dict[str, range] = {}
        cur_start: int = 0

        column_types: ColumnTypes = self.input_types if input else self.output_types
        cts: List[PTColumnTransformer] = self.input_ct_pts if input else self.output_ct_pts

        angle_columns = 2 if transformed else 1

        for col_name in column_types.float_standard_cols:
            result[col_name] = range(cur_start, cur_start + 1)
            cur_start += 1

        for col_name in column_types.delta_float_column_names():
            result[col_name] = range(cur_start, cur_start + 1)
            cur_start += 1

        for col_name in column_types.float_180_angle_cols:
            result[col_name] = range(cur_start, cur_start + angle_columns)
            cur_start += angle_columns

        for col_name in column_types.delta_180_angle_column_names():
            result[col_name] = range(cur_start, cur_start + angle_columns)
            cur_start += angle_columns

        for col_name in column_types.float_90_angle_cols:
            result[col_name] = range(cur_start, cur_start + angle_columns)
            cur_start += angle_columns

        for col_name in column_types.delta_90_angle_column_names():
            result[col_name] = range(cur_start, cur_start + angle_columns)
            cur_start += angle_columns

        for ct in cts:
            if ct.pt_ct_type == ColumnTransformerType.CATEGORICAL:
                if transformed:
                    result[ct.col_name] = range(cur_start, cur_start + ct.num_classes)
                    cur_start += ct.num_classes
                else:
                    result[ct.col_name] = range(cur_start, cur_start + 1)
                    cur_start += 1

        return result

    @cache
    def get_name_ranges_in_time_range(self, input: bool, transformed: bool,
                                      time_offset_range: range, include_non_temporal: bool,
                                      include_cat: bool) -> Tuple[List[int], List[str]]:
        result: List[int] = []
        result_str: List[str] = []
        cur_start: int = 0

        column_types: ColumnTypes = self.input_types if input else self.output_types
        cts: List[PTColumnTransformer] = self.input_ct_pts if input else self.output_ct_pts

        angle_columns = 2 if transformed else 1

        for col_name in column_types.float_standard_cols:
            if column_types.col_time_offsets[col_name].offset_valid_for_range(time_offset_range, include_non_temporal):
                result.append(cur_start)
                result_str.append(col_name)
            cur_start += 1

        for col_name in column_types.delta_float_column_names():
            if column_types.col_time_offsets[col_name].offset_valid_for_range(time_offset_range, include_non_temporal):
                result.append(cur_start)
                result_str.append(col_name)
            cur_start += 1

        for col_name in column_types.float_180_angle_cols:
            if column_types.col_time_offsets[col_name].offset_valid_for_range(time_offset_range, include_non_temporal):
                for i in range(angle_columns):
                    result.append(cur_start + i)
                    result_str.append(col_name)
            cur_start += angle_columns

        for col_name in column_types.delta_180_angle_column_names():
            if column_types.col_time_offsets[col_name].offset_valid_for_range(time_offset_range, include_non_temporal):
                for i in range(angle_columns):
                    result.append(cur_start + i)
                    result_str.append(col_name)
            cur_start += angle_columns

        for col_name in column_types.float_90_angle_cols:
            if col_name == 'victim relative first head cur head view angle y (t+13)':
                x = 1
            if column_types.col_time_offsets[col_name].offset_valid_for_range(time_offset_range, include_non_temporal):
                for i in range(angle_columns):
                    result.append(cur_start + i)
                    result_str.append(col_name)
            cur_start += angle_columns

        for col_name in column_types.delta_90_angle_column_names():
            if column_types.col_time_offsets[col_name].offset_valid_for_range(time_offset_range, include_non_temporal):
                for i in range(angle_columns):
                    result.append(cur_start + i)
                    result_str.append(col_name)
            cur_start += angle_columns

        if include_cat:
            for ct in cts:
                if ct.pt_ct_type == ColumnTransformerType.CATEGORICAL:
                    if transformed:
                        if column_types.col_time_offsets[ct.col_name].offset_valid_for_range(time_offset_range,
                                                                                             include_non_temporal):
                            for i in range(ct.num_classes):
                                result.append(cur_start + i)
                                result_str.append(ct.col_name)
                        cur_start += ct.num_classes
                    else:
                        if column_types.col_time_offsets[ct.col_name].offset_valid_for_range(time_offset_range,
                                                                                             include_non_temporal):
                            result.append(cur_start)
                            result_str.append(ct.col_name)
                        cur_start += 1

        return result, result_str

    # given the locations of columns in an input tensor that are used as reference for producing delta columns
    def get_input_delta_reference_positions(self, types: ColumnTypes, delta_type: ColumnTransformerType) -> List[int]:
        result: List[int] = []

        if delta_type == ColumnTransformerType.FLOAT_DELTA:
            _, output_reference_cols = split_delta_columns(types.float_delta_cols)
            input_reference_cols = self.input_types.float_standard_cols
        elif delta_type == ColumnTransformerType.FLOAT_180_ANGLE_DELTA:
            _, output_reference_cols = split_delta_columns(types.float_180_angle_delta_cols)
            input_reference_cols = self.input_types.float_180_angle_cols
        else:
            _, output_reference_cols = split_delta_columns(types.float_90_angle_delta_cols)
            input_reference_cols = self.input_types.float_90_angle_cols

        cur_col_index = 0
        all_cols = self.input_types.column_names()
        # input has baseline to compare to
        for reference_col in output_reference_cols:
            if reference_col in input_reference_cols:
                result.append(all_cols.index(reference_col))
            cur_col_index += 1

        return result

    def transform_columns(self, input: bool, x: torch.Tensor, x_input: torch.Tensor) -> torch.Tensor:
        cur_device = x.device
        x = x.to(CPU_DEVICE_STR)
        x_input = x_input.to(CPU_DEVICE_STR)

        uncat_result: List[torch.Tensor] = []

        ct_pts = self.input_ct_pts if input else self.output_ct_pts
        types = self.input_types if input else self.output_types

        ct_offset = 0
        x_float_standard_name_ranges = self.get_name_ranges(input, False,
                                                            frozenset({ColumnTransformerType.FLOAT_STANDARD}))
        if x_float_standard_name_ranges:
            x_floats = x[:, x_float_standard_name_ranges[0].start:x_float_standard_name_ranges[-1].stop]
            uncat_result.append(ct_pts[ct_offset].convert(x_floats))
            ct_offset += 1

        x_float_delta_name_ranges = self.get_name_ranges(input, False, frozenset({ColumnTransformerType.FLOAT_DELTA}))
        if x_float_delta_name_ranges:
            x_relative_floats = x[:, x_float_delta_name_ranges[0].start:x_float_delta_name_ranges[-1].stop]
            x_float_delta_reference_positions = self.get_input_delta_reference_positions(types,
                                                                                         ColumnTransformerType.FLOAT_DELTA)
            x_reference_floats = x_input[:, x_float_delta_reference_positions]
            uncat_result.append(ct_pts[ct_offset].delta_convert(x_relative_floats, x_reference_floats))
            ct_offset += 1

        x_float_180_angle_name_ranges = self.get_name_ranges(input, False,
                                                             frozenset({ColumnTransformerType.FLOAT_180_ANGLE}))
        if x_float_180_angle_name_ranges:
            x_floats_180_angle = x[:, x_float_180_angle_name_ranges[0].start:x_float_180_angle_name_ranges[-1].stop]
            uncat_result.append(ct_pts[ct_offset].convert(x_floats_180_angle))
            ct_offset += 1

        x_float_180_angle_delta_name_ranges = self.get_name_ranges(input, False,
                                                                   frozenset(
                                                                       {ColumnTransformerType.FLOAT_180_ANGLE_DELTA}))
        if x_float_180_angle_delta_name_ranges:
            x_relative_angles = x[:, x_float_180_angle_delta_name_ranges[0].start:x_float_180_angle_delta_name_ranges[
                -1].stop]
            x_float_delta_reference_positions = self.get_input_delta_reference_positions(types,
                                                                                         ColumnTransformerType.FLOAT_180_ANGLE_DELTA)
            x_reference_angles = x_input[:, x_float_delta_reference_positions]
            uncat_result.append(ct_pts[ct_offset].delta_convert(x_relative_angles, x_reference_angles))
            ct_offset += 1

        x_float_90_angle_name_ranges = self.get_name_ranges(input, False,
                                                            frozenset({ColumnTransformerType.FLOAT_90_ANGLE}))
        if x_float_90_angle_name_ranges:
            x_floats_90_angle = x[:, x_float_90_angle_name_ranges[0].start:x_float_90_angle_name_ranges[-1].stop]
            uncat_result.append(ct_pts[ct_offset].convert(x_floats_90_angle))
            ct_offset += 1

        x_float_90_angle_delta_name_ranges = self.get_name_ranges(input, False,
                                                                  frozenset(
                                                                      {ColumnTransformerType.FLOAT_90_ANGLE_DELTA}))
        if x_float_90_angle_delta_name_ranges:
            x_relative_angles = x[:,
                                x_float_90_angle_delta_name_ranges[0].start:x_float_90_angle_delta_name_ranges[-1].stop]
            x_float_delta_reference_positions = self.get_input_delta_reference_positions(types,
                                                                                         ColumnTransformerType.FLOAT_90_ANGLE_DELTA)
            x_reference_angles = x_input[:, x_float_delta_reference_positions]
            uncat_result.append(ct_pts[ct_offset].delta_convert(x_relative_angles, x_reference_angles))
            ct_offset += 1

        x_categorical_name_ranges = self.get_name_ranges(input, False, frozenset({ColumnTransformerType.CATEGORICAL}))
        for i, categorical_name_range in enumerate(x_categorical_name_ranges):
            uncat_result.append(ct_pts[i + ct_offset].convert(x[:, categorical_name_range]))

        return torch.cat(uncat_result, dim=1).to(cur_device)

    def untransform_columns(self, input: bool, x: torch.Tensor, x_input: torch.Tensor) -> torch.Tensor:
        cur_device = x.device
        x = x.to(CPU_DEVICE_STR)
        x_input = x_input.to(CPU_DEVICE_STR)

        uncat_result: List[torch.Tensor] = []

        ct_pts = self.input_ct_pts if input else self.output_ct_pts
        types = self.input_types if input else self.output_types

        ct_offset = 0
        x_float_name_ranges = self.get_name_ranges(input, True, frozenset({ColumnTransformerType.FLOAT_STANDARD}))
        if x_float_name_ranges:
            x_floats = x[:, x_float_name_ranges[0].start:x_float_name_ranges[-1].stop]
            uncat_result.append(ct_pts[ct_offset].inverse(x_floats))
            ct_offset += 1

        x_float_delta_name_ranges = self.get_name_ranges(input, True, frozenset({ColumnTransformerType.FLOAT_DELTA}))
        if x_float_delta_name_ranges:
            x_relative_floats = x[:, x_float_delta_name_ranges[0].start:x_float_delta_name_ranges[-1].stop]
            x_float_delta_reference_positions = self.get_input_delta_reference_positions(types,
                                                                                         ColumnTransformerType.FLOAT_DELTA)
            x_reference_floats = x_input[:, x_float_delta_reference_positions]
            uncat_result.append(ct_pts[ct_offset].delta_inverse(x_relative_floats, x_reference_floats))
            ct_offset += 1

        x_float_180_angle_name_ranges = self.get_name_ranges(input, True,
                                                             frozenset({ColumnTransformerType.FLOAT_180_ANGLE}))
        if x_float_180_angle_name_ranges:
            x_floats_180_angle = x[:, x_float_180_angle_name_ranges[0].start:x_float_180_angle_name_ranges[-1].stop]
            uncat_result.append(ct_pts[ct_offset].inverse(x_floats_180_angle))
            ct_offset += 1

        x_float_180_delta_name_ranges = self.get_name_ranges(input, True,
                                                             frozenset({ColumnTransformerType.FLOAT_180_ANGLE_DELTA}))
        if x_float_180_delta_name_ranges:
            x_relative_floats = x[:, x_float_180_delta_name_ranges[0].start:x_float_180_delta_name_ranges[-1].stop]
            x_float_delta_reference_positions = self.get_input_delta_reference_positions(types,
                                                                                         ColumnTransformerType.FLOAT_180_ANGLE_DELTA)
            x_reference_floats = x_input[:, x_float_delta_reference_positions]
            uncat_result.append(ct_pts[ct_offset].delta_inverse(x_relative_floats, x_reference_floats))
            ct_offset += 1

        x_float_90_angle_name_ranges = self.get_name_ranges(input, True,
                                                            frozenset({ColumnTransformerType.FLOAT_90_ANGLE}))
        if x_float_90_angle_name_ranges:
            x_floats_90_angle = x[:, x_float_90_angle_name_ranges[0].start:x_float_90_angle_name_ranges[-1].stop]
            uncat_result.append(ct_pts[ct_offset].inverse(x_floats_90_angle))
            ct_offset += 1

        x_float_90_delta_name_ranges = self.get_name_ranges(input, True,
                                                            frozenset({ColumnTransformerType.FLOAT_90_ANGLE_DELTA}))
        if x_float_90_delta_name_ranges:
            x_relative_floats = x[:, x_float_90_delta_name_ranges[0].start:x_float_90_delta_name_ranges[-1].stop]
            x_float_delta_reference_positions = self.get_input_delta_reference_positions(types,
                                                                                         ColumnTransformerType.FLOAT_90_ANGLE_DELTA)
            x_reference_floats = x_input[:, x_float_delta_reference_positions]
            uncat_result.append(ct_pts[ct_offset].delta_inverse(x_relative_floats, x_reference_floats))
            ct_offset += 1

        x_categorical_name_ranges = self.get_name_ranges(input, True, frozenset({ColumnTransformerType.CATEGORICAL}))
        for i, categorical_name_range in enumerate(x_categorical_name_ranges):
            uncat_result.append(ct_pts[i + ct_offset].inverse(x[:, categorical_name_range]))

        return torch.cat(uncat_result, dim=1).to(cur_device)

    def get_untransformed_value(self, x: Union[torch.Tensor, ModelOutput], col_name: str, input: bool) -> float:
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
            return x[1][col_ranges[col_index].start].item()

    def get_untransformed_values(self, x: Union[torch.Tensor, ModelOutput], input: bool) -> Dict:
        col_names = self.input_types.column_names() if input else self.output_types.column_names()
        col_ranges = self.get_name_ranges(input, False)

        x_tensor: torch.Tensor = x if input else x[1]
        result = {}

        for col_name, col_range in zip(col_names, col_ranges):
            result[col_name] = x_tensor[col_range.start].item()

        return result

    def set_untransformed_input_value(self, x: torch.Tensor, col_name: str, value: float):
        col_names = self.input_types.column_names() if input else self.output_types.column_names()
        col_ranges = self.get_name_ranges(True, False)
        col_index = 0
        for i, col_name_ in enumerate(col_names):
            if col_name_ == col_name:
                col_index = i
                break

        x[col_ranges[col_index].start] = value

    def get_column_transformer_by_type(self, input: bool, ct_type: ColumnTransformerType) -> List[PTColumnTransformer]:
        column_transformers = self.input_ct_pts if input else self.output_ct_pts
        valid_cts = []
        for column_transformer in column_transformers:
            if column_transformer.pt_ct_type == ct_type:
                valid_cts.append(column_transformer)
        return valid_cts


def get_transformed_outputs(x: ModelOutput) -> torch.Tensor:
    return x[0]


def get_untransformed_outputs(x: ModelOutput):
    return x[1]
