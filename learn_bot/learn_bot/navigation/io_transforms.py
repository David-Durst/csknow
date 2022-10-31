import torch
import pandas as pd
from typing import List

from learn_bot.engagement_aim.column_management import PTMeanStdColumnTransformer, ColumnTransformerType, \
    CPU_DEVICE_STR, CUDA_DEVICE_STR, ColumnTypes, PTColumnTransformer, IOColumnTransformers

PRIOR_TICKS = -4
FUTURE_TICKS = 1
CUR_TICK = 1

class PTImageTransformer(PTMeanStdColumnTransformer):
    # IMAGE data
    # same conversion and inversion, but means are not flattened out like
    # they are from tabular data
    # technically not a column transformer, as images aren't columns, but the infrastructure is useful
    pt_ct_type: ColumnTransformerType = ColumnTransformerType.FLOAT_STANDARD

    def __init__(self, means: torch.Tensor, standard_deviations: torch.Tensor):
        self.cpu_means = means
        self.cpu_standard_deviations = standard_deviations
        self.means = self.cpu_means.to(CUDA_DEVICE_STR)
        self.standard_deviations = self.cpu_standard_deviations.to(CUDA_DEVICE_STR)


class IOColumnAndImageTransformers(IOColumnTransformers):
    input_types: ColumnTypes
    output_types: ColumnTypes

    input_img_transformer: PTImageTransformer
    input_ct_pts: List[PTColumnTransformer]
    output_ct_pts: List[PTColumnTransformer]

    def __init__(self, input_types: ColumnTypes, output_types: ColumnTypes, non_img_df: pd.DataFrame,
                 sample_images_tensor: torch.tensor = torch.empty((1,1))):
        super().__init__(input_types, output_types, non_img_df)

        # last two dims of image are ones to reduce, as computing mean per channel
        # last two chanels are row/width after channel counter
        channels_to_reduce = [len(sample_images_tensor.shape) - 2, len(sample_images_tensor.shape) - 1]
        img_std, img_mean = torch.std_mean(sample_images_tensor, dim=channels_to_reduce, keepdim=True)
        self.input_img_transformer = PTImageTransformer(img_mean, img_std)

    def transform_images_and_columns(self, input: bool, non_img_tensor: torch.Tensor,
                                     img_tensor: torch.Tensor) -> torch.Tensor:
        super().transform_columns(input, non_img_tensor)
        # fix this when I call it, need to broadcast non_image_tensor to all parts of img_tensor
        raise NotImplementedError

    # for now, assuming no images out, so can just call parent untransform_columns
