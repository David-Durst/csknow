//
// Created by durst on 12/28/22.
//

#include "bots/analysis/pytorch_utils.h"

std::string print2DTensor(torch::Tensor & tensor) {
    std::ostringstream ss;
    for (int64_t i = 0; i < tensor.size(0); i++) {
        for (int64_t j = 0; j < tensor.size(1); j++) {
            ss << tensor[i][j].item<float>() << ",";
        }
        ss << std::endl;
    }
    return ss.str();
}