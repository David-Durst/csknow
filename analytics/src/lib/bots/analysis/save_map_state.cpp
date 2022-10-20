//
// Created by durst on 10/19/22.
//

#include "bots/analysis/save_map_state.h"
#include <opencv2/opencv.hpp>

namespace csknow {

    MapState::MapState(const VisPoints & visPoints) : data{} {
        for (const auto & cellVisPoint : visPoints.getCellVisPoints()) {
            data[cellVisPoint.cellDiscreteCoordinates[0]][cellVisPoint.cellDiscreteCoordinates[1]] =
                std::numeric_limits<uint8_t>::max();
        }
    }

    void MapState::saveMapState(const fs::path &path) {
        cv::Mat cvMapState = cv::Mat(NAV_CELLS_PER_ROW, NAV_CELLS_PER_ROW, CV_8U, data[0].data());
        cv::imwrite(path.string(), cvMapState);
    }

    MapState & MapState::operator+=(const MapState & value) {
        for (size_t i = 0; i < data.size(); i++) {
            for (size_t j = 0; j < data[i].size(); j++) {
                data[i][j] += value.data[i][j];
            }
        }
        return *this;
    }

    MapState & MapState::operator+=(const uint8_t & value) {
        for (size_t i = 0; i < data.size(); i++) {
            for (size_t j = 0; j < data[i].size(); j++) {
                data[i][j] += value;
            }
        }
        return *this;
    }

    MapState & MapState::operator-=(const uint8_t & value) {
        for (size_t i = 0; i < data.size(); i++) {
            for (size_t j = 0; j < data[i].size(); j++) {
                data[i][j] -= value;
            }
        }
        return *this;
    }

    MapState & MapState::operator*=(const uint8_t & value) {
        for (size_t i = 0; i < data.size(); i++) {
            for (size_t j = 0; j < data[i].size(); j++) {
                data[i][j] *= value;
            }
        }
        return *this;
    }

    MapState & MapState::operator/=(const uint8_t & value) {
        for (size_t i = 0; i < data.size(); i++) {
            for (size_t j = 0; j < data[i].size(); j++) {
                data[i][j] /= value;
            }
        }
        return *this;
    }

    MapState & MapState::conv(const conv_matrix & mat) {
        auto oldData(data);
        uint16_t norm = 0;
        for (int64_t i = 0; i < CONV_SIZE; i++) {
            for (int64_t j = 0; j < CONV_SIZE; j++) {
                norm += mat[i][j];
            }
        }
        for (int64_t i = 0; i < static_cast<int64_t>(mat.size()); i++) {
            for (int64_t j = 0; j < static_cast<int64_t>(mat[i].size()); j++) {
                data[i][j] = 0;
                for (int64_t ii = (-1 * CONV_SIZE / 2); ii <= CONV_SIZE / 2; ii++) {
                    for (int64_t jj = (-1 * CONV_SIZE / 2); jj <= CONV_SIZE / 2; jj++) {
                        int64_t sum_i = i + ii;
                        int64_t sum_j = j + jj;
                        if (sum_i < 0 || sum_i > static_cast<int64_t>(mat.size()) ||
                            sum_j < 0 || sum_j > static_cast<int64_t>(mat[i].size())) {
                            data[i][j] += 0;
                        }
                        else {
                            data[i][j] += oldData[sum_i][sum_j] * mat[ii + CONV_SIZE / 2][jj + CONV_SIZE / 2];
                        }
                    }
                }
                data[i][j] /= norm;
            }
        }
    }
}