//
// Created by durst on 10/19/22.
//

#include "bots/analysis/save_map_state.h"
#include <opencv2/opencv.hpp>

namespace csknow {

    void MapState::saveNewMapState(const CellBits & value, const fs::path & path) {
        *this = value;
        saveMapState(path);
    }

    void MapState::saveNewMapState(const vector<uint8_t> & value, const fs::path & path) {
        *this = value;
        saveMapState(path);
    }

    void MapState::saveMapState(const fs::path &path) {
        cv::Mat cvMapState = cv::Mat(NAV_CELLS_PER_ROW, NAV_CELLS_PER_ROW, CV_8U, data[0].data());
        cv::imwrite(path.string(), cvMapState);
    }

    MapState & MapState::operator=(const CellBits & value) {
        memset(data.data(), 0, data.size() * data[0].size());
        const auto & cellVisPoints = visPoints.getCellVisPoints();
        double maxYNum = visPoints.getMaxCellNumbersByDim()[1];
        for (size_t i = 0; i < cellVisPoints.size(); i++) {
            if (value[i]) {
                data[maxYNum - cellVisPoints[i].cellDiscreteCoordinates[1]]
                    [cellVisPoints[i].cellDiscreteCoordinates[0]] = std::numeric_limits<uint8_t>::max();
            }
        }
        return *this;
    }

    MapState & MapState::operator=(const vector<uint8_t> & value) {
        memset(data.data(), 0, data.size() * data[0].size());
        const auto & cellVisPoints = visPoints.getCellVisPoints();
        double maxYNum = visPoints.getMaxCellNumbersByDim()[1];
        for (size_t i = 0; i < cellVisPoints.size(); i++) {
            int64_t yCoord = maxYNum - cellVisPoints[i].cellDiscreteCoordinates[1];
            int64_t xCoord = cellVisPoints[i].cellDiscreteCoordinates[0];
            data[yCoord][xCoord] = std::max(data[yCoord][xCoord], value[i]);
        }
        return *this;
    }

    MapState & MapState::operator|=(const MapState & value) {
        for (size_t i = 0; i < data.size(); i++) {
            for (size_t j = 0; j < data[i].size(); j++) {
                data[i][j] = std::max(data[i][j], value.data[i][j]);
            }
        }
        return *this;
    }

    MapState & MapState::conv(const conv_matrix & mat, uint16_t floorValue) {
        auto oldData(data);
        uint16_t norm = 0;
        for (int64_t i = 0; i < CONV_SIZE; i++) {
            for (int64_t j = 0; j < CONV_SIZE; j++) {
                norm += mat[i][j];
            }
        }
        for (int64_t i = 0; i < static_cast<int64_t>(data.size()); i++) {
            for (int64_t j = 0; j < static_cast<int64_t>(data[i].size()); j++) {
                uint16_t tmpData = 0;
                for (int64_t ii = (-1 * CONV_SIZE / 2); ii <= CONV_SIZE / 2; ii++) {
                    for (int64_t jj = (-1 * CONV_SIZE / 2); jj <= CONV_SIZE / 2; jj++) {
                        int64_t sum_i = i + ii;
                        int64_t sum_j = j + jj;
                        if (sum_i >= 0 && sum_i < static_cast<int64_t>(data.size()) &&
                            sum_j >= 0 && sum_j < static_cast<int64_t>(data[i].size())) {
                            tmpData += static_cast<uint16_t>(oldData[sum_i][sum_j]) *
                                mat[ii + CONV_SIZE / 2][jj + CONV_SIZE / 2];
                        }
                    }
                }
                if (oldData[i][j] < floorValue) {
                    data[i][j] = tmpData / norm;
                }
                else {
                    data[i][j] = std::max(floorValue, static_cast<uint16_t>(tmpData / norm));
                }
            }
        }
        return *this;
    }

    MapState & MapState::spread(const MapState & bounds, const MapState & barrier, float decayValue, uint8_t floorValue) {
        auto oldData(data);
        for (int64_t i = 0; i < static_cast<int64_t>(data.size()); i++) {
            for (int64_t j = 0; j < static_cast<int64_t>(data[i].size()); j++) {
                if (bounds.data[i][j] == 0) {
                    data[i][j] = 0;
                }
                else if (barrier.data[i][j] > 0) {
                    data[i][j] = 0;
                }
                else {
                    uint8_t tmpData = 0;
                    for (int64_t ii = (-1 * CONV_SIZE / 2); ii <= CONV_SIZE / 2; ii++) {
                        for (int64_t jj = (-1 * CONV_SIZE / 2); jj <= CONV_SIZE / 2; jj++) {
                            int64_t sum_i = i + ii;
                            int64_t sum_j = j + jj;
                            if (sum_i >= 0 && sum_i < static_cast<int64_t>(data.size()) &&
                                sum_j >= 0 && sum_j < static_cast<int64_t>(data[i].size())) {
                                if (ii == 0 && jj == 0) {
                                    tmpData = std::max(tmpData, oldData[sum_i][sum_j]);
                                }
                                else {
                                    tmpData = std::max(tmpData, static_cast<uint8_t>(std::pow(oldData[sum_i][sum_j], decayValue)));
                                }
                            }
                        }
                    }
                    if (tmpData == 0) {
                        data[i][j] = 0;
                    }
                    else if (tmpData < floorValue) {
                        data[i][j] = floorValue;
                    }
                    else {
                        data[i][j] = std::max(floorValue, static_cast<uint8_t>(std::pow(tmpData, decayValue)));
                    }
                }
            }
        }
        return *this;
    }
}