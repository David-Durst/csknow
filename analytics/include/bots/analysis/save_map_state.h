//
// Created by durst on 10/19/22.
//

#ifndef CSKNOW_SAVE_MAP_STATE_H
#define CSKNOW_SAVE_MAP_STATE_H

#include "bots/analysis/load_save_vis_points.h"
#include <filesystem>

namespace fs = std::filesystem;

namespace csknow {
    constexpr int CONV_SIZE = 3;
    typedef array<array<uint16_t, CONV_SIZE>, CONV_SIZE> conv_matrix;
    constexpr conv_matrix GAUSSIAN_BLUR_MATRIX = {{{1, 2, 1}, {2, 4, 2}, {1, 2, 1}}};
    class MapState {
        array<array<uint8_t, NAV_CELLS_PER_ROW>, NAV_CELLS_PER_ROW> data;
        const VisPoints & visPoints;

    public:
        MapState(const VisPoints & visPoints) : data{}, visPoints(visPoints) {};
        void saveNewMapState(const CellBits & value, const fs::path & path);
        void saveNewMapState(const vector<uint8_t> & value, const fs::path & path);
        void saveMapState(const fs::path & path);
        MapState & operator=(const CellBits & value);
        MapState & operator=(const vector<uint8_t> & value);
        [[maybe_unused]] MapState & operator|=(const MapState & value);
        [[maybe_unused]] MapState & operator+=(const MapState & value);
        [[maybe_unused]] MapState & operator+=(const uint8_t & value);
        [[maybe_unused]] MapState & operator-=(const uint8_t & value);
        [[maybe_unused]] MapState & operator*=(const uint8_t & value);
        [[maybe_unused]] MapState & operator/=(const uint8_t & value);
        [[maybe_unused]] MapState & conv(const conv_matrix & mat);
    };
}


#endif //CSKNOW_SAVE_MAP_STATE_H
