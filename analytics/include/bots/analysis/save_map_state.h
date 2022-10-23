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
    typedef array<array<uint8_t, CONV_SIZE>, CONV_SIZE> conv_matrix;
    class MapState {
        array<array<uint8_t, NAV_CELLS_PER_ROW>, NAV_CELLS_PER_ROW> data;
        const VisPoints & visPoints;

    public:
        MapState(const VisPoints & visPoints) : data{}, visPoints(visPoints) {};
        void saveNewMapState(const CellBits & value, const fs::path & path);
        void saveMapState(const fs::path & path);
        MapState & operator=(const CellBits & value);
        [[maybe_unused]] MapState & operator+=(const MapState & value);
        [[maybe_unused]] MapState & operator+=(const uint8_t & value);
        [[maybe_unused]] MapState & operator-=(const uint8_t & value);
        [[maybe_unused]] MapState & operator*=(const uint8_t & value);
        [[maybe_unused]] MapState & operator/=(const uint8_t & value);
        [[maybe_unused]] MapState & conv(const conv_matrix & mat);
    };
}


#endif //CSKNOW_SAVE_MAP_STATE_H
