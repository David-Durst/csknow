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
    class MapState {
        array<array<uint8_t, NAV_CELLS_PER_ROW>, NAV_CELLS_PER_ROW> data;
        MapState(const VisPoints & visPoints);
        void saveMapState(const fs::path & path);
    };
}


#endif //CSKNOW_SAVE_MAP_STATE_H
