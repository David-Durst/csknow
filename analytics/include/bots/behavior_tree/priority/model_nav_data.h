//
// Created by durst on 4/23/23.
//

#ifndef CSKNOW_MODEL_NAV_DATA_H
#define CSKNOW_MODEL_NAV_DATA_H

#include <queries/query.h>
#include "bots/load_save_bot_data.h"
#include "geometryNavConversions.h"

struct ModelNavData {
    bool deltaPosMode;
    vector<float> deltaPosProbs;
    int64_t radialVelIndex;
    int deltaZVal;
    float deltaXVal, deltaYVal;
    std::optional<AreaId> disabledArea;
    // this is a target pos that doesn't need to be modified to match the world
    vector<string> orderPlaceOptions;
    vector<float> orderPlaceProbs;
    string curPlace;
    string nextPlace;
    PlaceIndex nextPlaceIndex;
    size_t nextArea;

    [[nodiscard]]
    string print(const ServerState &) const ;
};

#endif //CSKNOW_MODEL_NAV_DATA_H
