//
// Created by durst on 4/23/23.
//

#ifndef CSKNOW_MODEL_NAV_DATA_H
#define CSKNOW_MODEL_NAV_DATA_H

#include <queries/query.h>
#include "bots/load_save_bot_data.h"

struct ModelNavData {
    vector<string> orderPlaceOptions;
    vector<float> orderPlaceProbs;
    string curPlace;
    string nextPlace;
    PlaceIndex nextPlaceIndex;
    size_t nextArea;

    [[nodiscard]]
    string print(const ServerState &) const {
        stringstream result;

        result << "cur place " << curPlace << ", next place " << nextPlace << ", next area " << nextArea
            << "order places: ";
        for (const auto & orderPlace : orderPlaceOptions) {
            result << orderPlace << ", ";
        }
        result << "; place prob: ";
        for (const auto & orderProb : orderPlaceProbs) {
            result << orderProb << ", ";
        }

        return result.str();
    }
};

#endif //CSKNOW_MODEL_NAV_DATA_H
