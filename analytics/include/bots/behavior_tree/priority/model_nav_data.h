//
// Created by durst on 4/23/23.
//

#ifndef CSKNOW_MODEL_NAV_DATA_H
#define CSKNOW_MODEL_NAV_DATA_H

#include <queries/query.h>
#include "bots/load_save_bot_data.h"

struct ModelNavData {
    vector<string> orderPlaceOptions;
    string curPlace;
    string nextPlace;
    size_t nextArea;

    [[nodiscard]]
    string print(const ServerState &) const {
        stringstream result;

        result << "cur place " << curPlace << ", next place " << nextPlace << ", next area " << nextArea
            << "order places: ";
        for (const auto & orderPlace : orderPlaceOptions) {
            result << orderPlace << ", ";
        }

        return result.str();
    }
};

#endif //CSKNOW_MODEL_NAV_DATA_H
