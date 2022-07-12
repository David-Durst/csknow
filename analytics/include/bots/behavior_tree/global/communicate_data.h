//
// Created by durst on 7/10/22.
//

#ifndef CSKNOW_COMMUNICATE_DATA_H
#define CSKNOW_COMMUNICATE_DATA_H

#include "load_save_bot_data.h"
#include "bots/load_save_vis_points.h"
#include "navmesh/nav_file.h"

typedef map<CSKnowId, map<AreaId, CSKnowTime>> PossibleNavAreas;
static set<AreaId> getEnemiesPossiblePositions(const ServerState & state, CSGOId sourceId, PossibleNavAreas possibleNavAreas) {
    set<AreaId> result;
    for (const auto & client : state.clients) {
        if (client.team != state.getClient(sourceId).team) {
            for (const auto & [possibleAreaId, _] : possibleNavAreas[client.csgoId]) {
                result.insert(possibleAreaId);
            }
        }
    }
    return result;
}

#endif //CSKNOW_COMMUNICATE_DATA_H
