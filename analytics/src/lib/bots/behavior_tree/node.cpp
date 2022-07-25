//
// Created by durst on 5/3/22.
//

#include "bots/behavior_tree/node.h"

uint32_t Node::getNearestAreaInNextPlace(const ServerState & state, const TreeThinker & treeThinker, string nextPlace) {
    const ServerState::Client & curClient = state.getClient(treeThinker.csgoId);
    const nav_mesh::nav_area & curArea = blackboard.navFile.get_nearest_area_by_position(
            {curClient.lastEyePosX, curClient.lastEyePosY, curClient.lastFootPosZ});

    uint32_t minAreaId = INVALID_ID;
    double minDistance = std::numeric_limits<double>::max();
    for (const AreaId areaId : blackboard.distanceToPlaces.placeToArea[nextPlace]) {
        double newDistance = blackboard.getDistance(curArea.get_id(), areaId);
        if (newDistance < minDistance) {
            minAreaId = areaId;
            minDistance = newDistance;
        }
    }

    if (minAreaId == INVALID_ID) {
        auto z = blackboard.distanceToPlaces.placeToArea[nextPlace];
        int x = 1;
    }

    return minAreaId;
}

uint32_t Node::getRandomAreaInNextPlace(const ServerState & state, string nextPlace) {
    const vector<uint32_t> & nextAreaOptions = blackboard.distanceToPlaces.placeToArea[nextPlace];
    std::uniform_int_distribution<> dist(0, nextAreaOptions.size() - 1);

    return nextAreaOptions[dist(blackboard.gen)];
}
