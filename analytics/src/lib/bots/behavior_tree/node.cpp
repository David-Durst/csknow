//
// Created by durst on 5/3/22.
//

#include "bots/behavior_tree/node.h"

void Blackboard::computeDistanceMatrix() {
    for (const auto &srcNavArea: navFile.m_areas) {
        vector<double> tmpDistances(navFile.m_areas.size());
#pragma parallel for
        for (size_t i = 0; i < navFile.m_areas.size(); i++) {
            const auto &dstNavArea = navFile.m_areas[i];
            std::optional<vector<nav_mesh::vec3_t>> pathOption =
                    navFile.find_path(srcNavArea.get_center(), dstNavArea.get_center());

            if (pathOption) {
                double totalDistance = 0;
                vector<nav_mesh::vec3_t> path = pathOption.value();
                for (size_t i = 0; i < path.size() - 1; i++) {
                    totalDistance += computeDistance(vec3tConv(path[i]), vec3tConv(path[i + 1]));
                }
                tmpDistances[i] = totalDistance;
            } else {
                tmpDistances[i] = std::numeric_limits<double>::max();
            }
        }


        for (size_t i = 0; i < navFile.m_areas.size(); i++) {
            distanceMatrix[srcNavArea.get_id()][navFile.m_areas[i].get_id()] = tmpDistances[i];
        }
    }
}

uint32_t Node::getNearestAreaInNextPlace(const ServerState & state, const TreeThinker & treeThinker, string nextPlace) {
    const ServerState::Client curClient = state.clients[state.csgoIdToCSKnowId[treeThinker.csgoId]];
    const nav_mesh::nav_area & curArea = blackboard.navFile.get_nearest_area_by_position(
            {curClient.lastEyePosX, curClient.lastEyePosY, curClient.lastFootPosZ});

    uint32_t minAreaId = INVALID_ID;
    double minDistance = std::numeric_limits<double>::max();
    for (const uint32_t areaId : blackboard.navPlaceToArea[nextPlace]) {
        double newDistance = blackboard.getDistance(curArea.get_id(), areaId);
        if (newDistance < minDistance) {
            minAreaId = areaId;
            minDistance = newDistance;
        }
    }

    return minAreaId;
}
