//
// Created by durst on 7/10/22.
//

#ifndef CSKNOW_COMMUNICATE_DATA_H
#define CSKNOW_COMMUNICATE_DATA_H
#define MAX_NAV_AREAS 2000
#include "load_save_bot_data.h"
#include "bots/load_save_vis_points.h"
#include "navmesh/nav_file.h"
#include <bitset>
using std::bitset;

struct PossibleNavAreas {
    map<CSGOId, vector<AreaId>> initialAreas;
    map<CSGOId, bitset<MAX_NAV_AREAS>> possiblyInArea;
    map<CSGOId, map<AreaId, CSKnowTime>> entryTime;
    const nav_mesh::nav_file & navFile;

    PossibleNavAreas (const nav_mesh::nav_file & navFile) : navFile(navFile) { }

    void reset(CSGOId playerId) {
        possiblyInArea[playerId].reset();
    }

    void set(CSGOId playerId, AreaId areaId, bool inArea, CSKnowTime time) {
        size_t index = navFile.m_area_ids_to_indices.find(areaId)->second;
        possiblyInArea[playerId][index] = inArea;
        entryTime[playerId][areaId] = time;
    }

    vector<AreaId> getEnemiesPossiblePositions(const ServerState & state, CSGOId sourceId) {
        bitset<MAX_NAV_AREAS> resultBits;
        for (const auto & client : state.clients) {
            if (client.team != state.getClient(sourceId).team && client.isAlive) {
                resultBits |= possiblyInArea[client.csgoId];
            }
        }

        vector<AreaId> result;
        for (size_t i = 0; i < navFile.m_areas.size(); i++) {
            if (resultBits[i]) {
                result.push_back(navFile.m_areas[i].get_id());
            }
        }

        return result;
    }

   void addNeighbors(const ServerState & state, ReachableResult reachability, CSGOId playerId) {
        bitset<MAX_NAV_AREAS> resultBits;
        double distances[navFile.m_areas.size()];
        const auto & playerPossiblyInArea = possiblyInArea[playerId];
        for (size_t i = 0; i < navFile.m_areas.size(); i++)  {
            for (const auto & connection : navFile.m_areas[i].get_connections()) {
                size_t areaIndex = navFile.m_area_ids_to_indices.find(connection.id)->second;
                if (!resultBits[areaIndex] ||
                    distances[areaIndex] / MAX_RUN_SPEED < state.getSecondsBetweenTimes(, state.loadTime)) {

                }
                resultBits[areaIndex] = true;
                distances[] = true;
            }
        }

        vector<AreaId> result;

    };
};
/*
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
 */

#endif //CSKNOW_COMMUNICATE_DATA_H
