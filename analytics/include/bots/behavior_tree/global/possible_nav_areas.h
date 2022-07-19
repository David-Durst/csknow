//
// Created by steam on 7/18/22.
//

#ifndef CSKNOW_POSSIBLE_NAV_AREAS_H
#define CSKNOW_POSSIBLE_NAV_AREAS_H
#include "load_save_bot_data.h"
#include "bots/load_save_vis_points.h"
#include "navmesh/nav_file.h"
#include "queries/reachable.h"

class PossibleNavAreas {
    map<CSGOId, AreaBits> possiblyInArea;
    //map<CSGOId, AreaBits> boundary; // possiblyInArea nodes connected directly to not possibly nodes
    map<CSGOId, map<AreaId, CSKnowTime>> entryTime;
    const nav_mesh::nav_file & navFile;

public:
    PossibleNavAreas (const nav_mesh::nav_file & navFile) : navFile(navFile) { }

    void reset(CSGOId playerId) {
        possiblyInArea[playerId].reset();
        //boundary[playerId].set();
    }

    void set(CSGOId playerId, AreaId areaId, bool inArea, CSKnowTime time) {
        size_t index = navFile.m_area_ids_to_indices.find(areaId)->second;
        possiblyInArea[playerId][index] = inArea;
        //boundary[playerId].set();
        entryTime[playerId][areaId] = time;
    }

    void set(CSGOId playerId, AreaBits playerPossiblyInArea, CSKnowTime curTime) {
        possiblyInArea[playerId] = playerPossiblyInArea;
        //boundary[playerId].set();
        for (size_t i = 0; i < playerPossiblyInArea.size(); i++) {
            if (playerPossiblyInArea[i]) {
                entryTime[playerId][navFile.m_areas[i].get_id()] = curTime;
            }
        }
    }

    bool get(CSGOId playerId, AreaId areaId) const {
        size_t index = navFile.m_area_ids_to_indices.find(areaId)->second;
        return possiblyInArea.find(playerId)->second[index];
    }

    void andBits(CSGOId playerId, const AreaBits & mask) {
        // any where that changed or a connection to it needs to be updated
        /*
        AreaBits changedAreas = possiblyInArea[playerId] ^ mask;
        for (size_t i = 0; i < navFile.m_areas.size(); i++) {
            if (changedAreas[i]) {
                boundary[i] = true;
                for (size_t j = 0; j < connectionsAreaLength[i]; j++) {
                    boundary[connections[connectionsAreaStart[i] + j]] = true;
                }
            }
        }
         */
        possiblyInArea[playerId] &= mask;
    }

    vector<AreaId> getPossibleAreas(CSGOId playerId) const {
        vector<AreaId> result;
        for (size_t i = 0; i < possiblyInArea.find(playerId)->second.size(); i++) {
            if (possiblyInArea.find(playerId)->second[i]) {
                result.push_back(navFile.m_areas[i].get_id());
            }
        }
        return result;
    }


    vector<size_t> getEnemiesPossiblePositions(const ServerState & state, CSGOId sourceId) const {
        AreaBits resultBits;
        for (const auto & client : state.clients) {
            if (client.team != state.getClient(sourceId).team && client.isAlive) {
                resultBits |= possiblyInArea.find(client.csgoId)->second;
            }
        }

        vector<size_t> result;
        for (size_t i = 0; i < navFile.m_areas.size(); i++) {
            if (resultBits[i]) {
                result.push_back(i);
            }
        }

        return result;
    }

    AreaBits getEnemiesPositionStatus(const ServerState & state, CSGOId sourceId) const {
        AreaBits resultBits;
        for (const auto & client : state.clients) {
            if (client.team != state.getClient(sourceId).team && client.isAlive) {
                resultBits |= possiblyInArea.find(client.csgoId)->second;
            }
        }
        return resultBits;
    }

    void addNeighbors(const ServerState & state, const ReachableResult & reachability, CSGOId playerId) {
        AreaBits newAreas;
        AreaBits & playerPossiblyInArea = possiblyInArea[playerId];
        //AreaBits & playerBoundary = boundary[playerId];
        map<AreaId, CSKnowTime> & playerEntryTime = entryTime[playerId];
        for (size_t i = 0; i < navFile.m_areas.size(); i++)  {
            //if (playerPossiblyInArea[i] && playerBoundary[i]) {
            if (playerPossiblyInArea[i]) {
                AreaId iAreaId = navFile.m_areas[i].get_id();
                //bool anyConsNotPossible = false;
                for (size_t j = 0; j < navFile.connections_area_length[i]; j++) {
                    size_t conAreaIndex = navFile.connections[navFile.connections_area_start[i] + j];
                    AreaId conAreaId = navFile.m_areas[conAreaIndex].get_id();
                    if (!playerPossiblyInArea[conAreaIndex] &&
                        reachability.getDistance(i, conAreaIndex) / MAX_RUN_SPEED
                        < state.getSecondsBetweenTimes(playerEntryTime[iAreaId], state.loadTime)) {
                        newAreas[conAreaIndex] = true;
                        playerEntryTime[conAreaId] = state.loadTime;
                    }
                    // may keep areas on boundary too long as one of their cons may be covered by a later area.
                    // this will only last until next frame, so not a big deal
                    // I think better than looping through all values twice and having to reload them into cache
                    //anyConsNotPossible |= !playerPossiblyInArea[conAreaIndex] && !newAreas[conAreaIndex];
                }
                /*
                if (!anyConsNotPossible) {
                    playerBoundary[i] = false;
                }
                 */
            }
        }

        playerPossiblyInArea |= newAreas;
    };
};

#endif //CSKNOW_POSSIBLE_NAV_AREAS_H
