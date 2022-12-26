//
// Created by durst on 6/9/22.
//

#ifndef CSKNOW_BLACKBOARD_H
#define CSKNOW_BLACKBOARD_H

#include "bots/analysis/save_nav_overlay.h"
#include "bots/load_save_bot_data.h"
#include "geometryNavConversions.h"
#include "navmesh/nav_file.h"
#include "bots/behavior_tree/global/strategy_data.h"
#include "bots/behavior_tree/priority/priority_data.h"
#include "bots/behavior_tree/pathing_data.h"
#include "bots/behavior_tree/action/action_data.h"
#include "queries/nav_mesh.h"
#include "queries/reachable.h"
#include "queries/distance_to_places.h"
#include "bots/analysis/load_save_vis_points.h"
#include "bots/behavior_tree/priority/memory_data.h"
#include "bots/behavior_tree/global/possible_nav_areas.h"
#include "bots/behavior_tree/action/second_order_controller.h"
#include "bots/analysis/streaming_manager.h"
#include <filesystem>
#include <memory>
#include <random>
using std::map;
using std::make_unique;

enum class AggressiveType {
    Push,
    Bait,
    NUM_AGGESSIVE_TYPE [[maybe_unused]]
};

struct EngagementParams {
    double standDistance;
    double moveDistance;
    double burstDistance;
    double sprayDistance;
};

struct TreeThinker {
    // constant values across game
    CSGOId csgoId;
    AggressiveType aggressiveType;
    EngagementParams engagementParams = {INVALID_ID, INVALID_ID, INVALID_ID, INVALID_ID};
    double maxMemorySeconds = INVALID_ID;
};

struct PrintState {
    vector<PrintState> childrenStates;
    vector<string> curState;
    bool appendNewline = false;

    void getStateInner(size_t depth, stringstream & ss) const {
        for (const auto & curStateLine : curState) {
            for (size_t i = 0; i < depth; i++) {
                ss << "  ";
            }
            ss << curStateLine;
            ss << std::endl;
        }
        for (const auto & childState : childrenStates) {
            childState.getStateInner(depth + 1, ss);
        }
    }

    [[nodiscard]]
    string getState() const {
        stringstream ss;
        getStateInner(0, ss);
        return ss.str();
    }

    PrintState() = default;
    explicit PrintState(const string & curState, bool appendNewline = false) :
        curState({curState}), appendNewline(appendNewline) { }
    PrintState(const vector<PrintState> & childrenStates, const vector<string> & curState, bool appendNewline = false) :
        childrenStates(childrenStates), curState(curState), appendNewline(appendNewline) { }
};

struct Blackboard {
    string navPath, mapsPath;
    nav_mesh::nav_file navFile;
    set<AreaId> removedAreas;
    map<AreaId, AreaId> removedAreaAlternatives;
    ServerState lastFrameState;
    StreamingManager streamingManager;

    // helpers
    std::random_device rd;
    std::mt19937 gen;
    NavFileOverlay navFileOverlay;

    // general map data
    VisPoints visPoints;
    MapMeshResult mapMeshResult;
    ReachableResult reachability;
    DistanceToPlacesResult distanceToPlaces;

    // all player data
    map<CSGOId, TreeThinker> playerToTreeThinkers;
    [[nodiscard]]
    const nav_mesh::nav_area & getPlayerNavArea(const ServerState::Client & client) const {
        return navFile.get_nearest_area_by_position(vec3Conv(client.getFootPosForPlayer()));
    }
    [[nodiscard]]
    const nav_mesh::nav_area & getC4NavArea(const ServerState & state) const {
        return navFile.get_nearest_area_by_position(vec3Conv(state.getC4Pos()));
    }
    std::uniform_real_distribution<> aggressionDis;

    // order data (movedC4 is for debugging, need to reset orders)
    bool newOrderThisFrame, recomputeOrders = false;
    Strategy strategy;

    bool executeIfAllFinishedSetup(const ServerState & state) {
        for (const auto & [playerId, treeThinker] : playerToTreeThinkers) {
            const auto & curClient = state.getClient(playerId);
            if (curClient.team != ENGINE_TEAM_CT || !curClient.isBot || !curClient.isAlive) {
                continue;
            }
            if (!strategy.playerFinishedSetup(playerId)) {
                return false;
            }
        }
        for (const auto & [playerId, treeThinker] : playerToTreeThinkers) {
            const auto & curClient = state.getClient(playerId);
            if (curClient.team != ENGINE_TEAM_CT || !curClient.isBot || !curClient.isAlive) {
                continue;
            }
            strategy.playerExecuting(playerId);
        }
        return true;
    }
    /*
    vector<Order> orders;
    map<CSGOId, int64_t> playerToOrder;
     */

    // prediction data
    map<CSGOId, AreaId> playerToDangerAreaId;
    vector<CSKnowTime> tDangerAreaLastCheckTime, ctDangerAreaLastCheckTime;
    vector<CSKnowTime> & getDangerAreaLastCheckTime(const ServerState::Client & client) {
        if (client.team == ENGINE_TEAM_T) {
            return tDangerAreaLastCheckTime;
        }
        else {
            return ctDangerAreaLastCheckTime;
        }
    }
    map<CSGOId, CSKnowTime> lastDangerAssignment;

    // knowledge data
    double tMemorySeconds = 1.0, ctMemorySeconds = 1.0;
    EnemyPositionsMemory tMemory, ctMemory;
    map<CSGOId, map<CSGOId, EnemyPositionMemory>> playerToRelevantCommunicatedEnemies;
    [[nodiscard]]
    const EnemyPositionsMemory & getCommunicatedPlayers(const ServerState & state, TreeThinker & treeThinker) const {
        if (state.getClient(treeThinker.csgoId).team == ENGINE_TEAM_T) {
            return tMemory;
        }
        else {
            return ctMemory;
        }
    }
    map<CSGOId, EnemyPositionsMemory> playerToMemory;
    PossibleNavAreas possibleNavAreas;
    bool resetPossibleNavAreas = false;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
    [[maybe_unused]] bool inTest = false; // inTest just for debugging, setting break points once test setup
#pragma GCC diagnostic pop
    map<TeamId, RoundNumber> teamToLastRoundSawEnemy;
    bool sawEnemyThisRound(const ServerState & state, TeamId team) {
        return teamToLastRoundSawEnemy.find(team) != teamToLastRoundSawEnemy.end() &&
                teamToLastRoundSawEnemy[team] == state.roundNumber;
    }
    optional<CSGOId> defuserId;
    [[nodiscard]]
    bool isPlayerDefuser(CSGOId playerId) const {
        return defuserId && defuserId.value() == playerId;
    }

    // priority data
    map<CSGOId, Priority> playerToPriority;
    map<CSGOId, Path> playerToPath;
    map<CSGOId, uint32_t> playerToLastPathingSourceNavAreaId;
    map<CSGOId, uint32_t> playerToLastPathingTargetNavAreaId;
    std::uniform_real_distribution<> standDis;

    // action data
    map<CSGOId, SecondOrderController> playerToMouseController;
    map<CSGOId, Action> playerToAction;
    map<CSGOId, Action> lastPlayerToAction;
    map<CSGOId, PIDState> playerToPIDStateX, playerToPIDStateY;
    std::uniform_real_distribution<> aimDis;

    [[nodiscard]]
    string getPlayerPlace(Vec3 pos) const {
        return navFile.get_place(navFile.get_nearest_area_by_position(vec3Conv(pos)).m_place);
    }

    [[nodiscard]] [[maybe_unused]]
    double getDistance(AreaId srcArea, AreaId dstArea) const {
        return computeDistance(vec3tConv(navFile.get_area_by_id_fast(srcArea).get_center()),
                               vec3tConv(navFile.get_area_by_id_fast(dstArea).get_center()));
    }

    [[nodiscard]]
    AreaBits getVisibleAreasByTeam(const ServerState & state, int32_t team) const {
        AreaBits result;
        for (const auto & client : state.clients) {
            if (client.isAlive) {
                AreaId curArea =
                        navFile.get_nearest_area_by_position(vec3Conv(client.getFootPosForPlayer())).get_id();
                if (client.team == team) {
                    result |= visPoints.getVisibilityRelativeToSrc(curArea);
                }
            }
        }
        return result;
    }

    PrintState printStrategyState(const ServerState & state);
    PrintState printCommunicateState(const ServerState & state);
    vector<PrintState> printPerPlayerState(const ServerState & state, CSGOId playerId);

    Blackboard(const string & navPath, const string & mapName) :
        navPath(navPath), mapsPath(std::filesystem::path(navPath).remove_filename().string()),
        navFile(navPath.c_str()),
        gen(rd()), navFileOverlay(navFile),
        visPoints(navFile), mapMeshResult(queryMapMesh(navFile, "")),
        reachability(queryReachable(visPoints, mapMeshResult, "", mapsPath, mapName)),
        distanceToPlaces(queryDistanceToPlaces(navFile, reachability, "", mapsPath, mapName)),
        aggressionDis(0., 1.),
        tDangerAreaLastCheckTime(navFile.m_areas.size(), defaultTime),
        ctDangerAreaLastCheckTime(navFile.m_areas.size(), defaultTime),
        possibleNavAreas(navFile), standDis(0, 100.0), aimDis(0., 2.0) {

        navFileOverlay.setMapsPath(mapsPath);
        visPoints.load(mapsPath, mapName, true, navFile, true);
        visPoints.load(mapsPath, mapName, false, navFile, true);

        tMemory.considerAllTeammates = true;
        tMemory.team = ENGINE_TEAM_T;
        ctMemory.considerAllTeammates = true;
        ctMemory.team = ENGINE_TEAM_CT;
    }

};

#endif //CSKNOW_BLACKBOARD_H
