//
// Created by durst on 6/9/22.
//

#ifndef CSKNOW_BLACKBOARD_H
#define CSKNOW_BLACKBOARD_H

#include "load_save_bot_data.h"
#include "geometryNavConversions.h"
#include "navmesh/nav_file.h"
#include "bots/behavior_tree/global/order_data.h"
#include "bots/behavior_tree/priority/priority_data.h"
#include "bots/behavior_tree/pathing_data.h"
#include "bots/behavior_tree/action_data.h"
#include "queries/nav_mesh.h"
#include "queries/reachable.h"
#include "bots/load_save_vis_points.h"
#include "bots/testing/script_data.h"
#include "bots/behavior_tree/priority/memory_data.h"
#include "bots/behavior_tree/global/communicate_data.h"
#include <filesystem>
#include <memory>
#include <random>
using std::map;
using std::make_unique;

enum class AggressiveType {
    Push,
    Bait,
    NUM_AGGESSIVE_TYPE
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
    EngagementParams engagementParams;
    double maxMemorySeconds;

    int64_t orderWaypointIndex;
    int64_t orderGrenadeIndex;
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

    string getState() const {
        stringstream ss;
        getStateInner(0, ss);
        return ss.str();
    }

    PrintState() { }
    PrintState(string curState, bool appendNewline = false) : curState({curState}), appendNewline(appendNewline) { }
    PrintState(vector<PrintState> childrenStates, vector<string> curState, bool appendNewline = false) :
        childrenStates(childrenStates), curState(curState), appendNewline(appendNewline) { }
};

struct Blackboard {
    nav_mesh::nav_file navFile;
    ServerState lastFrameState;

    // helpers
    std::random_device rd;
    std::mt19937 gen;

    // general map data
    ReachableResult reachability;
    VisPoints visPoints;
    map<string, vector<uint32_t>> navPlaceToArea;

    // all player data
    map<CSGOId, TreeThinker> playerToTreeThinkers;

    // order data
    bool newOrderThisFrame;
    vector<Order> orders;
    map<CSGOId, int64_t> playerToOrder;
    map<CSGOId, int32_t> playerToPushOrder;
    map<CSGOId, AreaId> playerToDangerAreaId;

    // knowledge data
    double tMemorySeconds = 1.0, ctMemorySeconds = 1.0;
    EnemyPositionsMemory tMemory, ctMemory;
    const EnemyPositionsMemory & getCommunicatedPlayers(const ServerState & state, TreeThinker & treeThinker) {
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

    // priority data
    map<CSGOId, Priority> playerToPriority;
    map<CSGOId, Path> playerToPath;
    map<CSGOId, uint32_t> playerToLastPathingSourceNavAreaId;
    map<CSGOId, uint32_t> playerToLastPathingTargetNavAreaId;

    // action data
    map<CSGOId, Action> playerToAction;
    map<CSGOId, Action> lastPlayerToAction;
    map<CSGOId, PIDState> playerToPIDStateX, playerToPIDStateY;
    std::uniform_real_distribution<> aimDis;

    string getPlayerPlace(Vec3 pos) {
        return navFile.get_place(navFile.get_nearest_area_by_position(vec3Conv(pos)).m_place);
    }

    double getDistance(uint32_t srcArea, uint32_t dstArea) {
        return computeDistance(vec3tConv(navFile.get_area_by_id_fast(srcArea).get_center()),
                               vec3tConv(navFile.get_area_by_id_fast(dstArea).get_center()));
    }

    PrintState printOrderState(const ServerState & state);
    PrintState printCommunicateState(const ServerState & state);
    vector<PrintState> printPerPlayerState(const ServerState & state, CSGOId playerId);

    Blackboard(string navPath, string mapName) :
        navFile(navPath.c_str()), gen(rd()), aimDis(0., 2.0), visPoints(navFile) {
        string mapsPath = std::filesystem::path(navPath).remove_filename().string();
        reachability.load(mapsPath, mapName);
        visPoints.load(mapsPath, mapName);

        for (const auto & area : navFile.m_areas) {
            navPlaceToArea[navFile.get_place(area.m_place)].push_back(area.get_id());
        }

        tMemory.considerAllTeammates = true;
        tMemory.team = ENGINE_TEAM_T;
        ctMemory.considerAllTeammates = true;
        ctMemory.team = ENGINE_TEAM_CT;
    }

};

#endif //CSKNOW_BLACKBOARD_H
