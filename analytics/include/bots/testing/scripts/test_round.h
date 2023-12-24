//
// Created by durst on 4/27/23.
//

#ifndef CSKNOW_TEST_ROUND_H
#define CSKNOW_TEST_ROUND_H

#include "bots/testing/script.h"
#include "bots/behavior_tree/pathing_node.h"
#include "bots/behavior_tree/tree.h"
#include "bots/testing/blackboard_management.h"
#include "bots/testing/state_checks.h"
#include "queries/moments/plant_states.h"

constexpr int maxCT = 4;
constexpr int maxT = 3;

class RoundScript : public Script {
    Vec3 c4Pos;
    vector<Vec3> playerPos;
    vector<Vec2> playerViewAngle;
    vector<int> health;
    vector<int> armor;
    size_t plantStateIndex, numRounds;
    std::optional<vector<bool>> playerFreeze;

public:
    explicit RoundScript(const csknow::plant_states::PlantStatesResult & plantStatesResult, size_t plantStateIndex,
                         size_t numRounds, std::mt19937 gen, std::uniform_real_distribution<> dis,
                         std::optional<vector<bool>> playerFreeze, string baseName, std::optional<Vec3> cameraOrigin,
                         std::optional<Vec2> cameraAngle);

    void initialize(Tree & tree, ServerState & state) override;
};

class WaitUntilScoreScript : public Script {
public:
    WaitUntilScoreScript();
    void initialize(Tree & tree, ServerState & state) override;
};

vector<Script::Ptr> createRoundScripts(const csknow::plant_states::PlantStatesResult & plantStatesResult,
                                       int startSituationId, bool quitAtEnd);
vector<Script::Ptr> createPrebakedRoundScripts(const nav_mesh::nav_file & navFile, bool shouldRandomizePositions,
                                               int situationId, bool quitAtEnd);

#endif //CSKNOW_TEST_ROUND_H
