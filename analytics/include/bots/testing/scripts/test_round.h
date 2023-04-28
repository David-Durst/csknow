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

public:
    explicit RoundScript(const csknow::plant_states::PlantStatesResult & plantStatesResult, size_t plantStateIndex);

    void initialize(Tree & tree, ServerState & state) override;
};

#endif //CSKNOW_TEST_ROUND_H
