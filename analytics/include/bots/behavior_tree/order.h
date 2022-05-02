//
// Created by durst on 5/1/22.
//

#ifndef CSKNOW_ORDER_H
#define CSKNOW_ORDER_H

#include "bots/behavior_tree/nodes.h"
#include <map>

struct GrenadeThrow {
    CSGOId thrower;
    Vec3 origin;
    Vec2 angle;
    Vec3 target;
};

struct Order {
    std::vector<Vec3> waypoints;
    std::vector<GrenadeThrow> grenadeThrows;
    int16_t numTeammates;
};

class D2OrderTaskNode : Node {
    D2OrderTaskNode(Blackboard & blackboard) : Node(blackboard) { };

    bool relevant(const ServerState &state, const TreeThinker & treeThinker) override;
    void exec(const ServerState & state, const TreeThinker & treeThinker) override;
};

class GeneralOrderTaskNode : Node {
    GeneralOrderTaskNode(Blackboard & blackboard) : Node(blackboard) { };

    bool relevant(const ServerState &state, const TreeThinker & treeThinker) override;
    void exec(const ServerState & state, const TreeThinker & treeThinker) override;
};

#endif //CSKNOW_ORDER_H
