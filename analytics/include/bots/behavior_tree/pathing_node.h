//
// Created by durst on 6/9/22.
//

#ifndef CSKNOW_PATHING_NODE_H
#define CSKNOW_PATHING_NODE_H

#include "bots/behavior_tree/node.h"
#include "bots/behavior_tree/pathing_data.h"

#define MOVING_THRESHOLD 0.1

namespace movement {
    class PathingNode : public Node {
    public:
        PathingNode(Blackboard & blackboard) : Node(blackboard, "PathingTaskNode") { };
        virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };

    class WaitNode : public Node {
        double waitSeconds;
        map<CSGOId, int32_t> startFrame;
    public:
        WaitNode(Blackboard & blackboard, double waitSeconds) : Node(blackboard, "WaitTaskNode"), waitSeconds(waitSeconds) { };
        virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };
}

#endif //CSKNOW_PATHING_NODE_H
