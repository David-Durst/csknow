//
// Created by durst on 6/9/22.
//

#ifndef CSKNOW_PATHING_NODE_H
#define CSKNOW_PATHING_NODE_H

#include "bots/behavior_tree/node.h"
#include "bots/behavior_tree/pathing_data.h"

// max is 18, this is based on that - https://developer.valvesoftware.com/wiki/Dimensions#Ground_Obstacle_Height
#define OBSTACLE_SIZE 10
#define MAX_OBSTACLE_SIZE 18
#define MOVING_THRESHOLD 0.1

namespace movement {
    Path computePath(const ServerState &state, Blackboard & blackboard, nav_mesh::vec3_t preCheckTargetPos,
                     const ServerState::Client & curClient);

    class PathingNode : public Node {
    public:
        PathingNode(Blackboard & blackboard) : Node(blackboard, "PathingTaskNode") { };
        virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };

    class WaitNode : public Node {
        double waitSeconds;
        map<CSGOId, CSKnowTime> startTime;
        bool succeedOnEnd;
    public:
        WaitNode(Blackboard & blackboard, double waitSeconds, bool succeedOnEnd = true) :
            Node(blackboard, "WaitTaskNode"), waitSeconds(waitSeconds), succeedOnEnd(succeedOnEnd) { };
        virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };

    class WaitTicksNode : public Node {
        size_t waitTicks;
        map<CSGOId, size_t> numTicksWaited;
        bool succeedOnEnd;
    public:
        WaitTicksNode(Blackboard & blackboard, size_t waitTicks, bool succeedOnEnd = true) :
                Node(blackboard, "WaitTaskNode"), waitTicks(waitTicks), succeedOnEnd(succeedOnEnd) { };
        virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };
}

#endif //CSKNOW_PATHING_NODE_H
