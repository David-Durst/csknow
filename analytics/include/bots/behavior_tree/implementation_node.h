//
// Created by durst on 5/8/22.
//

#ifndef CSKNOW_IMPLEMENTATION_NODE_H
#define CSKNOW_IMPLEMENTATION_NODE_H

#include "bots/behavior_tree/node.h"
#define MOVING_THRESHOLD 0.1
#define STUCK_TICKS_THRESHOLD 5

namespace implementation {
    class PathingTaskNode : public Node {
    public:
        PathingTaskNode(Blackboard & blackboard) : Node(blackboard, "PathingTaskNode") { };
        virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };

    class StuckTaskNode : public Node {
    public:
        StuckTaskNode(Blackboard & blackboard) : Node(blackboard, "UnstuckTaskNode") { };
        virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };

    class FireSelectionTaskNode : public Node {
    public:
        FireSelectionTaskNode(Blackboard & blackboard) : Node(blackboard, "FireSelectionTaskNode") { };
        virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };
}

class ImplementationParSelectorNode : public ParSelectorNode {
public:
    ImplementationParSelectorNode(Blackboard & blackboard) :
            ParSelectorNode(blackboard, Node::makeList(
                                            make_unique<implementation::PathingTaskNode>(blackboard),
                                            make_unique<implementation::StuckTaskNode>(blackboard),
                                            make_unique<implementation::FireSelectionTaskNode>(blackboard)),
                            "ImplementationParSelectorNode") { };

    virtual PrintState printState(const ServerState & state, CSGOId playerId) const override {
        PrintState printState = ParSelectorNode::printState(state, playerId);
        printState.curState.push_back(blackboard.playerToPath[playerId].print(state, blackboard.navFile));
        return printState;
    }
};

#endif //CSKNOW_IMPLEMENTATION_NODE_H
