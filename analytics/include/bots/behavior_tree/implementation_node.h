//
// Created by durst on 5/8/22.
//

#ifndef CSKNOW_IMPLEMENTATION_NODE_H
#define CSKNOW_IMPLEMENTATION_NODE_H

#include "bots/behavior_tree/node.h"

namespace implementation {
    class PathingTaskNode : public Node {
    public:
        PathingTaskNode(Blackboard & blackboard) : Node(blackboard, "PathingTaskNode") { };
        NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };

    class FireSelectionTaskNode : public Node {
    public:
        FireSelectionTaskNode(Blackboard & blackboard) : Node(blackboard, "FireSelectionTaskNode") { };
        NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };
}

class ImplementationParSelectorNode : public ParSelectorNode {
    vector<Node> nodes;
public:
    ImplementationParSelectorNode(Blackboard & blackboard) :
            ParSelectorNode(blackboard, { implementation::PathingTaskNode(blackboard),
                                          implementation::FireSelectionTaskNode(blackboard)},
                            "ImplementationParSelectorNode") { };

    PrintState printState(const ServerState & state, CSGOId playerId) const override {
        PrintState printState = ParSelectorNode::printState(state, playerId);
        printState.curState = {blackboard.playerToPath[playerId].print(state, blackboard.navFile)};
        return printState;
    }
};

#endif //CSKNOW_IMPLEMENTATION_NODE_H
