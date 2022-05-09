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

};

#endif //CSKNOW_IMPLEMENTATION_NODE_H
