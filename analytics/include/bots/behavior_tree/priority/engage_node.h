//
// Created by durst on 5/5/22.
//

#ifndef CSKNOW_ENGAGE_NODE_H
#define CSKNOW_ENGAGE_NODE_H

#include "bots/behavior_tree/node.h"

namespace engage {
    class TargetSelectionTaskNode : public Node {
    public:
        TargetSelectionTaskNode(Blackboard & blackboard) : Node(blackboard) { };
        NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };

    class FireSelectionTaskNode : public Node {
    public:
        FireSelectionTaskNode(Blackboard & blackboard) : Node(blackboard) { };
        NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };
}

class EngageParSelectorNode : public ParSelectorNode {
    vector<Node> nodes;
    EngageParSelectorNode(Blackboard & blackboard) :
            ParSelectorNode(blackboard, { engage::TargetSelectionTaskNode(blackboard),
                                                      engage::FireSelectionTaskNode(blackboard)}) { };

};

#endif //CSKNOW_ENGAGE_NODE_H
