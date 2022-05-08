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

    class ShootTaskNode : public Node {
    public:
        ShootTaskNode(Blackboard & blackboard) : Node(blackboard) { };
        NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };
}

class EngageSeqSelectorNode : public ParSelectorNode {
    vector<Node> nodes;
    EngageSeqSelectorNode(Blackboard & blackboard) :
            ParSelectorNode(blackboard, { engage::TargetSelectionTaskNode(blackboard),
                                                      engage::ShootTaskNode(blackboard)}) { };

};

#endif //CSKNOW_ENGAGE_NODE_H
