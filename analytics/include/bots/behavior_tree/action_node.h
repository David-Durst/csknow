//
// Created by durst on 5/8/22.
//

#ifndef CSKNOW_ACTION_NODE_H
#define CSKNOW_ACTION_NODE_H

#include "bots/behavior_tree/node.h"

namespace action {
    class MovementTaskNode : public Node {
    public:
        MovementTaskNode(Blackboard & blackboard) : Node(blackboard) { };
        NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };

    class AimTaskNode : public Node {
    public:
        AimTaskNode(Blackboard & blackboard) : Node(blackboard) { };
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
    ActionParSelectorNode(Blackboard & blackboard) :
            ParSelectorNode(blackboard, { action::MovementTaskNode(blackboard),
                                          action::AimTaskNode(blackboard),
                                          action::ShootTaskNode(blackboard)}) { };

};

#endif //CSKNOW_ACTION_NODE_H
