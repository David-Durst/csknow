//
// Created by durst on 5/8/22.
//

#ifndef CSKNOW_ACTION_NODE_H
#define CSKNOW_ACTION_NODE_H

#include "bots/behavior_tree/node.h"

namespace action {
    class MovementTaskNode : public Node {
    public:
        MovementTaskNode(Blackboard & blackboard) : Node(blackboard, "MovementTaskNode") { };
        NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };

    class AimTaskNode : public Node {
    public:
        AimTaskNode(Blackboard & blackboard) : Node(blackboard, "AimTaskNode") { };
        NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };

    class FireTaskNode : public Node {
    public:
        FireTaskNode(Blackboard & blackboard) : Node(blackboard, "FireTaskNode") { };
        NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };
}

class ActionParSelectorNode : public ParSelectorNode {
    vector<Node> nodes;
public:
    ActionParSelectorNode(Blackboard & blackboard) :
            ParSelectorNode(blackboard, { action::MovementTaskNode(blackboard),
                                          action::AimTaskNode(blackboard),
                                          action::FireTaskNode(blackboard)},
                            "ActionParSelectorNode") { };

};

#endif //CSKNOW_ACTION_NODE_H
