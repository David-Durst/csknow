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
public:
    ActionParSelectorNode(Blackboard & blackboard) :
            ParSelectorNode(blackboard, { make_unique<action::MovementTaskNode>(blackboard),
                                          make_unique<action::AimTaskNode>(blackboard),
                                          make_unique<action::FireTaskNode>(blackboard) },
                            "ActionParSelectorNode") { };

    PrintState printState(const ServerState & state, CSGOId playerId) const override {
        PrintState printState = ParSelectorNode::printState(state, playerId);
        printState.curState = {blackboard.playerToAction[playerId].print()};
        return printState;
    }
};

#endif //CSKNOW_ACTION_NODE_H
