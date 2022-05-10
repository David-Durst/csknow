//
// Created by durst on 5/3/22.
//

#ifndef CSKNOW_FOLLOW_ORDER_NODE_H
#define CSKNOW_FOLLOW_ORDER_NODE_H

#include "bots/behavior_tree/node.h"
#include <map>

namespace follow {
    class PushTaskNode : public Node {
    public:
        PushTaskNode(Blackboard & blackboard) : Node(blackboard, "PushTaskNode") { };
        NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };

    class BaitTaskNode : public Node {
    public:
        BaitTaskNode(Blackboard & blackboard) : Node(blackboard, "BaitTaskNode") { };
        NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };
}

class FollowOrderSeqSelectorNode : public FirstNonFailSeqSelectorNode {
public:
    FollowOrderSeqSelectorNode(Blackboard & blackboard) :
            FirstNonFailSeqSelectorNode(blackboard, {make_unique<follow::PushTaskNode>(blackboard),
                                                     make_unique<follow::BaitTaskNode>(blackboard)},
                                        "FollowOrderSeqSelectorNode") { };

    NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        int childIndex;
        if (treeThinker.aggressiveType == AggressiveType::Push) {
            childIndex = 0;
        }
        else {
            childIndex = 1;
        }
        playerNodeState[treeThinker.csgoId] = children[childIndex].exec(state, treeThinker);
        return playerNodeState[treeThinker.csgoId];
    }
};

#endif //CSKNOW_FOLLOW_ORDER_NODE_H
