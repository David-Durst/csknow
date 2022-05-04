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
        PushTaskNode(Blackboard & blackboard) : Node(blackboard) { };
        NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };

    class BaitTaskNode : public Node {
    public:
        BaitTaskNode(Blackboard & blackboard) : Node(blackboard) { };
        NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };
}

class FollowOrderSeqSelectorNode : public FirstSuccessSeqSelectorNode {
    vector<Node> nodes;
    FollowOrderSeqSelectorNode(Blackboard & blackboard) :
            FirstSuccessSeqSelectorNode(blackboard, { follow::PushTaskNode(blackboard),
                                                     follow::BaitTaskNode(blackboard)}) { };
};

#endif //CSKNOW_FOLLOW_ORDER_NODE_H
