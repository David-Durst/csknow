//
// Created by durst on 5/1/22.
//

#ifndef CSKNOW_ORDER_NODE_H
#define CSKNOW_ORDER_NODE_H

#include "bots/behavior_tree/node.h"
#include <map>

namespace order {
    class D2TaskNode : public Node {
    public:
        D2TaskNode(Blackboard & blackboard) : Node(blackboard) { };
        NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };

    class GeneralTaskNode : public Node {
    public:
        GeneralTaskNode(Blackboard & blackboard) : Node(blackboard) { };
        NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };
}

class OrderSeqSelectorNode : public FirstNonFailSeqSelectorNode {
    vector<Node> nodes;
public:
    OrderSeqSelectorNode(Blackboard & blackboard) :
            FirstNonFailSeqSelectorNode(blackboard, {order::D2TaskNode(blackboard), order::GeneralTaskNode(blackboard)}) { };
};

#endif //CSKNOW_ORDER_NODE_H
