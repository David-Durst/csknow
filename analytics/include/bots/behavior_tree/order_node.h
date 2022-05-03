//
// Created by durst on 5/1/22.
//

#ifndef CSKNOW_ORDER_NODE_H
#define CSKNOW_ORDER_NODE_H

#include "bots/behavior_tree/node.h"
#include <map>

class D2OrderTaskNode : public Node {
public:
    D2OrderTaskNode(Blackboard & blackboard) : Node(blackboard) { };
    NodeState exec(const ServerState & state, const TreeThinker & treeThinker) override;
};

class GeneralOrderTaskNode : public Node {
public:
    GeneralOrderTaskNode(Blackboard & blackboard) : Node(blackboard) { };
    NodeState exec(const ServerState & state, const TreeThinker & treeThinker) override;
};

class OrderSeqSelectorNode : public FirstSuccessSeqSelectorNode {
    vector<Node> nodes;
    OrderSeqSelectorNode(Blackboard & blackboard) :
        FirstSuccessSeqSelectorNode(blackboard, {D2OrderTaskNode(blackboard), GeneralOrderTaskNode(blackboard)}) { };
};

#endif //CSKNOW_ORDER_NODE_H
