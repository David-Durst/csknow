//
// Created by steam on 7/5/22.
//

#ifndef CSKNOW_GLOBAL_NODE_H
#define CSKNOW_GLOBAL_NODE_H

#include "bots/behavior_tree/global/order_node.h"
#include "bots/behavior_tree/global/communicate_node.h"

class GlobalNode : public SequenceNode {
public:
    GlobalNode(Blackboard & blackboard) :
            SequenceNode(blackboard, Node::makeList(
                    make_unique<OrderNode>(blackboard),
                    make_unique<CommunicateNode>(blackboard)
            ), "GlobalNode") { };
};

#endif //CSKNOW_GLOBAL_NODE_H
