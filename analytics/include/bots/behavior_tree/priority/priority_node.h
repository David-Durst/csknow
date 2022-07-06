//
// Created by durst on 5/9/22.
//

#ifndef CSKNOW_PRIORITY_NODE_H
#define CSKNOW_PRIORITY_NODE_H

#include "bots/behavior_tree/node.h"
#include "bots/behavior_tree/priority/follow_order_node.h"
#include "bots/behavior_tree/priority/engage_node.h"
#include "bots/behavior_tree/priority/memory_node.h"

class PriorityDecisionNode : public SelectorNode {
public:
    PriorityDecisionNode(Blackboard & blackboard) :
            SelectorNode(blackboard, Node::makeList(
                                                make_unique<EnemyEngageCheckNode>(blackboard),
                                                make_unique<NoEnemyOrderCheckNode>(blackboard)),
                            "PriorityDecisionNode") { };
};

class PriorityNode : public ParallelFirstNode {
public:
    PriorityNode(Blackboard & blackboard) :
            ParallelFirstNode(blackboard, Node::makeList(
                                 make_unique<memory::PerPlayerMemory>(blackboard),
                                 make_unique<PriorityDecisionNode>(blackboard)),
                         "PrioritySetupAndDecideParallelNode") { };
};

#endif //CSKNOW_PRIORITY_NODE_H
