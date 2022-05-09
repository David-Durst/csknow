//
// Created by durst on 5/9/22.
//

#ifndef CSKNOW_PRIORITY_PAR_NODE_H
#define CSKNOW_PRIORITY_PAR_NODE_H

#include "bots/behavior_tree/node.h"
#include "bots/behavior_tree/priority/follow_order_node.h"
#include "bots/behavior_tree/priority/target_selection_node.h"

class PriorityParNode : public ParSelectorNode {
    vector<Node> nodes;
public:
    PriorityParNode(Blackboard & blackboard) :
            ParSelectorNode(blackboard, { FollowOrderSeqSelectorNode(blackboard),
                                          TargetSelectionTaskNode(blackboard)},
                            "PriorityParNode") { };

};

#endif //CSKNOW_PRIORITY_PAR_NODE_H
