//
// Created by durst on 5/9/22.
//

#ifndef CSKNOW_PRIORITY_PAR_NODE_H
#define CSKNOW_PRIORITY_PAR_NODE_H

#include "bots/behavior_tree/node.h"
#include "bots/behavior_tree/priority/follow_order_node.h"
#include "bots/behavior_tree/priority/target_selection_node.h"

class PriorityParNode : public ParSelectorNode {
public:
    PriorityParNode(Blackboard & blackboard) :
            ParSelectorNode(blackboard, { make_unique<FollowOrderSeqSelectorNode>(blackboard),
                                          make_unique<TargetSelectionTaskNode>(blackboard)},
                            "PriorityParNode") { };

    PrintState printState(const ServerState & state, CSGOId playerId) const override {
        PrintState printState = ParSelectorNode::printState(state, playerId);
        printState.curState = {blackboard.playerToPriority[playerId].print(state)};
        return printState;
    }
};

#endif //CSKNOW_PRIORITY_PAR_NODE_H
