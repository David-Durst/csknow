//
// Created by durst on 5/9/22.
//

#ifndef CSKNOW_PRIORITY_NODE_H
#define CSKNOW_PRIORITY_NODE_H

#include "bots/behavior_tree/node.h"
#include "bots/behavior_tree/priority/follow_order_node.h"
#include "bots/behavior_tree/priority/engage_node.h"

class PriorityNode : public SelectorNode {
public:
    PriorityNode(Blackboard & blackboard) :
            SelectorNode(blackboard, Node::makeList(
                                                make_unique<EnemyEngageCheckNode>(blackboard),
                                                make_unique<NoEnemyOrderCheckNode>(blackboard)),
                            "PriorityNode") { };

    /*
    virtual PrintState printState(const ServerState & state, CSGOId playerId) const override {
        PrintState printState = ParSelectorNode::printState(state, playerId);
        printState.curState.push_back(blackboard.playerToPriority[playerId].print(state));
        return printState;
    }
     */
};

#endif //CSKNOW_PRIORITY_NODE_H
