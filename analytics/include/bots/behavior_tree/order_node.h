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
        D2TaskNode(Blackboard & blackboard) : Node(blackboard, "D2TaskNode") { };
        virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };

    class GeneralTaskNode : public Node {
    public:
        GeneralTaskNode(Blackboard & blackboard) : Node(blackboard, "GeneralTaskNode") { };
        virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };
}

class OrderSeqSelectorNode : public FirstNonFailSeqSelectorNode {
public:
    OrderSeqSelectorNode(Blackboard & blackboard) :
            FirstNonFailSeqSelectorNode(blackboard, {order::D2TaskNode(blackboard), order::GeneralTaskNode(blackboard)},
                                        "OrderSeqSelectorNode") {
        auto z = order::D2TaskNode(blackboard);
        int q = 1;
    };


    PrintState printState(const ServerState & state, CSGOId playerId) const override {
        PrintState printState = FirstNonFailSeqSelectorNode::printState(state, playerId);

        printState.curState.clear();
        for (size_t i = 0; i < blackboard.orders.size(); i++) {
            const auto & order = blackboard.orders[i];
            vector<string> orderResult = order.print(state, i);
            printState.curState.insert(printState.curState.end(), orderResult.begin(), orderResult.end());
        }

        return printState;
    }
};

#endif //CSKNOW_ORDER_NODE_H
