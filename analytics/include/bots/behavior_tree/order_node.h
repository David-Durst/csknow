//
// Created by durst on 5/1/22.
//

#ifndef CSKNOW_ORDER_NODE_H
#define CSKNOW_ORDER_NODE_H

#include "bots/behavior_tree/node.h"
#include <map>
#include <memory>

namespace order {
    class D2OrderNode : public Node {
    public:
        D2OrderNode(Blackboard & blackboard) : Node(blackboard, "D2TaskNode") { };
        virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };

    class GeneralOrderNode : public Node {
    public:
        GeneralOrderNode(Blackboard & blackboard) : Node(blackboard, "GeneralTaskNode") { };
        virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };
}

class OrderNode : public SelectorNode {
public:
    OrderNode(Blackboard & blackboard) :
            SelectorNode(blackboard, Node::makeList(
                                                            make_unique<order::D2OrderNode>(blackboard)),
                                        "OrderNode") { };


    virtual PrintState printState(const ServerState & state, CSGOId playerId) const override {
        PrintState printState = SelectorNode::printState(state, playerId);

        map<CSGOId, int64_t> playerToCurWaypoint;
        for (const auto & [csgoId, treeThinker] : blackboard.playerToTreeThinkers) {
            playerToCurWaypoint[csgoId] = treeThinker.orderWaypointIndex;
        }

        for (size_t i = 0; i < blackboard.orders.size(); i++) {
            const auto & order = blackboard.orders[i];
            vector<string> orderResult = order.print(playerToCurWaypoint, state, i);
            printState.curState.insert(printState.curState.end(), orderResult.begin(), orderResult.end());
        }

        return printState;
    }
};

#endif //CSKNOW_ORDER_NODE_H
