//
// Created by durst on 5/9/22.
//

#include "bots/behavior_tree/node.h"
#include "bots/behavior_tree/order_node.h"
#include "bots/behavior_tree/priority/priority_par_node.h"
#include "bots/behavior_tree/implementation_node.h"
#include "bots/behavior_tree/action_node.h"
#include <memory>

#ifndef CSKNOW_TREE_H
#define CSKNOW_TREE_H


class Tree {
    // one order node overall, sets all team behavior
    std::unique_ptr<OrderSeqSelectorNode> orderNode;
    map<CSGOId, vector<Node>> playerToNodes;
    map<CSGOId, TreeThinker> playerToTreeThinkers;
    std::unique_ptr<Blackboard> blackboard;
    int32_t curMapNumber = INVALID_ID;

public:
    void tick(ServerState & state, string mapsPath) {
        string navPath = mapsPath + "/" + state.mapName + ".nav";

        if (state.mapNumber != curMapNumber) {
            blackboard = std::unique_ptr<Blackboard>( new Blackboard(navPath) );
            orderNode = std::unique_ptr<OrderSeqSelectorNode>( new OrderSeqSelectorNode(*blackboard) );
            playerToNodes.clear();
            curMapNumber = state.mapNumber;
        }

        for (const auto & client : state.clients) {
            if (client.isBot && playerToNodes.find(client.csgoId) == playerToNodes.end()) {
                playerToNodes[client.csgoId] = {
                        PriorityParNode(*blackboard),
                        ImplementationParSelectorNode(*blackboard),
                        ActionParSelectorNode(*blackboard)
                };
                playerToTreeThinkers[client.csgoId] = {
                        client.csgoId,
                        AggressiveType::Push,
                        {100, 20, 40, 70},
                        0, 0
                };
            }
        }

        // just take any tree thinker for general node
        orderNode->exec(state, playerToTreeThinkers[state.clients[0].csgoId]);
        for (const auto & client : state.clients) {
            TreeThinker & treeThinker = playerToTreeThinkers[client.csgoId];
            for (auto & node : playerToNodes[client.csgoId]) {
                node.exec(state, treeThinker);
            }
        }
    }

};

#endif //CSKNOW_TREE_H
