//
// Created by durst on 5/9/22.
//

#include "bots/behavior_tree/tree.h"

void Tree::tick(ServerState & state, string mapsPath) {
    string navPath = mapsPath + "/" + state.mapName + ".nav";

    if (state.mapNumber != curMapNumber) {
        blackboard = make_unique<Blackboard>(navPath);
        orderNode = make_unique<OrderSeqSelectorNode>(*blackboard);
        priorityNode = make_unique<PriorityParNode>(*blackboard);
        implementationNode = make_unique<ImplementationParSelectorNode>(*blackboard);
        actionNode = make_unique<ActionParSelectorNode>(*blackboard);
        curMapNumber = state.mapNumber;
    }

    // insert tree thinkers for new bots
    for (const auto & client : state.clients) {
        if (client.isBot && playerToTreeThinkers.find(client.csgoId) == playerToTreeThinkers.end()) {
            playerToTreeThinkers[client.csgoId] = {
                    client.csgoId,
                    AggressiveType::Push,
                    {100, 20, 40, 70},
                    0, 0
            };
        }
    }

    if (!state.clients.empty()) {
        vector<PrintState> printStates;

        // update all nodes in tree
        // don't care about which player as order is for all players
        orderNode->exec(state, playerToTreeThinkers[state.clients[0].csgoId]);
        printStates.push_back(orderNode->printState(state, state.clients[0].csgoId));

        for (auto & client : state.clients) {
            if (!client.isAlive || !client.isBot) {
                continue;
            }
            TreeThinker & treeThinker = playerToTreeThinkers[client.csgoId];
            // reset all buttons before logic runs
            blackboard->playerToAction[treeThinker.csgoId].buttons = 0;

            priorityNode->exec(state, treeThinker);
            implementationNode->exec(state, treeThinker);
            actionNode->exec(state, treeThinker);

            // update state actions with actions per player
            const Action & clientAction = blackboard->playerToAction[client.csgoId];

            state.setInputs(client.csgoId, clientAction.buttons, clientAction.inputAngleDeltaPctX,
                            clientAction.inputAngleDeltaPctY);
            //state.setInputs(client.csgoId, 0, clientAction.inputAngleDeltaPctX,
            //                clientAction.inputAngleDeltaPctY);

            // log state
            printStates.push_back({{}, {state.getPlayerString(client.csgoId) +
                ", pos: " + client.getFootPosForPlayer().toString()}});
            printStates.push_back(priorityNode->printState(state, treeThinker.csgoId));
            printStates.push_back(implementationNode->printState(state, treeThinker.csgoId));
            printStates.push_back(actionNode->printState(state, treeThinker.csgoId));
        }

        stringstream logCollector;
        for (const auto & printState : printStates) {
            logCollector << printState.getState() << std::endl;
        }
        curLog = logCollector.str();
    }
}

