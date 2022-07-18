//
// Created by durst on 5/9/22.
//

#include "bots/behavior_tree/tree.h"
#include "bots/behavior_tree/global/global_node.h"

void Tree::tick(ServerState & state, const string & mapsPath) {
    string navPath = mapsPath + "/" + state.mapName + ".nav";

    // track when players leave to recompute all plans
    bool samePlayers = true;
    if (state.clients.size() != lastFramePlayers.size()) {
        samePlayers = false;
    }
    else {
        for (const auto & client : state.clients) {
            if (lastFramePlayers.find(client.csgoId) == lastFramePlayers.end()) {
                samePlayers = false;
            }
        }
    }
    lastFramePlayers.clear();
    for (const auto & client : state.clients) {
        lastFramePlayers.insert(client.csgoId);
    }

    if (state.mapNumber != curMapNumber || !samePlayers || resetState) {
        newBlackboard = true;
        blackboard = make_unique<Blackboard>(navPath, state.mapName);
        blackboard->navFile.remove_incoming_edges_to_areas({6938, 9026});
        globalNode = make_unique<GlobalNode>(*blackboard);
        priorityNode = make_unique<PriorityNode>(*blackboard);
        actionNode = make_unique<ActionNode>(*blackboard);
        curMapNumber = state.mapNumber;
    }
    else {
        newBlackboard = false;
    }

    // insert tree thinkers and memories for new bots
    for (const auto & client : state.clients) {
        if (client.isBot && blackboard->playerToTreeThinkers.find(client.csgoId) == blackboard->playerToTreeThinkers.end()) {
            blackboard->playerToTreeThinkers[client.csgoId] = {
                    client.csgoId,
                    AggressiveType::Push,
                    {100, 20, 40, 70},
                    2.5, 0, 0
            };
            blackboard->playerToMemory[client.csgoId] = {
                    false,
                    client.team,
                    client.csgoId,
                    { }
            };
        }
    }

    // compute the ids that are valid
    if (filterMutex.try_lock()) {
        localLogFilterNames = sharedLogFilterNames;
        filterMutex.unlock();
    }

    if (!blackboard->playerToTreeThinkers.empty()) {
        vector<PrintState> printStates;

        // wait until plant to do anything (HACK FOR NOW)

        if (!state.c4IsPlanted) {
            return;
        }

        // update all nodes in tree
        // don't care about which player as order is for all players
        globalNode->exec(state, defaultThinker);
        printStates.push_back(globalNode->printState(state, defaultThinker.csgoId));
        printStates.push_back(blackboard->printOrderState(state));
        printStates.push_back(blackboard->printCommunicateState(state));

        for (auto & client : state.clients) {
            if (!client.isAlive || !client.isBot) {
                continue;
            }
            TreeThinker & treeThinker = blackboard->playerToTreeThinkers[client.csgoId];
            // reset all buttons before logic runs
            blackboard->lastPlayerToAction = blackboard->playerToAction;
            blackboard->playerToAction[treeThinker.csgoId].buttons = 0;
            // ensure default history
            blackboard->playerToPIDStateX[treeThinker.csgoId].errorHistory.fill(0.);
            blackboard->playerToPIDStateY[treeThinker.csgoId].errorHistory.fill(0.);

            priorityNode->exec(state, treeThinker);
            actionNode->exec(state, treeThinker);

            // update state actions with actions per player
            const Action & clientAction = blackboard->playerToAction[client.csgoId];

            state.setInputs(client.csgoId, clientAction.buttons, clientAction.inputAngleDeltaPctX,
                            clientAction.inputAngleDeltaPctY);
            //state.setInputs(client.csgoId, 0, clientAction.inputAngleDeltaPctX,
            //                clientAction.inputAngleDeltaPctY);

            // log state
            if (localLogFilterNames.empty() || localLogFilterNames.find(state.getClient(treeThinker.csgoId).name) != localLogFilterNames.end()) {
                vector<PrintState> blackboardPrintStates = blackboard->printPerPlayerState(state, treeThinker.csgoId);
                printStates.insert(printStates.end(), blackboardPrintStates.begin(), blackboardPrintStates.end());
                printStates.push_back(priorityNode->printState(state, treeThinker.csgoId));
                printStates.push_back(actionNode->printState(state, treeThinker.csgoId));
                printStates.back().appendNewline = true;

            }
        }

        stringstream logCollector;
        for (const auto & printState : printStates) {
            logCollector << printState.getState();
            if (printState.appendNewline) {
                logCollector << std::endl;
            }
        }
        curLog = logCollector.str();
    }
}

void Tree::readFilterNames() {
    string line, value;
#pragma clang diagnostic push
#pragma ide diagnostic ignored "EndlessLoop"
    while (true) {
        std::cin >> line;
        stringstream lineStream(line);

        // get lock only when have input
        const std::lock_guard<std::mutex> lock(filterMutex);
        sharedLogFilterNames.clear();
        bool includeAll = false;
        while (getline(lineStream, value, ',')) {
            if (!value.empty()) {
                sharedLogFilterNames.insert(value);
            }
            includeAll = value == "*";
        }
        if (includeAll) {
            sharedLogFilterNames.clear();
        }
    }
#pragma clang diagnostic pop
}
