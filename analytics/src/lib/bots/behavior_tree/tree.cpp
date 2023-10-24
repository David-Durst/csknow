//
// Created by durst on 5/9/22.
//

#include "bots/behavior_tree/tree.h"
#include "bots/behavior_tree/global/global_node.h"
#include "queries/inference_moments/inference_latent_engagement.h"

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
        // occasionally nav file fails to load, just reload it (happens infrequently so this hack is ok)
        bool makeBlackboardSuccesfully = false;
        while (!makeBlackboardSuccesfully) {
            try {
                if (state.mapNumber != curMapNumber) {
                    blackboard = make_unique<Blackboard>(navPath, state.mapName, inferenceManager,
                                                         featureStoreResult.defaultBuffer);
                }
                else {
                    blackboard = make_unique<Blackboard>(navPath, inferenceManager,
                                                         blackboard->visPoints, blackboard->nearestNavCell,
                                                         blackboard->mapMeshResult,
                                                         blackboard->reachability, blackboard->distanceToPlaces,
                                                         blackboard->navAboveBelow,
                                                         blackboard->ordersResult,
                                                         featureStoreResult.defaultBuffer);
                }
                featureStoreResult.teamFeatureStoreResult.setOrders(blackboard->ordersResult.orders);
                makeBlackboardSuccesfully = true;
            }
            catch (const std::exception & ex) {
                std::cout << "what():  " << ex.what() << std::endl;
            }
        }

        blackboard->removedAreas = {
                6938, 9026, // these are barrels on A that I get stuck on
                8251, // this one is under t spawn
                8631, // this one is on cat next to boxes, weird
                //4232, 4417, // bad wall and box on long
                8531, // mid doors ct side
                8753, 8550, 8574, // b car
                8594, 8600, 8601, 8602, 8607, // boxes under cat to a
                8966, 8967, 8970, 8969, 8968, // t spawn
                //8973 // under hole inside B
                3973, 3999, 4000, // out of bounds near b tunnels entrance from t
        };
        // get a connection for each area that isn't also invalid
        for (const auto & areaId : blackboard->removedAreas) {
            blackboard->removedAreaAlternatives[areaId] = INVALID_ID;
        }
        for (const auto & areaId : blackboard->removedAreas) {
            size_t areaIndex = blackboard->navFile.m_area_ids_to_indices[areaId];
            bool foundValidConnection = false;
            for (size_t j = 0; j < blackboard->navFile.connections_area_length[areaIndex]; j++) {
                size_t conAreaIndex = blackboard->navFile.connections[blackboard->navFile.connections_area_start[areaIndex] + j];
                AreaId conAreaId = blackboard->navFile.m_areas[conAreaIndex].get_id();
                if (blackboard->removedAreaAlternatives.find(conAreaId) == blackboard->removedAreaAlternatives.end()) {
                    foundValidConnection = true;
                    blackboard->removedAreaAlternatives[areaId] = conAreaId;
                    break;
                }
            }
            // if not connected to anything valid, just get the nearest valid
            if (!foundValidConnection) {
                Vec3 curAreaCenter = getCenter(blackboard->mapMeshResult.coordinate[areaIndex]);
                double minDistance = std::numeric_limits<double>::infinity();
                for (size_t otherAreaIndex = 0; otherAreaIndex < blackboard->mapMeshResult.areaId.size(); otherAreaIndex++) {
                    AABB otherAreaCoordinate = blackboard->mapMeshResult.coordinate[otherAreaIndex];
                    double newDistance = computeDistance((curAreaCenter), getCenter(otherAreaCoordinate));
                    AreaId otherAreaId = blackboard->mapMeshResult.areaId[otherAreaIndex];
                    if (blackboard->removedAreaAlternatives.count(otherAreaId) == 0 && newDistance < minDistance) {
                        blackboard->removedAreaAlternatives[areaId] = otherAreaId;
                        minDistance = newDistance;
                    }
                }
            }
        }
        blackboard->navFile.remove_incoming_edges_to_areas(blackboard->removedAreas);
        blackboard->navFile.remove_edges({
            {1650, 1644}, // wall near b car clips into end of 1650 preventing getting to 1644
            {1650, 1683}, // wall near b car clips into end of 1650 preventing getting to 1644
            {8547, 7911}, // b car annoying
            {3743, 3745}, // box near b doors
            {9034, 9031}, // mid to b doors
            {5241, 6714}, // x box
            {5241, 6700}, // x box
        });
        globalNode = make_unique<GlobalNode>(*blackboard);
        priorityNode = make_unique<PriorityNode>(*blackboard);
        actionNode = make_unique<ActionNode>(*blackboard);
        curMapNumber = state.mapNumber;
    }
    else {
        newBlackboard = false;
    }

    // insert tree thinkers and memories for new bots
    addTreeThinkersToBlackboard(state, blackboard.get());

    // compute the ids that are valid
    if (filterMutex.try_lock()) {
        localLogFilterNames = sharedLogFilterNames;
        filterMutex.unlock();
    }

    // clear history state on round number
    bool newRound = false;
    if (state.roundNumber != curRoundNumber) {
        newRound = true;
        curRoundNumber = state.roundNumber;
        featureStoreResult.defaultBuffer.clearHistory();
    }

    // also clear history on teleports (aka teleport confirmation id doesn't equal telport id)
    bool newTeleport = false;
    for (const auto & client : state.clients) {
        if (client.lastTeleportConfirmationId != client.lastTeleportId) {
            newTeleport = true;
            break;
        }
    }
    if (newTeleport) {
        featureStoreResult.defaultBuffer.clearHistory();
    }

    if (!blackboard->playerToTreeThinkers.empty()) {
        vector<PrintState> printStates;

        // wait until plant to do anything (HACK FOR NOW)

        /*
        if (!state.c4IsPlanted) {
            return;
        }
         */

        // update streaming analytics database used in tree
        // blackboard->streamingManager.update(state);

        blackboard->featureStorePreCommitBuffer.updateFeatureStoreBufferPlayers(state, newRound);
        //updateStateVisibility(state, *blackboard);

        // update cur clients before global node so have playerToInferenceData ready
        inferenceManager.setCurClients(state.clients);
        inferenceManager.teamSaveControlParameters.update(state);

        // update all nodes in tree
        // don't care about which player as order is for all players
        globalNode->exec(state, defaultThinker);
        featureStoreResult.teamFeatureStoreResult.commitTeamRow(state, featureStoreResult.defaultBuffer,
                                                                blackboard->distanceToPlaces, blackboard->navFile );
        printStates.push_back(globalNode->printState(state, defaultThinker.csgoId));
        printStates.push_back(blackboard->printStrategyState(state));
        printStates.push_back(blackboard->printCommunicateState(state));

        inferenceManager.recordInputFeatureValues(featureStoreResult);
        featureStoreResult.teamFeatureStoreResult.reinit();

        for (auto & client : state.clients) {
            // disable force and absolute positioning for all players, testing infrastructure can set force
            blackboard->playerToAction[client.csgoId].forceInput = false;
            blackboard->playerToAction[client.csgoId].enableAbsPos = false;
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

            state.setInputs(client.csgoId, clientAction.lastTeleportConfirmationId, clientAction.buttons,
                            clientAction.intendedToFire, clientAction.inputAngleX, clientAction.inputAngleY,
                            clientAction.inputAngleAbsolute, clientAction.forceInput, clientAction.enableAbsPos,
                            clientAction.absPos, clientAction.absView);

            // log state
            if (localLogFilterNames.empty() || localLogFilterNames.find(state.getClient(treeThinker.csgoId).name) != localLogFilterNames.end()) {
                vector<PrintState> blackboardPrintStates = blackboard->printPerPlayerState(state, treeThinker.csgoId);
                printStates.insert(printStates.end(), blackboardPrintStates.begin(), blackboardPrintStates.end());
                printStates.push_back(priorityNode->printState(state, treeThinker.csgoId));
                printStates.push_back(actionNode->printState(state, treeThinker.csgoId));
                printStates.back().appendNewline = true;

            }

            //featureStoreResult.commitPlayerRow(featureStoreResult.defaultBuffer);
            //inferenceManager.recordPlayerValues(featureStoreResult, client.csgoId);
            featureStoreResult.reinit();
        }

        inferenceManager.runInferences();
        featureStoreResult.teamFeatureStoreResult.curBaiting = false;
        for (const auto & [_, treeThinker] : blackboard->playerToTreeThinkers) {
            if (treeThinker.aggressiveType == AggressiveType::Bait) {
                featureStoreResult.teamFeatureStoreResult.curBaiting = true;
                break;
            }
        }

        stringstream logCollector;
        logCollector << "inference time " << inferenceManager.inferenceSeconds << "s" << std::endl;
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
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
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
#pragma GCC diagnostic pop
}

void addTreeThinkersToBlackboard(const ServerState & state, Blackboard * blackboard) {
    // insert tree thinkers and memories for new bots
    bool haveCTPusher = false;
    for (const auto & [playerId, thinker] : blackboard->playerToTreeThinkers) {
        if (state.getClientSlowSafe(playerId).team == ENGINE_TEAM_CT && thinker.aggressiveType == AggressiveType::Push) {
            haveCTPusher = true;
        }
    }
    for (const auto & client : state.clients) {
        if (client.isBot && blackboard->playerToTreeThinkers.find(client.csgoId) == blackboard->playerToTreeThinkers.end()) {
            AggressiveType aggressiveType;
            if (ALL_PUSH) {
                aggressiveType = AggressiveType::Push;
            }
            else {
                if (client.team == ENGINE_TEAM_CT && !haveCTPusher) {
                    aggressiveType = AggressiveType::Push;
                }
                else {
                    aggressiveType = blackboard->aggressionDis(blackboard->gen) < 0.5 ?
                                     AggressiveType::Push : AggressiveType::Bait;
                    if (client.team == ENGINE_TEAM_CT && aggressiveType == AggressiveType::Push) {
                        haveCTPusher = true;
                    }
                }
            }
            blackboard->playerToTreeThinkers[client.csgoId] = {
                client.csgoId,
                aggressiveType,
                {100, 60, 1200, 600},
                2.5
            };
            blackboard->playerToMemory[client.csgoId] = {
                false,
                client.team,
                client.csgoId,
                { }
            };
            blackboard->playerToMouseController.insert({client.csgoId, SecondOrderController(1.6, 0.76, 0.5)});
        }
    }
}
