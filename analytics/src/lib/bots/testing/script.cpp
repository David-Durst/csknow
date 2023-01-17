//
// Created by steam on 6/20/22.
//

#include "bots/testing/script.h"
#include <iterator>

void Script::initialize(Tree &, ServerState & state) {
    // allocate the bots to use
    set<CSGOId> usedBots;
    // get non-human bots
    for (auto &neededBot: neededBots) {
        for (const auto &client: state.clients) {
            if (client.isBot && client.team == neededBot.team && !neededBot.human &&
                usedBots.find(client.csgoId) == usedBots.end()) {
                neededBot.id = client.csgoId;
                usedBots.insert(client.csgoId);
                break;
            }
        }
    }
    // get human bots
    for (auto &neededBot: neededBots) {
        for (const auto &client: state.clients) {
            if (!client.isBot && client.team == neededBot.team && neededBot.human &&
                usedBots.find(client.csgoId) == usedBots.end()) {
                neededBot.id = client.csgoId;
                usedBots.insert(client.csgoId);
                break;
            }
        }
    }
}

/*
vector<string> Script::generateCommands(ServerState & state) {
    commands.push_back(make_unique<InitTestingRound>(blackboard));

    for (const auto & client : state.clients) {
        if (!client.isBot) {
            if (observeSettings.observeType == ObserveType::FirstPerson) {
                CSGOId neededBotCSGOId = neededBots[observeSettings.neededBotIndex].id;
                commands.push_back(make_unique<SpecPlayerToTarget>(blackboard, client.name, state.getClient(neededBotCSGOId).name, false));
            }
            else if (observeSettings.observeType == ObserveType::ThirdPerson) {
                CSGOId neededBotCSGOId = neededBots[observeSettings.neededBotIndex].id;
                commands.push_back(make_unique<SpecPlayerToTarget>(blackboard, client.name, state.getClient(neededBotCSGOId).name, true));
            }
            else if (observeSettings.observeType == ObserveType::Absolute) {
                commands.push_back(make_unique<SpecGoto>(blackboard, client.name, observeSettings.cameraOrigin, observeSettings.cameraAngle));
            }
        }
    }

    commands.insert(commands.end(), std::make_move_iterator(logicCommands.begin()), std::make_move_iterator(logicCommands.end()));

    std::cout << "started " << name << std::endl;

    vector<string> result;
    for (const auto & command : commands) {
        result.push_back(command->ToString());
    }
    return result;
}
 */

bool Script::tick(Tree & tree, ServerState & state) {

    TreeThinker defaultThinker(getDefaultThinker());

    vector<PrintState> printStates;
    NodeState conditionResult = commands->exec(state, defaultThinker);

    for (auto & client : state.clients) {
        if (!client.isAlive) {
            continue;
        }
        const Action & clientAction = tree.blackboard->playerToAction[client.csgoId];
        state.setInputs(client.csgoId, clientAction.lastTeleportConfirmationId, clientAction.buttons,
                        clientAction.intendedToFire, clientAction.inputAngleX, clientAction.inputAngleY,
                        clientAction.inputAngleAbsolute, clientAction.forceInput);
    }

    bool finished = true;
    if (conditionResult == NodeState::Running) {
        finished = false;
    }
    else if (initScript) { } // don't print for init script
    else if (conditionResult == NodeState::Success) {
        if (tree.blackboard->streamingManager.streamingTestLogger.testActive()) {
            tree.blackboard->streamingManager.streamingTestLogger.endTest(true);
        }
        std::cout << name << " succeeded" << std::endl;
    }
    else if (conditionResult == NodeState::Failure) {
        if (tree.blackboard->streamingManager.streamingTestLogger.testActive()) {
            tree.blackboard->streamingManager.streamingTestLogger.endTest(false);
        }
        std::cout << name << " failed: " << std::endl;
        std::cout << commands->printState(state, defaultThinker.csgoId).getState() << std::endl;
    }
    else {
        std::cout << name << " invalid state" << std::endl;
    }
    curLog = commands->printState(state, defaultThinker.csgoId).getState();
    return finished;
}

void ScriptsRunner::initialize(Tree & tree, ServerState & state) {
    if (curScript > 0) {
        for (auto & script : scripts) {
            script->initialize(tree, state);
        }
    }
    else {
        scripts[0]->initialize(tree, state);
    }
}

bool ScriptsRunner::tick(Tree & tree, ServerState & state) {
    if (startingNewScript) {
        // skip init printing
        if (curScript > 0) {
            std::cout << scripts[curScript]->name << " starting" << std::endl;
        }
        startingNewScript = false;
        tree.resetState = true;
    }
    else {
        tree.resetState = false;
    }
    for (const auto & neededBot : scripts[curScript]->getNeededBots()) {
        if (tree.blackboard->playerToTreeThinkers.find(neededBot.id) != tree.blackboard->playerToTreeThinkers.end()) {
            tree.blackboard->playerToTreeThinkers[neededBot.id].aggressiveType = neededBot.type;
        }
    }

    // wait until tree finished reseting
    if (!tree.resetState && scripts[curScript]->tick(tree, state)) {
        startingNewScript = true;
        curScript++;
    }
    if (curScript >= scripts.size()) {
        curScript = 0;
        if (restartOnFinish) {
            restart();
            return false;
        }
        else {
            return true;
        }
    }
    else {
        return false;
    }
}
