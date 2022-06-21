//
// Created by steam on 6/20/22.
//

#include "bots/testing/script.h"
#include <iterator>

void Script::initialize(ServerState & state, string navPath) {
    commands.push_back(make_unique<InitTestingRound>());

    // allocate the bots to use
    set<CSGOId> usedBots;
    for (auto &neededBot: neededBots) {
        for (const auto &client: state.clients) {
            if (client.isBot && client.team == neededBot.team &&
                usedBots.find(client.csgoId) == usedBots.end()) {
                neededBot.id = client.csgoId;
            }
        }
    }
}

vector<string> Script::generateCommands(ServerState & state) {
    for (const auto & client : state.clients) {
        if (!client.isBot) {
            if (observeSettings.observeType == ObserveType::FirstPerson) {
                CSGOId neededBotCSGOId = neededBots[observeSettings.neededBotIndex].id;
                commands.push_back(make_unique<SpecPlayerToTarget>(client.name, state.getClient(neededBotCSGOId).name, false));
            }
            else if (observeSettings.observeType == ObserveType::ThirdPerson) {
                CSGOId neededBotCSGOId = neededBots[observeSettings.neededBotIndex].id;
                commands.push_back(make_unique<SpecPlayerToTarget>(client.name, state.getClient(neededBotCSGOId).name, true));
            }
            else if (observeSettings.observeType == ObserveType::Absolute) {
                commands.push_back(make_unique<SpecGoto>(client.name, observeSettings.cameraOrigin, observeSettings.cameraPos));
            }
        }
    }

    commands.push_back(make_unique<InitTestingRound>());
    commands.insert(commands.end(), std::make_move_iterator(logicCommands.begin()), std::make_move_iterator(logicCommands.end()));

    std::cout << "started " << name << std::endl;

    vector<string> result;
    for (const auto & command : commands) {
        result.push_back(command->ToString());
    }
    return result;
}

bool Script::tick(ServerState & state) {

    TreeThinker defaultThinker;
    defaultThinker.csgoId = 0;

    vector<PrintState> printStates;
    NodeState conditionResult = conditions->exec(state, defaultThinker);

    if (conditionResult == NodeState::Running) {
        return false;
    }
    else if (conditionResult == NodeState::Success) {
        std::cout << "succeeded" << std::endl;
    }
    else if (conditionResult == NodeState::Failure) {
        std::cout << "failed: " << std::endl;
        std::cout << conditions->printState(state, defaultThinker.csgoId).getState() << std::endl;
    }
    else {
        std::cout << "invalid state" << std::endl;
    }
    return true;
}
