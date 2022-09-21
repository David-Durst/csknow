//
// Created by durst on 6/10/22.
//

#include "bots/behavior_tree/blackboard.h"

PrintState Blackboard::printStrategyState(const ServerState &state) {
    PrintState printState;
    printState.curState = strategy.print(state);
    return printState;
}

PrintState Blackboard::printCommunicateState(const ServerState &state) {
    PrintState printState;

    printState.curState.push_back(tMemory.print(state));
    printState.curState.push_back(ctMemory.print(state));

    printState.appendNewline = true;
    return printState;
}

vector<PrintState> Blackboard::printPerPlayerState(const ServerState &state, CSGOId playerId) {
    const ServerState::Client & curClient = state.getClient(playerId);
    vector<PrintState> printStates;

    string dangerArea = "none";
    if (playerToDangerAreaId.find(playerId) != playerToDangerAreaId.end()) {
        dangerArea = std::to_string(playerToDangerAreaId.find(playerId)->second);
    }

    string aggressiveTypeString;
    switch (playerToTreeThinkers[playerId].aggressiveType) {
        case AggressiveType::Push:
            aggressiveTypeString = "Push";
            break;
        case AggressiveType::Bait:
            aggressiveTypeString = "Bait";
            break;
        default:
            aggressiveTypeString = "invalid";
    }

    printStates.emplace_back(state.getPlayerString(playerId) +
                                  ", aggressive type " + aggressiveTypeString +
                                  ", pos " + curClient.getFootPosForPlayer().toString() +
                                  ", cur nav area " + std::to_string(navFile.get_nearest_area_by_position(
                                          vec3Conv(curClient.getFootPosForPlayer())).get_id()) +
                                  ", danger area " + dangerArea);
    printStates.emplace_back(playerToPriority[playerId].print(state));
    printStates.emplace_back(playerToPath[playerId].print(state, navFile));
    printStates.emplace_back(playerToAction[playerId].print());
    printStates.emplace_back(playerToMemory[playerId].print(state));

    return printStates;
}
