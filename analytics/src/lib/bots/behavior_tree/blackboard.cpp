//
// Created by durst on 6/10/22.
//

#include "bots/behavior_tree/blackboard.h"

PrintState Blackboard::printOrderState(const ServerState &state) {
    PrintState printState;

    map<CSGOId, int64_t> playerToCurWaypoint;
    for (const auto & [csgoId, treeThinker] : playerToTreeThinkers) {
        playerToCurWaypoint[csgoId] = treeThinker.orderWaypointIndex;
    }

    for (size_t i = 0; i < orders.size(); i++) {
        const auto & order = orders[i];
        vector<string> orderResult = order.print(playerToCurWaypoint, state, i);
        printState.curState.insert(printState.curState.end(), orderResult.begin(), orderResult.end());
    }

    printState.appendNewline = true;
    return printState;
}

vector<PrintState> Blackboard::printPerPlayerState(const ServerState &state, CSGOId playerId) {
    const ServerState::Client & curClient = state.getClient(playerId);
    vector<PrintState> printStates;

    printStates.push_back(state.getPlayerString(playerId) +
                                ", pos: " + curClient.getFootPosForPlayer().toString() +
                                ", cur nav area " + std::to_string(navFile.get_nearest_area_by_position(
                                                        vec3Conv(curClient.getFootPosForPlayer())).get_id()));
    printStates.push_back(playerToPriority[playerId].print(state));
    printStates.push_back(playerToPath[playerId].print(state, navFile));
    printStates.push_back({playerToAction[playerId].print(), true});

    return printStates;
}
