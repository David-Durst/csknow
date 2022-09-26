//
// Created by durst on 9/15/22.
//

#include "queries/rolling_window.h"
using std::pair;

map<int64_t, int64_t> RollingWindow::getPATIdForPlayerId(int64_t tickIndex) const {
    map<int64_t, int64_t> playerIdToPatID;
    for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
         patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
        playerIdToPatID[playerAtTick.playerId[patIndex]] = patIndex;
    }
    return playerIdToPatID;
}

void RollingWindow::setTemporalRange(int64_t curTick, const TickRates &tickRates, double secondsBefore,
                                     double secondsAfter) {
    playerToPATWindow.clear();
    int64_t startTick = getLookbackDemoTick(rounds, ticks, curTick, tickRates, secondsBefore),
        endTick = getLookforwardDemoTick(rounds, ticks, curTick, tickRates, secondsAfter);
    numTicks = endTick - startTick + 1;

    // get all players that appear anywhere in ticks
    // init last valid tick with first valid tick, this handles windows where a player doesn't appear at start
    lastValidPATId.clear();
    for (int64_t tickIndex = startTick; tickIndex <= endTick; tickIndex++) {
        for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
             patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
            int64_t playerId = playerAtTick.playerId[patIndex];
            if (playerToPATWindow.find(playerId) == playerToPATWindow.end()) {
                playerToPATWindow.insert({playerId, CircularBuffer<int64_t>(numTicks)});
                lastValidPATId[playerId] = patIndex;
            }
        }
    }

    // create record of players to check every tick
    playersToCover.clear();
    for (const auto & [playerId, _] : playerToPATWindow) {
        playersToCover.insert(playerId);
    }

    // insert values, repeating priors ones if any missing. If on first tick, then look ahead to first valid pat id.
    // can't have problem if startTick = endTick because then only 1 tick, so no missing players in that tick
    // that are in other ticks
    nextReadTickId
    for (int64_t tickIndex = startTick; tickIndex <= endTick; tickIndex++) {
        set<int64_t> playersCurTick;
        for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
             patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
            int64_t playerId = playerAtTick.playerId[patIndex];
            playerToPATWindow.at(playerId).enqueue(patIndex);
            playersCurTick.insert(playerId);
            lastValidPATId[playerId] = patIndex;
        }
        vector<int64_t> playersToRepeat;
        std::set_difference(playersToCover.begin(), playersToCover.end(),
                            playersCurTick.begin(), playersCurTick.end(),
                            std::inserter(playersToRepeat, playersToRepeat.begin()));
        for (const auto & playerId : playersToRepeat) {
            playerToPATWindow.at(playerId).enqueue(lastValidPATId[playerId]);
        }
    }
}

void RollingWindow::readNextTick() {
    // repeating priors ones if any missing. If on first tick, then look ahead to first valid pat id.
    // can't have problem if startTick = endTick because then only 1 tick, so no missing players in that tick
    // that are in other tickssCurTick;
    set<int64_t> playersCurTick;
    for (int64_t patIndex = ticks.patPerTick[nextReadTickId].minId;
         patIndex <= ticks.patPerTick[nextReadTickId].maxId; patIndex++) {
        int64_t playerId = playerAtTick.playerId[patIndex];

        // if a new player not encountered during init, then fill entire region with cur value and set last to cur
        if (playerToPATWindow.find(playerId) == playerToPATWindow.end()) {
            playerToPATWindow.insert()
        }

        playerToPATWindow.at(playerId).enqueue(patIndex);
        playersCurTick.insert(playerId);
        lastValidPATId[playerId] = patIndex;
    }
    vector<int64_t> playersToRepeat;
    std::set_difference(playersToCover.begin(), playersToCover.end(),
                        playersCurTick.begin(), playersCurTick.end(),
                        std::inserter(playersToRepeat, playersToRepeat.begin()));
    for (const auto & playerId : playersToRepeat) {
        playerToPATWindow.at(playerId).enqueue(lastValidPATId[playerId]);
    }
    nextReadTickId++;
}
