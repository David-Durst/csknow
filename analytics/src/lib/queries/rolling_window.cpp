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

void RollingWindow::setTemporalRange(int64_t curTick, const TickRates &tickRates, WindowDuration duration) {
    playerToPatWindows.clear();
    int64_t startTick, endTick;
    if (duration.type == DurationType::Seconds) {
        startTick = getLookbackDemoTick(rounds, ticks, curTick, tickRates, duration.secondsBefore);
        endTick = getLookforwardDemoTick(rounds, ticks, curTick, tickRates, duration.secondsAfter);
    }
    else {
        startTick = getLookbackDemoTick(rounds, ticks, curTick, duration.ticksBefore);
        endTick = getLookforwardDemoTick(rounds, ticks, curTick, duration.ticksAfter);
    }
    numTicks = endTick - startTick + 1;

    // get all players that appear anywhere in ticks
    // init last valid tick with first valid tick, this handles windows where a player doesn't appear at start
    // but appear in first window
    // readNextTick handles those after first window
    lastValidPATId.clear();
    for (int64_t tickIndex = startTick; tickIndex <= endTick; tickIndex++) {
        for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
             patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
            int64_t playerId = playerAtTick.playerId[patIndex];
            if (playerToPatWindows.find(playerId) == playerToPatWindows.end()) {
                playerToPatWindows.insert({playerId, CircularBuffer<int64_t>(numTicks)});
                lastValidPATId[playerId] = patIndex;
            }
        }
    }

    // create record of players to check every tick
    playersToCover.clear();
    for (const auto & [playerId, _] : playerToPatWindows) {
        playersToCover.insert(playerId);
    }

    nextReadTickId = startTick;
    for (int64_t tickIndex = startTick; tickIndex <= endTick; tickIndex++) {
        readNextTick();
    }
}

void RollingWindow::readNextTick() {
    set<int64_t> playersCurTick;
    for (int64_t patIndex = ticks.patPerTick[nextReadTickId].minId;
         patIndex <= ticks.patPerTick[nextReadTickId].maxId; patIndex++) {
        int64_t playerId = playerAtTick.playerId[patIndex];

        // if a new player not encountered during init window, then fill entire region with cur value and set last to cur
        if (playerToPatWindows.find(playerId) == playerToPatWindows.end()) {
            playerToPatWindows.insert({playerId, CircularBuffer<int64_t>(numTicks)});
            lastValidPATId[playerId] = patIndex;
            playerToPatWindows.at(playerId).fill(patIndex);
        }

        playerToPatWindows.at(playerId).enqueue(patIndex);
        playersCurTick.insert(playerId);
        lastValidPATId[playerId] = patIndex;
    }

    // if player in window but not this tick, repeat last seen value.
    // If on first tick in window, then look ahead to first valid pat id (done via lastValidPATId filled on init).
    // can't have problem if startTick = endTick because then only 1 tick, so no missing players in that tick
    // that are in other ticks.
    vector<int64_t> playersToRepeat;
    std::set_difference(playersToCover.begin(), playersToCover.end(),
                        playersCurTick.begin(), playersCurTick.end(),
                        std::inserter(playersToRepeat, playersToRepeat.begin()));
    for (const auto & playerId : playersToRepeat) {
        playerToPatWindows.at(playerId).enqueue(lastValidPATId[playerId]);
    }
    nextReadTickId++;
}
