//
// Created by durst on 9/15/22.
//

#include "queries/rolling_window.h"

map<int64_t, int64_t> getPATIdForPlayerId(const Ticks & ticks, const PlayerAtTick & playerAtTick, int64_t tickIndex) {
    map<int64_t, int64_t> playerIdToPatID;
    for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
         patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
        playerIdToPatID[playerAtTick.playerId[patIndex]] = patIndex;
    }
    return playerIdToPatID;
}

map<int64_t, vector<int64_t>> getPerPlayerPATIdsInTemporalRange(const Rounds & rounds, const Ticks & ticks,
                                                                const PlayerAtTick & playerAtTick, int64_t curTick,
                                                                const TickRates & tickRates,
                                                                double secondsBefore, double secondsAfter) {
    map<int64_t, vector<int64_t>> result;
    int64_t startTick = getLookbackDemoTick(rounds, ticks, playerAtTick, curTick, tickRates, secondsBefore),
        endTick = getLookforwardDemoTick(rounds, ticks, playerAtTick, curTick, tickRates, secondsAfter);
    for (size_t tickIndex = startTick; tickIndex <= endTick; tickIndex++) {
        map<int64_t, int64_t> playerToPAT = getPATIdForPlayerId(ticks, playerAtTick, tickIndex - 1);
        for (const auto [playerId, patId] : playerToPAT) {
            result[playerId].push_back(patId);
        }
    }
    return result;
}
