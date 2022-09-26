//
// Created by durst on 9/15/22.
//

#ifndef CSKNOW_ROLLING_WINDOW_H
#define CSKNOW_ROLLING_WINDOW_H

#include "queries/query.h"
#include "queries/lookback.h"
#include "circular_buffer.h"
#include <map>

using std::map;

class RollingWindow {
    const Rounds & rounds;
    const Ticks & ticks;
    const PlayerAtTick & playerAtTick;
    map<int64_t, CircularBuffer<int64_t>> playerToPATWindow;
    set<int64_t> playersToCover;
    map<int64_t, int64_t> lastValidPATId;
    int64_t nextReadTickId = INVALID_ID;
    int64_t numTicks = INVALID_ID;


public:
    RollingWindow(const Rounds & rounds, const Ticks & ticks, const PlayerAtTick & playerAtTick) :
        rounds(rounds), ticks(ticks), playerAtTick(playerAtTick) { };

    map<int64_t, int64_t> getPATIdForPlayerId(int64_t tickIndex) const;

    void setTemporalRange(int64_t curTick, const TickRates & tickRates, double secondsBefore, double secondsAfter);

    void readNextTick();
};


#endif //CSKNOW_ROLLING_WINDOW_H
