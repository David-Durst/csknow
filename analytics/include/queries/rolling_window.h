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

typedef map<int64_t, CircularBuffer<int64_t>> PlayerToPATWindows;

enum class DurationType {
    Seconds,
    Ticks,
    NUM_DURATION_TYPES [[maybe_unused]]
};

struct WindowDuration {
    DurationType type = DurationType::Seconds;
    double secondsBefore = 0., secondsAfter = 0.;
    int64_t ticksBefore = 0, ticksAfter = 0;
};

class RollingWindow {
    const Rounds & rounds;
    const Ticks & ticks;
    const PlayerAtTick & playerAtTick;
    PlayerToPATWindows playerToPatWindows;
    set<int64_t> playersToCover;
    map<int64_t, int64_t> lastValidPATId;
    map<int64_t, int64_t> lastValidTickId;
    int64_t nextReadTickId = INVALID_ID;
    int64_t numTicks = INVALID_ID;
    WindowDuration curDuration;


public:
    RollingWindow(const Rounds & rounds, const Ticks & ticks, const PlayerAtTick & playerAtTick) :
        rounds(rounds), ticks(ticks), playerAtTick(playerAtTick) { };

    [[nodiscard]]
    map<int64_t, int64_t> getPATIdForPlayerId(int64_t tickIndex) const;

    void setTemporalRange(int64_t curTick, const TickRates & tickRates, WindowDuration duration);

    int64_t readNextTick();

    int64_t lastReadTickId() { return nextReadTickId - 1; };

    int64_t lastCurTickId() { return nextReadTickId - 1 - curDuration.ticksAfter; };

    [[nodiscard]]
    const PlayerToPATWindows & getWindows() { return playerToPatWindows; }
};


#endif //CSKNOW_ROLLING_WINDOW_H
