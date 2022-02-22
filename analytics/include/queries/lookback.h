//
// Created by durst on 1/2/22.
//

#ifndef CSKNOW_LOOKBACK_H
#define CSKNOW_LOOKBACK_H

#include "load_data.h"
#include <cmath>

// max time you can look back in ms
// this assumes tick in demo for every tick in game
const int maxLookBackTime = 300;
const int clInterp = 31;

struct TickRates {
    int demoTickRate;
    int gameTickRate;
};

static inline __attribute__((always_inline))
TickRates computeTickRates(const Games & games, const Rounds & rounds, int64_t roundIndex) {
    int demoTickRate = static_cast<int>(games.demoTickRate[rounds.gameId[roundIndex]]);
    int gameTickRate = static_cast<int>(games.gameTickRate[rounds.gameId[roundIndex]]);
    return {demoTickRate, gameTickRate};
}

static inline __attribute__((always_inline))
int computeMaxLookbackDemoTicks(const TickRates & tickRates) {
    return ceil(tickRates.demoTickRate * maxLookBackTime / 1000.0);
}

static inline __attribute__((always_inline))
double secondsBetweenTicks(const Ticks & ticks, TickRates tickRates, int64_t startTick, int64_t endTick) {
    return (ticks.gameTickNumber[endTick] - ticks.gameTickNumber[startTick]) / static_cast<double>(tickRates.gameTickRate);
}

/**
 * Convert number of game ticks back to number of demo ticks back
 * @param ticks vector of demo ticks
 * @param playerAtTick vector of player at ticks
 * @param tickIndex current demo tick index
 * @param patIndex current player at tick index
 * @param tickRates tick rates for the game
 * @return
 */
static inline __attribute__((always_inline))
int64_t getLookbackDemoTick(const Rounds & rounds, const Ticks & ticks, const PlayerAtTick & playerAtTick, const int64_t tickIndex,
                            const TickRates & tickRates, const int64_t lookbackGameTicks) {
    int maxLookBackDemoTicks = computeMaxLookbackDemoTicks(tickRates);
    int lookbackDemoTicks = 1;
    for (; ticks.gameTickNumber[tickIndex - lookbackDemoTicks] > ticks.gameTickNumber[tickIndex] - lookbackGameTicks &&
           lookbackDemoTicks < maxLookBackDemoTicks &&
           tickIndex - lookbackDemoTicks > rounds.startTick[ticks.roundId[tickIndex]];
           lookbackDemoTicks++);
    return lookbackDemoTicks;
}

#endif //CSKNOW_LOOKBACK_H
