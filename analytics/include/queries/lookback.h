//
// Created by durst on 1/2/22.
//

#ifndef CSKNOW_LOOKBACK_H
#define CSKNOW_LOOKBACK_H

#include "load_data.h"
#include <cmath>

// max time you can look back in ms
// this assumes tick in demo for every tick in game
const int clInterp = 31;
const double CL_INTERP_SECONDS = 0.031;

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
int computeMaxLookDemoTicks(const TickRates & tickRates, int maxLookBackTime = 300) {
    return ceil(tickRates.demoTickRate * maxLookBackTime / 1000.0);
}

static inline __attribute__((always_inline))
int secondsToDemoTicks(const TickRates & tickRates, double seconds) {
    return ceil(tickRates.demoTickRate * seconds);
}

static inline __attribute__((always_inline))
double perSecondRateToPerDemoTickRate(const TickRates & tickRates, double perSecondRate) {
    return perSecondRate / tickRates.demoTickRate;
}

static inline __attribute__((always_inline))
int secondsToGameTicks(const TickRates & tickRates, double seconds) {
    return ceil(tickRates.gameTickRate * seconds);
}

static inline __attribute__((always_inline))
double gameTicksToSeconds(const TickRates & tickRates, int64_t gameTicks) {
    return static_cast<double>(gameTicks) / tickRates.gameTickRate;
}

static double secondsBetweenTicks(const Ticks & ticks, TickRates tickRates, int64_t startTick, int64_t endTick) {
    return (ticks.gameTickNumber[endTick] - ticks.gameTickNumber[startTick]) / static_cast<double>(tickRates.gameTickRate);
}

static double secondsBetweenTicks(TickRates tickRates, int64_t startGameTickNumber, int64_t endGameTickNumber) {
    return (endGameTickNumber - startGameTickNumber) / static_cast<double>(tickRates.gameTickRate);
}

static int64_t getLookbackDemoTick(const Rounds & rounds, const Ticks & ticks, int64_t tickIndex,
                                   int64_t lookbackGameTicks, int64_t bufferTicks = 0) {
    int lookbackDemoTicks = 0;
    for (; ticks.gameTickNumber[tickIndex - lookbackDemoTicks] > ticks.gameTickNumber[tickIndex] - lookbackGameTicks &&
           // this makes sure don't run off end, next tick is no less than min
           // last tick will be equal here or above, abort loop and return
           // subtract buffer ticks for buffer to end early on
           tickIndex - lookbackDemoTicks - bufferTicks > rounds.ticksPerRound[ticks.roundId[tickIndex]].minId;
           lookbackDemoTicks++);
    return tickIndex - lookbackDemoTicks;
}

static int64_t getLookbackDemoTick(const Rounds & rounds, const Ticks & ticks, int64_t tickIndex,
                                   const TickRates & tickRates, double lookBackTime, int64_t bufferTicks = 0) {
    int lookbackGameTicks = secondsToGameTicks(tickRates, lookBackTime);
    return getLookbackDemoTick(rounds, ticks, tickIndex, lookbackGameTicks, bufferTicks);
}
/**
 * Convert number of game ticks forward to number of demo ticks back
 * @param ticks vector of demo ticks
 * @param playerAtTick vector of player at ticks
 * @param tickIndex current demo tick index
 * @param patIndex current player at tick index
 * @param tickRates tick rates for the game
 * @return
 */
static int64_t getLookforwardDemoTick(const Rounds & rounds, const Ticks & ticks, int64_t tickIndex,
                                      int64_t lookforwardGameTicks, int64_t bufferTicks = 0) {
    int lookforwardDemoTicks = 0;
    for (; ticks.gameTickNumber[tickIndex + lookforwardDemoTicks] < ticks.gameTickNumber[tickIndex] + lookforwardGameTicks &&
           // this makes sure don't run off end, next tick is no more than max
           // add buffer ticks for buffer to end early on
           tickIndex + lookforwardDemoTicks + bufferTicks < rounds.ticksPerRound[ticks.roundId[tickIndex]].maxId;
           lookforwardDemoTicks++);
    return tickIndex + lookforwardDemoTicks;
}

static int64_t getLookforwardDemoTick(const Rounds & rounds, const Ticks & ticks, int64_t tickIndex,
                                      const TickRates & tickRates, double lookForwardTime, int64_t bufferTicks = 0) {
    int lookforwardGameTicks = secondsToGameTicks(tickRates, lookForwardTime);
    return getLookforwardDemoTick(rounds, ticks, tickIndex, lookforwardGameTicks, bufferTicks);
}

#endif //CSKNOW_LOOKBACK_H
