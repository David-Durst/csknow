//
// Created by durst on 9/15/22.
//

#ifndef CSKNOW_ROLLING_WINDOW_H
#define CSKNOW_ROLLING_WINDOW_H

#include "queries/query.h"
#include "queries/lookback.h"
#include <map>

using std::map;

map<int64_t, int64_t> getPATIdForPlayerId(const Ticks & ticks, const PlayerAtTick & playerAtTick, int64_t tickIndex);
map<int64_t, vector<int64_t>> getPerPlayerPATIdsInTemporalRange(const Rounds & rounds, const Ticks & ticks,
                                                                const PlayerAtTick & playerAtTick, int64_t curTick,
                                                                const TickRates & tickRates,
                                                                double secondsBefore, double secondsAfter, int64_t playerId);

#endif //CSKNOW_ROLLING_WINDOW_H
