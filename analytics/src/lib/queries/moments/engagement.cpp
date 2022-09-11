//
// Created by durst on 9/11/22.
//

#include "queries/moments/engagement.h"
#include "queries/lookback.h"
#include <omp.h>
#include <atomic>

struct EngagementPlayers {
    int64_t attacker, victim;

    bool operator<(const EngagementPlayers & other) const {
        return attacker < other.attacker ||
               (attacker == other.attacker && victim < other.victim);
    }
};

struct EngagementTimes {
    int64_t startTick, endTick;
};

EngagementResult queryEngagementResult(const Games & games, const Rounds & rounds, const Ticks & ticks,
                                       const PlayerAtTick & playerAtTick, const Hurt & hurt) {

    int numThreads = omp_get_max_threads();
    vector<int64_t> tmpRoundIds[numThreads];
    vector<int64_t> tmpRoundStarts[numThreads];
    vector<int64_t> tmpRoundSizes[numThreads];
    vector<int64_t> tmpStartTickId[numThreads];
    vector<int64_t> tmpEndTickId[numThreads];
    vector<int64_t> tmpLength[numThreads];
    vector<vector<int64_t>> tmpPlayerId[numThreads];
    vector<vector<EngagementRole>> tmpRole[numThreads];
    std::atomic<int64_t> roundsProcessed = 0;

    // for each round
    // track events for each pairs of player.
    // start a new event for a pair when hurt event with no prior one or far away prior one
    // clear out all hurt events on end of round
#pragma omp parallel for
    for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
        int threadNum = omp_get_thread_num();
        tmpRoundIds[threadNum].push_back(roundIndex);
        tmpRoundStarts[threadNum].push_back(tmpStartTickId[threadNum].size());

        TickRates tickRates = computeTickRates(games, rounds, roundIndex);

        map<EngagementPlayers, EngagementTimes> curEngagements;

        for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
             tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {
            for (int64_t hurtIndex : ticks.hurtPerTick.at(tickIndex)) {
                EngagementPlayers curPair{hurt.attacker[hurtIndex], hurt.victim[hurtIndex]};

                // start new engagement if none present
                if (curEngagements.find(curPair) == curEngagements.end()) {
                    curEngagements[curPair] = {tickIndex, tickIndex};
                }
                else {
                    EngagementTimes & lastTimes = curEngagements[curPair];
                    // if current engagement hasn't ended, extend it
                    if (secondsBetweenTicks(ticks, tickRates, lastTimes.endTick, tickIndex)
                        <= POST_ENGAGEMENT_SECONDS) {
                        lastTimes.endTick = tickIndex;
                    }
                    // if current engagement ended, finish it and start new one
                    else {
                        // use pre and post periods to track behavior around engagement

                        int64_t preEngagementStart = getLookbackDemoTick(rounds, ticks, playerAtTick,
                                                                         lastTimes.startTick, tickRates,
                                                                         PRE_ENGAGEMENT_SECONDS);
                        int64_t postEngagementEnd = getLookforwardDemoTick(rounds, ticks, playerAtTick,
                                                                         lastTimes.startTick, tickRates,
                                                                         POST_ENGAGEMENT_SECONDS);
                        tmpStartTickId[threadNum].push_back(preEngagementStart);
                        tmpEndTickId[threadNum].push_back(postEngagementEnd);
                        tmpLength[threadNum].push_back(postEngagementEnd - preEngagementStart + 1);
                        tmpPlayerId[threadNum].push_back({curPair.attacker, curPair.victim});
                        tmpRole[threadNum].push_back({EngagementRole::Attacker, EngagementRole::Victim});
                    }
                }
            }
            tmpRoundSizes[threadNum].push_back(tmpStartTickId[threadNum].size() - tmpRoundStarts[threadNum].back());
        }

    }

    EngagementResult result;
    mergeThreadResults(numThreads, result.rowIndicesPerRound, tmpRoundIds, tmpRoundStarts, tmpRoundSizes,
                       result.startTickId, result.size,
                       [&](int64_t minThreadId, int64_t tmpRowId) {
                           result.startTickId.push_back(tmpStartTickId[minThreadId][tmpRowId]);
                           result.endTickId.push_back(tmpEndTickId[minThreadId][tmpRowId]);
                           result.tickLength.push_back(tmpLength[minThreadId][tmpRowId]);
                           result.playerId.push_back(tmpPlayerId[minThreadId][tmpRowId]);
                           result.role.push_back(tmpRole[minThreadId][tmpRowId]);
                       });
    return result;
}
