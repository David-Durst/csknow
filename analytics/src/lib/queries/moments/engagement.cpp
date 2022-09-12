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

struct EngagementData {
    int64_t startTick, endTick;
    int16_t numHits;
};

void finishEngagement(const Rounds &rounds, const Ticks &ticks, const PlayerAtTick &playerAtTick,
                      vector<int64_t> tmpStartTickId[], vector<int64_t> tmpEndTickId[],
                      vector<int64_t> tmpFirstHurtTickId[], vector<int64_t> tmpLastHurtTickId[],
                      vector<int64_t> tmpLength[], vector<vector<int64_t>> tmpPlayerId[],
                      vector<vector<EngagementRole>> tmpRole[], vector<int16_t> tmpNumHits[],
                      int threadNum, const TickRates &tickRates,
                      const EngagementPlayers &curPair, const EngagementData &eData) {
    // use pre and post periods to track behavior around engagement
    int64_t preEngagementStart = getLookbackDemoTick(rounds, ticks, playerAtTick,
                                                     eData.startTick, tickRates,
                                                     PRE_ENGAGEMENT_SECONDS);
    int64_t postEngagementEnd = getLookforwardDemoTick(rounds, ticks, playerAtTick,
                                                       eData.endTick, tickRates,
                                                       POST_ENGAGEMENT_SECONDS);
    tmpStartTickId[threadNum].push_back(preEngagementStart);
    tmpEndTickId[threadNum].push_back(postEngagementEnd);
    tmpFirstHurtTickId[threadNum].push_back(eData.startTick);
    tmpLastHurtTickId[threadNum].push_back(eData.endTick);
    tmpLength[threadNum].push_back(postEngagementEnd - preEngagementStart + 1);
    tmpPlayerId[threadNum].push_back({curPair.attacker, curPair.victim});
    tmpRole[threadNum].push_back({EngagementRole::Attacker, EngagementRole::Victim});
    tmpNumHits[threadNum].push_back(eData.numHits);
}

EngagementResult queryEngagementResult(const Games & games, const Rounds & rounds, const Ticks & ticks,
                                       const PlayerAtTick & playerAtTick, const Hurt & hurt) {

    int numThreads = omp_get_max_threads();
    vector<int64_t> tmpRoundIds[numThreads];
    vector<int64_t> tmpRoundStarts[numThreads];
    vector<int64_t> tmpRoundSizes[numThreads];
    vector<int64_t> tmpStartTickId[numThreads];
    vector<int64_t> tmpEndTickId[numThreads];
    vector<int64_t> tmpFirstHurtTickId[numThreads];
    vector<int64_t> tmpLastHurtTickId[numThreads];
    vector<int64_t> tmpLength[numThreads];
    vector<vector<int64_t>> tmpPlayerId[numThreads];
    vector<vector<EngagementRole>> tmpRole[numThreads];
    vector<int16_t> tmpNumHits[numThreads];
    std::atomic<int64_t> roundsProcessed = 0;

    // for each round
    // track events for each pairs of player.
    // start a new event for a pair when hurt event with no prior one or far away prior one
    // clear out all hurt events on end of round
//#pragma omp parallel for
    for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
        int threadNum = omp_get_thread_num();
        tmpRoundIds[threadNum].push_back(roundIndex);
        tmpRoundStarts[threadNum].push_back(tmpStartTickId[threadNum].size());

        TickRates tickRates = computeTickRates(games, rounds, roundIndex);

        map<EngagementPlayers, EngagementData> curEngagements;

        for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
             tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {
            for (const auto & [_0, _1, hurtIndex] :
                ticks.hurtPerTick.findOverlapping(tickIndex, tickIndex)) {
                EngagementPlayers curPair{hurt.attacker[hurtIndex], hurt.victim[hurtIndex]};

                // start new engagement if none present
                if (curEngagements.find(curPair) == curEngagements.end()) {
                    curEngagements[curPair] = {tickIndex, tickIndex, 1};
                }
                else {
                    EngagementData & eData = curEngagements[curPair];
                    // if current engagement hasn't ended, extend it
                    // since new event will get PRE_ENGAGEMENT_SECONDS buffer and old event will get
                    // POST_ENGAGEMENT_BUFFER, must consider entire region as restart time to prevent
                    // overlapping event
                    if (secondsBetweenTicks(ticks, tickRates, eData.endTick, tickIndex)
                        <= PRE_ENGAGEMENT_SECONDS + POST_ENGAGEMENT_SECONDS) {
                        eData.endTick = tickIndex;
                        eData.numHits++;
                    }
                    // if current engagement ended, finish it and start new one
                    else {
                        finishEngagement(rounds, ticks, playerAtTick, tmpStartTickId, tmpEndTickId,
                                         tmpFirstHurtTickId, tmpLastHurtTickId, tmpLength,
                                         tmpPlayerId, tmpRole, tmpNumHits, threadNum, tickRates,
                                         curPair, eData);
                        eData = {tickIndex, tickIndex, 1};
                    }
                }
            }
        }

        // at end of round, clear all engagements
        for (const auto engagement : curEngagements) {
            finishEngagement(rounds, ticks, playerAtTick, tmpStartTickId, tmpEndTickId,
                             tmpFirstHurtTickId, tmpLastHurtTickId, tmpLength,
                             tmpPlayerId, tmpRole, tmpNumHits, threadNum, tickRates,
                             engagement.first, engagement.second);
        }

        tmpRoundSizes[threadNum].push_back(tmpStartTickId[threadNum].size() - tmpRoundStarts[threadNum].back());
    }

    EngagementResult result;
    mergeThreadResults(numThreads, result.rowIndicesPerRound, tmpRoundIds, tmpRoundStarts, tmpRoundSizes,
                       result.startTickId, result.size,
                       [&](int64_t minThreadId, int64_t tmpRowId) {
                           result.startTickId.push_back(tmpStartTickId[minThreadId][tmpRowId]);
                           result.endTickId.push_back(tmpEndTickId[minThreadId][tmpRowId]);
                           result.firstHurtTickId.push_back(tmpFirstHurtTickId[minThreadId][tmpRowId]);
                           result.lastHurtTickId.push_back(tmpLastHurtTickId[minThreadId][tmpRowId]);
                           result.tickLength.push_back(tmpLength[minThreadId][tmpRowId]);
                           result.playerId.push_back(tmpPlayerId[minThreadId][tmpRowId]);
                           result.role.push_back(tmpRole[minThreadId][tmpRowId]);
                           result.numHits.push_back(tmpNumHits[minThreadId][tmpRowId]);
                       });
    return result;
}
