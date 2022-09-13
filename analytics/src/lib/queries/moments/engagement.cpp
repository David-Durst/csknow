//
// Created by durst on 9/11/22.
//

#include "queries/moments/engagement.h"
#include "queries/lookback.h"
#include "indices/build_indexes.h"
#include <omp.h>

struct EngagementPlayers {
    int64_t attacker, victim;

    bool operator<(const EngagementPlayers & other) const {
        return attacker < other.attacker ||
               (attacker == other.attacker && victim < other.victim);
    }
};

struct EngagementData {
    int64_t startTick, endTick;
    vector<int64_t> hurtTickIds, hurtIds;
};

void finishEngagement(const Rounds &rounds, const Ticks &ticks, const PlayerAtTick &playerAtTick,
                      vector<int64_t> tmpStartTickId[], vector<int64_t> tmpEndTickId[],
                      vector<int64_t> tmpLength[], vector<vector<int64_t>> tmpPlayerId[],
                      vector<vector<EngagementRole>> tmpRole[],
                      vector<vector<int64_t>> tmpHurtTickIds[], vector<vector<int64_t>> tmpHurtIds[],
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
    tmpLength[threadNum].push_back(postEngagementEnd - preEngagementStart + 1);
    tmpPlayerId[threadNum].push_back({curPair.attacker, curPair.victim});
    tmpRole[threadNum].push_back({EngagementRole::Attacker, EngagementRole::Victim});
    tmpHurtTickIds[threadNum].push_back(eData.hurtTickIds);
    tmpHurtIds[threadNum].push_back(eData.hurtIds);
}

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
    vector<vector<int64_t>> tmpHurtTickIds[numThreads];
    vector<vector<int64_t>> tmpHurtIds[numThreads];

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
                    curEngagements[curPair] = {tickIndex, tickIndex,
                                               {tickIndex}, {hurtIndex}};
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
                        eData.hurtTickIds.push_back(tickIndex);
                        eData.hurtIds.push_back(hurtIndex);
                    }
                    // if current engagement ended, finish it and start new one
                    else {
                        finishEngagement(rounds, ticks, playerAtTick, tmpStartTickId, tmpEndTickId,
                                         tmpLength, tmpPlayerId, tmpRole,
                                         tmpHurtTickIds, tmpHurtIds, threadNum, tickRates,
                                         curPair, eData);
                        eData = {tickIndex, tickIndex,
                                 {tickIndex}, {hurtIndex}};
                    }
                }
            }
        }

        // at end of round, clear all engagements
        for (const auto engagement : curEngagements) {
            finishEngagement(rounds, ticks, playerAtTick, tmpStartTickId, tmpEndTickId,
                             tmpLength, tmpPlayerId, tmpRole,
                             tmpHurtTickIds, tmpHurtIds, threadNum, tickRates,
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
                           result.tickLength.push_back(tmpLength[minThreadId][tmpRowId]);
                           result.playerId.push_back(tmpPlayerId[minThreadId][tmpRowId]);
                           result.role.push_back(tmpRole[minThreadId][tmpRowId]);
                           result.hurtTickIds.push_back(tmpHurtTickIds[minThreadId][tmpRowId]);
                           result.hurtIds.push_back(tmpHurtIds[minThreadId][tmpRowId]);
                       });
    vector<const int64_t *> foreignKeyCols{result.startTickId.data(), result.endTickId.data()};
    result.engagementsPerTick = buildIntervalIndex(foreignKeyCols, result.size);
    return result;
}
