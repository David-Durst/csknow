//
// Created by durst on 9/11/22.
//

#include "queries/moments/engagement.h"
#include "queries/lookback.h"
#include "indices/build_indexes.h"
#include <omp.h>
#include "queries/parser_constants.h"

struct EngagementPlayers {
    int64_t attacker, victim;

    bool operator<(const EngagementPlayers & other) const {
        return attacker < other.attacker ||
               (attacker == other.attacker && victim < other.victim);
    }
};

struct EngagementData {
    int64_t startTick = INVALID_ID, endTick = INVALID_ID;
    vector<int64_t> hurtTickIds, hurtIds;
};

void finishEngagement(const Rounds &rounds, const Ticks &ticks,
                      vector<vector<int64_t>> & tmpStartTickId, vector<vector<int64_t>> & tmpEndTickId,
                      vector<vector<int64_t>> & tmpLength, vector<vector<vector<int64_t>>> & tmpPlayerId,
                      vector<vector<vector<EngagementRole>>> & tmpRole,
                      vector<vector<vector<int64_t>>> & tmpHurtTickIds, vector<vector<vector<int64_t>>> & tmpHurtIds,
                      int threadNum, const TickRates &tickRates,
                      const EngagementPlayers &curPair, const EngagementData &eData) {
    // use pre and post periods to track behavior around engagement
    int64_t preEngagementStart = getLookbackDemoTick(rounds, ticks,
                                                     eData.startTick, tickRates,
                                                     PRE_ENGAGEMENT_SECONDS);
    int64_t postEngagementEnd = getLookforwardDemoTick(rounds, ticks,
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

EngagementResult queryEngagementResult(const Games & games, const Rounds & rounds, const Ticks & ticks, const Hurt & hurt) {

    int numThreads = omp_get_max_threads();
    vector<vector<int64_t>> tmpRoundIds(numThreads);
    vector<vector<int64_t>> tmpRoundStarts(numThreads);
    vector<vector<int64_t>> tmpRoundSizes(numThreads);
    vector<vector<int64_t>> tmpStartTickId(numThreads);
    vector<vector<int64_t>> tmpEndTickId(numThreads);
    vector<vector<int64_t>> tmpLength(numThreads);
    vector<vector<vector<int64_t>>> tmpPlayerId(numThreads);
    vector<vector<vector<EngagementRole>>> tmpRole(numThreads);
    vector<vector<vector<int64_t>>> tmpHurtTickIds(numThreads);
    vector<vector<vector<int64_t>>> tmpHurtIds(numThreads);

    // for each round
    // track events for each pairs of player.
    // start a new event for a pair when hurt event with no prior one or far away prior one
    // clear out all hurt events on end of round
//#pragma omp parallel for
    for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
        int threadNum = omp_get_thread_num();
        tmpRoundIds[threadNum].push_back(roundIndex);
        tmpRoundStarts[threadNum].push_back(static_cast<int64_t>(tmpStartTickId[threadNum].size()));

        TickRates tickRates = computeTickRates(games, rounds, roundIndex);

        map<EngagementPlayers, EngagementData> curEngagements;

        for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
             tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {
            for (const auto & [_0, _1, hurtIndex] :
                ticks.hurtPerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex)) {
                if (!isDemoEquipmentAGun(hurt.weapon[hurtIndex])) {
                    continue;
                }
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
                        finishEngagement(rounds, ticks, tmpStartTickId, tmpEndTickId,
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
        for (const auto & engagement : curEngagements) {
            finishEngagement(rounds, ticks, tmpStartTickId, tmpEndTickId,
                             tmpLength, tmpPlayerId, tmpRole,
                             tmpHurtTickIds, tmpHurtIds, threadNum, tickRates,
                             engagement.first, engagement.second);
        }

        tmpRoundSizes[threadNum].push_back(static_cast<int64_t>(tmpStartTickId[threadNum].size()) - tmpRoundStarts[threadNum].back());
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
    vector<std::reference_wrapper<const vector<int64_t>>> foreignKeyCols{result.startTickId, result.endTickId};
    result.engagementsPerTick = buildIntervalIndex(foreignKeyCols, result.size);
    return result;
}

void EngagementResult::computePercentMatchNearestCrosshair(const Rounds & rounds, const Ticks & ticks,
                                                           const PlayerAtTick & playerAtTick,
                                                           const csknow::feature_store::FeatureStoreResult & featureStoreResult) {
    map<int64_t, int64_t> engagementToNumCorrectTicksCurTick, engagementToNumCorrectTicks500ms, engagementToNumCorrectTicks1s, engagementToNumCorrectTicks2s;
    for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
        for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
             tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {
            map<int64_t, int64_t> playerToVictimCurTick;
            map<int64_t, int64_t> playerToEngagementCurTick;
            for (const auto & [_0, _1, engagementIndex] :
                engagementsPerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex)) {
                playerToVictimCurTick[playerId[engagementIndex][0]] = playerId[engagementIndex][1];
                playerToEngagementCurTick[playerId[engagementIndex][0]] = engagementIndex;
                if (engagementToNumCorrectTicks2s.find(engagementIndex) == engagementToNumCorrectTicks2s.end()) {
                    engagementToNumCorrectTicksCurTick[engagementIndex] = 0;
                    engagementToNumCorrectTicks500ms[engagementIndex] = 0;
                    engagementToNumCorrectTicks1s[engagementIndex] = 0;
                    engagementToNumCorrectTicks2s[engagementIndex] = 0;
                }
            }
            for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
                 patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
                int64_t attackerId = playerAtTick.playerId[patIndex];
                if (playerToVictimCurTick.find(attackerId) != playerToVictimCurTick.end()) {
                    /*
                    if (ticks.demoTickNumber[tickIndex] == 5117 && attackerId == 2) {
                        std::cout << "tick index " << tickIndex << " attacker id " << attackerId
                            << " victim id " << playerToVictimCurTick[attackerId] << " nearest crosshair enemy 2s "
                            << featureStoreResult.nearestCrosshairEnemy2s[patIndex] << std::endl;
                    }
                     */
                    int nearestEnemyIndexCurTick = featureStoreResult.nearestCrosshairCurTick[patIndex];
                    if (featureStoreResult.columnEnemyData[nearestEnemyIndexCurTick].playerId[patIndex] == playerToVictimCurTick[attackerId]) {
                        engagementToNumCorrectTicksCurTick[playerToEngagementCurTick[attackerId]]++;
                    }
                    int nearestEnemyIndex500ms = featureStoreResult.nearestCrosshairEnemy500ms[patIndex];
                    if (featureStoreResult.columnEnemyData[nearestEnemyIndex500ms].playerId[patIndex] == playerToVictimCurTick[attackerId]) {
                        engagementToNumCorrectTicks500ms[playerToEngagementCurTick[attackerId]]++;
                    }
                    int nearestEnemyIndex1s = featureStoreResult.nearestCrosshairEnemy1s[patIndex];
                    if (featureStoreResult.columnEnemyData[nearestEnemyIndex1s].playerId[patIndex] == playerToVictimCurTick[attackerId]) {
                        engagementToNumCorrectTicks1s[playerToEngagementCurTick[attackerId]]++;
                    }
                    int nearestEnemyIndex2s = featureStoreResult.nearestCrosshairEnemy2s[patIndex];
                    if (featureStoreResult.columnEnemyData[nearestEnemyIndex2s].playerId[patIndex] == playerToVictimCurTick[attackerId]) {
                        engagementToNumCorrectTicks2s[playerToEngagementCurTick[attackerId]]++;
                    }
                }
            }
        }
    }
    for (int64_t engagementIndex = 0; engagementIndex < size; engagementIndex++) {
        percentMatchNearestCrosshairEnemyCurTick.push_back(
            static_cast<double>(engagementToNumCorrectTicksCurTick[engagementIndex]) /
            static_cast<double>(tickLength[engagementIndex]));
        percentMatchNearestCrosshairEnemy500ms.push_back(
            static_cast<double>(engagementToNumCorrectTicks500ms[engagementIndex]) /
            static_cast<double>(tickLength[engagementIndex]));
        percentMatchNearestCrosshairEnemy1s.push_back(
            static_cast<double>(engagementToNumCorrectTicks1s[engagementIndex]) /
            static_cast<double>(tickLength[engagementIndex]));
        percentMatchNearestCrosshairEnemy2s.push_back(
            static_cast<double>(engagementToNumCorrectTicks2s[engagementIndex]) /
            static_cast<double>(tickLength[engagementIndex]));
    }
}