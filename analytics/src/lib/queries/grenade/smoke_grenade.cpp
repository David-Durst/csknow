//
// Created by durst on 2/26/23.
//

#include "queries/grenade/smoke_grenade.h"
#include <omp.h>

namespace csknow::smoke_grenade {
    void SmokeGrenadeResult::runQuery(const Rounds & rounds, const Ticks & ticks, const Grenades & grenades,
                                      const GrenadeTrajectories & grenadeTrajectories) {
        int numThreads = omp_get_max_threads();
        vector<vector<int64_t>> tmpRoundIds(numThreads);
        vector<vector<int64_t>> tmpRoundStarts(numThreads);
        vector<vector<int64_t>> tmpRoundSizes(numThreads);
        vector<vector<int64_t>> tmpTickId(numThreads);
        vector<vector<int64_t>> tmpThrowerId(numThreads);
        vector<vector<SmokeGrenadeState>> tmpState(numThreads);
        vector<vector<Vec3>> tmpPos(numThreads);

//#pragma omp parallel for
        for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
            int threadNum = omp_get_thread_num();
            tmpRoundIds[threadNum].push_back(roundIndex);
            tmpRoundStarts[threadNum].push_back(static_cast<int64_t>(tmpTickId[threadNum].size()));

            for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
                 tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {

                for (const auto & [_0, _1, grenadeIndex] :
                    ticks.grenadesPerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex)) {
                    tmpTickId[threadNum].push_back(tickIndex);
                    tmpThrowerId[threadNum].push_back(grenades.thrower[grenadeIndex]);
                    if (tickIndex < grenades.activeTick[grenadeIndex]) {
                        tmpState[threadNum].push_back(SmokeGrenadeState::Thrown);
                    }
                    else {
                        tmpState[threadNum].push_back(SmokeGrenadeState::Active);
                    }
                    tmpPos[threadNum].push_back()
                }
            }

            tmpRoundSizes[threadNum].push_back(static_cast<int64_t>(tmpTickId[threadNum].size()) - tmpRoundStarts[threadNum].back());
        }
        int64_t roundIndex = INVALID_ID;
        for (int64_t grenadeIndex = 0; grenadeIndex < grenades.size; grenadeIndex++) {
            int threadNum = omp_get_thread_num();
            int64_t throwTickIndex = grenades.throwTick[grenadeIndex];
            int64_t roundIndex = ticks.roundId[throwTickIndex];

            if (tmpRoundIds[threadNum].empty() || tmpRoundIds[threadNum].back() != roundIndex) {
                tmpRoundIds[threadNum].push_back(roundIndex);
                tmpRoundStarts[threadNum].push_back(static_cast<int64_t>(tmpStartTickId[threadNum].size()));
            }
            tmpStartTickId[threadNum].push_back(grenades.throwTick[grenadeIndex]);
            tmpEndTickId[threadNum].push_back(grenades.destroyTick[grenadeIndex]);
            tmpLength[threadNum].push_back(grenades.destroyTick[grenadeIndex] - grenades.throwTick[grenadeIndex]);
            tmpThrowTick[threadNum].push_back(grenades.throwTick[grenadeIndex]);
            tmpActiveTick[threadNum].push_back(grenades.activeTick[grenadeIndex]);
            tmpExpiredTick[threadNum].push_back(grenades.expiredTick[grenadeIndex]);
            tmpDestroyTick[threadNum].push_back(grenades.destroyTick[grenadeIndex]);
            tmpThrowerId[threadNum].push_back(grenades.thrower[grenadeIndex]);

            for (int64_t trajectoryIndex = grenades.trajectoryPerGrenade[grenadeIndex].minId;
                 trajectoryIndex <= grenades.trajectoryPerGrenade[grenadeIndex].maxId; trajectoryIndex++) {


            }

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
        vector<const int64_t *> foreignKeyCols{result.startTickId.data(), result.endTickId.data()};
        result.engagementsPerTick = buildIntervalIndex(foreignKeyCols, result.size);
        return result;
    }
}
