//
// Created by durst on 11/15/22.
//
#include "queries/moments/fire_history.h"
#include "queries/rolling_window.h"
#include <omp.h>


namespace csknow::fire_history {
    void FireHistoryResult::runQuery(const Games & games, const WeaponFire &weaponFire,
                                     const PlayerAtTick &playerAtTick) {
        int numThreads = omp_get_max_threads();
        vector<vector<int64_t>> tmpRoundIds(numThreads);
        vector<vector<int64_t>> tmpRoundStarts(numThreads);
        vector<vector<int64_t>> tmpRoundSizes(numThreads);
        vector<vector<int64_t>> tmpTickId(numThreads);
        vector<vector<int64_t>> tmpPlayerId(numThreads);
        vector<vector<int64_t>> tmpTicksSinceLastFire(numThreads);
        vector<vector<int64_t>> tmpLastShotFiredTickId(numThreads);
        vector<vector<int64_t>> tmpTicksUntilNextFire(numThreads);
        vector<vector<int64_t>> tmpNextShotFiredTickId(numThreads);
        vector<vector<int64_t>> tmpHoldingAttackButton(numThreads);
        vector<vector<DemoEquipmentType>> tmpActiveWeaponType(numThreads);

        // for each round
        // track fire state for each player
        // ok to read last value if player popped into existance as rolling window fills in old values (see readTick)
//#pragma omp parallel for
        for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
            int threadNum = omp_get_thread_num();
            tmpRoundIds[threadNum].push_back(roundIndex);
            tmpRoundStarts[threadNum].push_back(static_cast<int64_t>(tmpTickId[threadNum].size()));

            TickRates tickRates = computeTickRates(games, rounds, roundIndex);
            RollingWindow rollingWindow(rounds, ticks, playerAtTick);
            // need cur tick and prior tick
            rollingWindow.setTemporalRange(rounds.ticksPerRound[roundIndex].minId + 1, tickRates,
                                           {DurationType::Ticks, 0, 0, 1, 0});
            const PlayerToPATWindows & playerToPatWindows = rollingWindow.getWindows();

            // loop 1: compute past
            for (int64_t windowEndTickIndex = rollingWindow.lastReadTickId();
                 windowEndTickIndex <= rounds.ticksPerRound[roundIndex].maxId; windowEndTickIndex = rollingWindow.readNextTick()) {
                int64_t tickIndex = rollingWindow.lastCurTickId();

                map<int64_t, int64_t> playerToFirePerTick;
                for (const auto & [_0, _1, fireIndex] :
                    ticks.weaponFirePerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex)) {
                    playerToFirePerTick[weaponFire.shooter[fireIndex]] = fireIndex;
                }

                for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
                     patIndex != -1 && patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
                    const int64_t & curPlayerId = playerAtTick.playerId[patIndex];
                    const int64_t & priorPATIndex = playerToPatWindows.at(curPlayerId).fromOldest(0UL);
                    tmpTickId[threadNum].push_back(tickIndex);
                    tmpPlayerId[threadNum].push_back(curPlayerId);

                    if (playerToFirePerTick.find(curPlayerId) != playerToFirePerTick.end()) {
                        tmpTicksSinceLastFire[threadNum].push_back(0);
                        tmpLastShotFiredTickId[threadNum].push_back(tickIndex);
                    }
                    else {
                        // need a way to look backwards in the tmp vector
                        tmpTicksSinceLastFire[threadNum].push_back();
                        tmpLastShotFiredTickId[threadNum].push_back(tickIndex);
                    }
                    //tmpTicksSinceLastFire
                }

            }
            // loop 2: go backwards to compute future
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

            tmpRoundSizes[threadNum].push_back(static_cast<int64_t>(tmpTickId[threadNum].size()) - tmpRoundStarts[threadNum].back());
        }

        mergeThreadResults(numThreads, rowIndicesPerRound, tmpRoundIds, tmpRoundStarts, tmpRoundSizes,
                           tickId, size,
                           [&](int64_t minThreadId, int64_t tmpRowId) {
                               tickId.push_back(tmpTickId[minThreadId][tmpRowId]);
                               playerId.push_back(tmpPlayerId[minThreadId][tmpRowId]);
                               ticksSinceLastFire.push_back(tmpTicksSinceLastFire[minThreadId][tmpRowId]);
                               lastShotFiredTickId.push_back(tmpLastShotFiredTickId[minThreadId][tmpRowId]);
                               ticksUntilNextFire.push_back(tmpTicksUntilNextFire[minThreadId][tmpRowId]);
                               nextShotFiredTickId.push_back(tmpNextShotFiredTickId[minThreadId][tmpRowId]);
                               holdingAttackButton.push_back(tmpHoldingAttackButton[minThreadId][tmpRowId]);
                               activeWeaponType.push_back(tmpActiveWeaponType[minThreadId][tmpRowId]);
        });
    }
}
