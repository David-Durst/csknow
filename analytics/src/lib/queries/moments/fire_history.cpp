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
        vector<vector<int64_t>> tmpHoldingAttackButton(numThreads);
        vector<vector<int64_t>> tmpTicksSinceLastFire(numThreads);
        vector<vector<int64_t>> tmpTicksSinceLastHoldingAttack(numThreads);
        vector<vector<int64_t>> tmpTicksUntilNextFire(numThreads);
        vector<vector<int64_t>> tmpTicksUntilNextHoldingAttack(numThreads);

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

            map<int64_t, int64_t> playerToLastFireTickId, playerToLastHoldingAttackTickId;
            // if player hasn't fired in round present, assume the last first tick is first tick in round
            const int64_t defaultFirstTickId = rounds.ticksPerRound[roundIndex].minId;

            size_t threadMinIndex = tmpTickId[threadNum].size();
            // loop 1: compute past
            for (int64_t windowEndTickIndex = rollingWindow.lastReadTickId();
                 windowEndTickIndex <= rounds.ticksPerRound[roundIndex].maxId; windowEndTickIndex = rollingWindow.readNextTick()) {
                int64_t tickIndex = rollingWindow.lastCurTickId();

                for (const auto & [_0, _1, fireIndex] :
                    ticks.weaponFirePerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex)) {
                    playerToLastFireTickId[weaponFire.shooter[fireIndex]] = tickIndex;
                }

                for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
                     patIndex != -1 && patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
                    const int64_t & curPlayerId = playerAtTick.playerId[patIndex];
                    const int64_t & priorPATIndex = playerToPatWindows.at(curPlayerId).fromOldest(0UL);
                    tmpTickId[threadNum].push_back(tickIndex);
                    tmpPlayerId[threadNum].push_back(curPlayerId);

                    // add player to playerToLastFireTickId and playerToHoldingAttackTickId if not present
                    if (playerToLastFireTickId.find(curPlayerId) == playerToLastFireTickId.end()) {
                        playerToLastFireTickId[curPlayerId] = defaultFirstTickId;
                    }
                    if (playerToLastHoldingAttackTickId.find(curPlayerId) == playerToLastHoldingAttackTickId.end()) {
                        playerToLastHoldingAttackTickId[curPlayerId] = defaultFirstTickId;
                    }

                    tmpTicksSinceLastFire[threadNum].push_back(ticks.gameTickNumber[tickIndex] -
                        ticks.gameTickNumber[playerToLastFireTickId[curPlayerId]]);

                    // holding attack if not reloading and recoil index going up or holding constant and greater than 0.5
                    // recoil index increases by 1.0 increments, so can't be 0.5 and increasing
                    // (technically could check closer to 0, but why push limits?)
                    // recoil index resets to 0 on weapon switch, so anything that isn't 0 and isnt reloading
                    // results from firing
                    // recoil index is constant in between shots, only goes down over time
                    double curRecoilIndex = playerAtTick.recoilIndex[patIndex];
                    double priorRecoilIndex = playerAtTick.recoilIndex[priorPATIndex];
                    bool isReloading = playerAtTick.isReloading[patIndex];
                    bool holdingAttack = !isReloading && curRecoilIndex > 0.5 && curRecoilIndex >= priorRecoilIndex;
                    tmpHoldingAttackButton[threadNum].push_back(holdingAttack);
                    if (holdingAttack) {
                        playerToLastHoldingAttackTickId[curPlayerId] = tickIndex;
                    }
                    tmpTicksSinceLastHoldingAttack[threadNum].push_back(ticks.gameTickNumber[tickIndex] -
                        ticks.gameTickNumber[playerToLastHoldingAttackTickId[curPlayerId]]);
                }

            }
            size_t threadMaxIndex = tmpTickId[threadNum].size();

            // loop 2: go backwards to compute future
            const int64_t defaultLastTickId = rounds.ticksPerRound[roundIndex].maxId;
            map<int64_t, int64_t> playerToNextFireTickId, playerToNextHoldingAttackTickId;
            for (int64_t threadIndex = threadMaxIndex; threadIndex >= threadMinIndex; threadIndex--) {
                const int64_t & curTickId = tmpTickId[threadNum][threadIndex];
                const int64_t & curPlayerId = tmpPlayerId[threadNum][threadIndex];

                // add player to playerToLastFireTickId and playerToHoldingAttackTickId if not present
                if (playerToNextFireTickId.find(curPlayerId) == playerToNextFireTickId.end()) {
                    playerToNextFireTickId[curPlayerId] = defaultLastTickId;
                }
                if (playerToNextHoldingAttackTickId.find(curPlayerId) == playerToNextHoldingAttackTickId.end()) {
                    playerToNextHoldingAttackTickId[curPlayerId] = defaultLastTickId;
                }

                // update fire/holding if currently doing it
                if (tmpTicksSinceLastFire[threadNum][threadIndex] == 0) {
                    playerToNextFireTickId[curPlayerId] = curTickId;
                }
                if (tmpHoldingAttackButton[threadNum][threadIndex] == true) {
                    playerToNextHoldingAttackTickId[curPlayerId] = curTickId;
                }

                // record ticks until event happens
                tmpTicksUntilNextFire[threadNum].push_back(ticks.gameTickNumber[playerToNextFireTickId[curPlayerId]] -
                    ticks.gameTickNumber[curTickId]);
                tmpTicksUntilNextHoldingAttack[threadNum].push_back(ticks.gameTickNumber[curTickId] -
                    ticks.gameTickNumber[playerToNextHoldingAttackTickId[curPlayerId]]);
            }

            tmpRoundSizes[threadNum].push_back(static_cast<int64_t>(tmpTickId[threadNum].size()) - tmpRoundStarts[threadNum].back());
        }

        mergeThreadResults(numThreads, rowIndicesPerRound, tmpRoundIds, tmpRoundStarts, tmpRoundSizes,
                           tickId, size,
                           [&](int64_t minThreadId, int64_t tmpRowId) {
                               tickId.push_back(tmpTickId[minThreadId][tmpRowId]);
                               playerId.push_back(tmpPlayerId[minThreadId][tmpRowId]);
                               holdingAttackButton.push_back(tmpHoldingAttackButton[minThreadId][tmpRowId]);
                               ticksSinceLastFire.push_back(tmpTicksSinceLastFire[minThreadId][tmpRowId]);
                               ticksSinceLastHoldingAttack.push_back(tmpTicksSinceLastHoldingAttack[minThreadId][tmpRowId]);
                               ticksUntilNextFire.push_back(tmpTicksUntilNextFire[minThreadId][tmpRowId]);
                               ticksUntilNextHoldingAttack.push_back(tmpTicksUntilNextHoldingAttack[minThreadId][tmpRowId]);
        });
    }
}
