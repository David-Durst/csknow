//
// Created by durst on 11/15/22.
//
#include "queries/moments/fire_history.h"
#include "queries/rolling_window.h"
#include <omp.h>


namespace csknow::fire_history {
    void FireHistoryResult::runQuery(const Games & games, const WeaponFire &weaponFire,
                                     const PlayerAtTick &playerAtTick) {
        tickId.resize(playerAtTick.size, INVALID_ID);
        playerId.resize(playerAtTick.size, INVALID_ID);
        holdingAttackButton.resize(playerAtTick.size, INVALID_ID);
        ticksSinceLastFire.resize(playerAtTick.size, INVALID_ID);
        ticksSinceLastHoldingAttack.resize(playerAtTick.size, INVALID_ID);
        ticksUntilNextFire.resize(playerAtTick.size, INVALID_ID);
        ticksUntilNextHoldingAttack.resize(playerAtTick.size, INVALID_ID);

        // for each round
        // track fire state for each player
        // ok to read last value if player popped into existance as rolling window fills in old values (see readTick)
//#pragma omp parallel for
        for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
            TickRates tickRates = computeTickRates(games, rounds, roundIndex);
            RollingWindow rollingWindow(rounds, ticks, playerAtTick);
            // need cur tick and prior tick
            rollingWindow.setTemporalRange(rounds.ticksPerRound[roundIndex].minId + 1, tickRates,
                                           {DurationType::Ticks, 0, 0, 1, 0});
            const PlayerToPATWindows & playerToPatWindows = rollingWindow.getWindows();

            map<int64_t, int64_t> playerToLastFireTickId, playerToLastHoldingAttackTickId;
            // if player hasn't fired in round present, assume the last first tick is first tick in round
            int64_t minTickIdFromWindowing = rollingWindow.lastReadTickId();

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
                    tickId[patIndex] = tickIndex;
                    playerId[patIndex] = curPlayerId;

                    // if no fire yet this round, set to max value
                    if (playerToLastFireTickId.find(curPlayerId) == playerToLastFireTickId.end()) {
                        ticksSinceLastFire[patIndex] = std::numeric_limits<int64_t>::max();
                    }
                    else {
                        ticksSinceLastFire[patIndex] = ticks.gameTickNumber[tickIndex] -
                            ticks.gameTickNumber[playerToLastFireTickId[curPlayerId]];
                    }


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
                    holdingAttackButton[patIndex] = holdingAttack;
                    if (holdingAttack) {
                        playerToLastHoldingAttackTickId[curPlayerId] = tickIndex;
                    }
                    // if no holding attack yet this round, set to max value
                    if (playerToLastHoldingAttackTickId.find(curPlayerId) == playerToLastHoldingAttackTickId.end()) {
                        ticksSinceLastHoldingAttack[patIndex] = std::numeric_limits<int64_t>::max();
                    }
                    else {
                        ticksSinceLastHoldingAttack[patIndex] = ticks.gameTickNumber[tickIndex] -
                            ticks.gameTickNumber[playerToLastHoldingAttackTickId[curPlayerId]];
                    }
                }

            }

            // loop 2: go backwards to compute future
            const int64_t defaultLastTickId = rounds.ticksPerRound[roundIndex].maxId;
            map<int64_t, int64_t> playerToNextFireTickId, playerToNextHoldingAttackTickId;
            for (int64_t patIndex = ticks.patPerTick[defaultLastTickId].maxId;
                 patIndex != -1 && patIndex >= ticks.patPerTick[minTickIdFromWindowing].minId; patIndex--) {
                const int64_t & curTickId = playerAtTick.tickId[patIndex];
                const int64_t & curPlayerId = playerAtTick.playerId[patIndex];

                // add player to playerToLastFireTickId and playerToHoldingAttackTickId if not present
                if (playerToNextFireTickId.find(curPlayerId) == playerToNextFireTickId.end()) {
                    playerToNextFireTickId[curPlayerId] = defaultLastTickId;
                }
                if (playerToNextHoldingAttackTickId.find(curPlayerId) == playerToNextHoldingAttackTickId.end()) {
                    playerToNextHoldingAttackTickId[curPlayerId] = defaultLastTickId;
                }

                // update fire/holding if currently doing it
                if (ticksSinceLastFire[patIndex] == 0) {
                    playerToNextFireTickId[curPlayerId] = curTickId;
                }
                if (ticksSinceLastHoldingAttack[patIndex] == 0) {
                    playerToNextHoldingAttackTickId[curPlayerId] = curTickId;
                }

                // if no holding attack yet this round, set to max value
                if (playerToNextFireTickId.find(curPlayerId) == playerToNextFireTickId.end()) {
                    ticksUntilNextFire[patIndex] = std::numeric_limits<int64_t>::max();
                }
                else {
                    ticksUntilNextFire[patIndex] = ticks.gameTickNumber[playerToNextFireTickId[curPlayerId]] -
                        ticks.gameTickNumber[curTickId];
                }
                if (playerToNextHoldingAttackTickId.find(curPlayerId) == playerToNextHoldingAttackTickId.end()) {
                    ticksUntilNextHoldingAttack[patIndex] = std::numeric_limits<int64_t>::max();
                }
                else {
                    ticksUntilNextHoldingAttack[patIndex] = ticks.gameTickNumber[playerToNextHoldingAttackTickId[curPlayerId]] -
                        ticks.gameTickNumber[curTickId];
                }
            }
        }
        size = playerAtTick.size;
    }
}
