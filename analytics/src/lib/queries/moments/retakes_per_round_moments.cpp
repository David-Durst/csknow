//
// Created by durst on 4/29/23.
//

#include "queries/moments/retakes_per_round_moments.h"

namespace csknow::retakes_moments {
    struct RetakeInternalPlayerState {
        TeamId teamId = ENGINE_TEAM_UNASSIGNED;
        Vec3 plantPos = {0., 0., 0.};
        Vec3 lastPos = {0., 0., 0.};
        double distanceCovered = 0.;
        double farthestDistanceFromPlantPos = 0.;
        size_t numShots = 0;
        size_t numKills = 0;
        double sumSpeedDuringShooting = 0.;
        int64_t lastTickAlive = INVALID_ID;
    };

    void setBotTypes(RetakeBotType & ctBotType, RetakeBotType & tBotType, string demoName) {
        if (demoName.find("r_1") == 0) {
            ctBotType = RetakeBotType::CSKnowLearned;
            tBotType = RetakeBotType::CSKnowLearned;
        }
        else if (demoName.find("r_ct") == 0) {
            ctBotType = RetakeBotType::CSKnowLearned;
            tBotType = RetakeBotType::CSGODefault;
        }
        else if (demoName.find("r_t") == 0) {
            ctBotType = RetakeBotType::CSGODefault;
            tBotType = RetakeBotType::CSKnowLearned;
        }
        else if (demoName.find("r_0") == 0) {
            ctBotType = RetakeBotType::CSGODefault;
            tBotType = RetakeBotType::CSGODefault;
        }
        else if (demoName.find("rh_1") == 0) {
            ctBotType = RetakeBotType::CSKnowHeuristic;
            tBotType = RetakeBotType::CSKnowHeuristic;
        }
        else if (demoName.find("rh_ct") == 0) {
            ctBotType = RetakeBotType::CSKnowHeuristic;
            tBotType = RetakeBotType::CSGODefault;
        }
        else if (demoName.find("rh_t") == 0) {
            ctBotType = RetakeBotType::CSGODefault;
            tBotType = RetakeBotType::CSKnowHeuristic;
        }
        else if (demoName.find("rhct_1") == 0) {
            ctBotType = RetakeBotType::CSKnowHeuristic;
            tBotType = RetakeBotType::CSKnowLearned;
        }
        else if (demoName.find("rht_1") == 0) {
            ctBotType = RetakeBotType::CSKnowLearned;
            tBotType = RetakeBotType::CSKnowHeuristic;
        }
        else {
            ctBotType = RetakeBotType::Human;
            tBotType = RetakeBotType::Human;
        }

    }

    void RetakesPerRoundMoments::runQuery(const Games & games, const Rounds & rounds, const Ticks & ticks,
                                          const PlayerAtTick & playerAtTick,
                                          const WeaponFire & weaponFire, const Kills & kills,
                                          const Plants & plants, const Defusals & defusals,
                                          const csknow::round_extractor::ExtractValidBotRetakesRounds & extractValidBotRetakesRounds) {
        size_t numRounds = extractValidBotRetakesRounds.validRoundIds.size();
        plantTickId.resize(numRounds);
        roundEndTickId.resize(numRounds);
        tickLength.resize(numRounds);
        roundId.resize(numRounds);
        plantId.resize(numRounds, INVALID_ID);
        defusalId.resize(numRounds, INVALID_ID);

        ctMoments.win.resize(numRounds);
        ctMoments.distanceTraveledPerPlayer.resize(numRounds, 0.);
        ctMoments.maxDistanceFromStart.resize(numRounds, 0.);
        ctMoments.shotsPerKill.resize(numRounds, 0);
        ctMoments.averageSpeedWhileShooting.resize(numRounds, 0.);
        ctMoments.numPlayersAliveTickBeforeExplosion.resize(numRounds, 0.);
        ctMoments.botType.resize(numRounds);
        ctMoments.numPlayers.resize(numRounds, 0);

        tMoments.win.resize(numRounds);
        tMoments.distanceTraveledPerPlayer.resize(numRounds, 0.);
        tMoments.maxDistanceFromStart.resize(numRounds, 0.);
        tMoments.shotsPerKill.resize(numRounds, 0.);
        tMoments.averageSpeedWhileShooting.resize(numRounds, 0.);
        tMoments.numPlayersAliveTickBeforeExplosion.resize(numRounds, 0.);
        tMoments.botType.resize(numRounds);
        tMoments.numPlayers.resize(numRounds, 0);

//#pragma omp parallel for
        for (size_t validRoundIndex = 0; validRoundIndex < extractValidBotRetakesRounds.validRoundIds.size();
             validRoundIndex++) {
            //int threadNum = omp_get_thread_num();
            int64_t roundIndex = extractValidBotRetakesRounds.validRoundIds[validRoundIndex];
            roundId[validRoundIndex] = roundIndex;
            roundEndTickId[validRoundIndex] = rounds.endTick[roundIndex];
            int64_t gameId = rounds.gameId[roundIndex];
            setBotTypes(ctMoments.botType[validRoundIndex], tMoments.botType[validRoundIndex],
                        games.demoFile[gameId]);

            if (rounds.winner[roundIndex] == ENGINE_TEAM_CT) {
                ctMoments.win[validRoundIndex] = true;
                tMoments.win[validRoundIndex] = false;
            }
            else {
                ctMoments.win[validRoundIndex] = false;
                tMoments.win[validRoundIndex] = true;
            }

            map<int64_t, RetakeInternalPlayerState> playerToRetakeState;
            int64_t explosionTickId = INVALID_ID;

            bool foundFirstPlantInRound = false, foundFirstDefusalInRound;
            for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
                 tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {
                bool curTickIsFirstPlant = false, curTickIsFirstDefusal = false;
                for (const auto &[_0, _1, plantIndex]:
                    ticks.plantsEndPerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex)) {
                    if (plants.succesful[plantIndex] && !foundFirstPlantInRound) {
                        plantId[validRoundIndex] = plantIndex;
                        curTickIsFirstPlant = true;
                    }
                }

                for (const auto & [_0, _1, defusalIndex] :
                    ticks.defusalsEndPerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex)) {
                    if (defusals.succesful[defusalIndex] && !foundFirstDefusalInRound) {
                        defusalId[validRoundIndex] = defusalIndex;
                        curTickIsFirstDefusal = true;
                    }
                }

                for (const auto & [_0, _1, explosionIndex] :
                    ticks.explosionsPerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex)) {
                    explosionTickId = tickIndex;
                }

                set<int64_t> playersFiringThisTick;
                for (const auto & [_0, _1, weaponFireIndex] :
                    ticks.weaponFirePerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex)) {
                    if (isDemoEquipmentAGun(static_cast<DemoEquipmentType>(weaponFire.weapon[weaponFireIndex]))) {
                        playersFiringThisTick.insert(weaponFire.shooter[weaponFireIndex]);
                    }
                }

                set<int64_t> playersKillingThisTick;
                for (const auto & [_0, _1, killIndex] :
                    ticks.killsPerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex)) {
                    playersFiringThisTick.insert(kills.killer[killIndex]);
                }

                if (curTickIsFirstPlant) {
                    foundFirstPlantInRound = true;
                }
                if (curTickIsFirstDefusal) {
                    foundFirstDefusalInRound = true;
                }

                if (foundFirstPlantInRound) {
                    for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
                         patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
                        if (playerAtTick.isAlive[patIndex]) {
                            int64_t playerId = playerAtTick.playerId[patIndex];
                            Vec3 curPos = {
                                playerAtTick.posX[patIndex],
                                playerAtTick.posY[patIndex],
                                playerAtTick.posZ[patIndex]
                            };
                            if (playerToRetakeState.find(playerId) == playerToRetakeState.end()) {
                                playerToRetakeState[playerId] = {
                                    playerAtTick.team[patIndex],
                                    curPos, curPos,
                                    0., 0., 0, 0, 0.,
                                    tickIndex
                                };
                            }
                            double curDistanceFromPlantPos =
                                computeDistance(curPos, playerToRetakeState[playerId].plantPos);
                            RetakeInternalPlayerState & playerRetakeState = playerToRetakeState[playerId];
                            playerRetakeState.distanceCovered +=
                                std::abs(computeDistance(curPos, playerRetakeState.lastPos));
                            playerRetakeState.farthestDistanceFromPlantPos =
                                std::max(playerRetakeState.farthestDistanceFromPlantPos, curDistanceFromPlantPos);
                            if (playersFiringThisTick.find(playerId) != playersFiringThisTick.end()) {
                                playerRetakeState.numShots++;
                                playerRetakeState.sumSpeedDuringShooting += std::abs(computeMagnitude({
                                    playerAtTick.velX[patIndex],
                                    playerAtTick.velY[patIndex],
                                    playerAtTick.velZ[patIndex]
                                }));
                            }
                            if (playersKillingThisTick.find(playerId) != playersKillingThisTick.end()) {
                                playerRetakeState.numKills++;
                            }
                            playerRetakeState.lastTickAlive = tickIndex;
                            playerRetakeState.lastPos = curPos;
                        }
                    }
                }
            }

            // compute overall team stats per round
            int ctKills = 0, tKills, ctShots = 0, tShots = 0;
            for (const auto & [_, playerRetakeState] : playerToRetakeState) {
                if (playerRetakeState.teamId == ENGINE_TEAM_CT) {
                    ctKills += playerRetakeState.numKills;
                    ctShots += playerRetakeState.numShots;
                    ctMoments.distanceTraveledPerPlayer[validRoundIndex] += playerRetakeState.distanceCovered;
                    ctMoments.maxDistanceFromStart[validRoundIndex] = std::max(
                        ctMoments.maxDistanceFromStart[validRoundIndex],
                        playerRetakeState.farthestDistanceFromPlantPos);
                    ctMoments.shotsPerKill[validRoundIndex] += playerRetakeState.numShots;
                    ctMoments.averageSpeedWhileShooting[validRoundIndex] += playerRetakeState.sumSpeedDuringShooting;
                    if (explosionTickId != INVALID_ID && playerRetakeState.lastTickAlive + 1 >= explosionTickId) {
                        ctMoments.numPlayersAliveTickBeforeExplosion[validRoundIndex]++;
                    }
                    ctMoments.numPlayers[validRoundIndex]++;
                }
                else {
                    tKills += playerRetakeState.numKills;
                    tShots += playerRetakeState.numShots;
                    tMoments.distanceTraveledPerPlayer[validRoundIndex] += playerRetakeState.distanceCovered;
                    tMoments.maxDistanceFromStart[validRoundIndex] = std::max(
                        tMoments.maxDistanceFromStart[validRoundIndex],
                        playerRetakeState.farthestDistanceFromPlantPos);
                    tMoments.shotsPerKill[validRoundIndex] += playerRetakeState.numShots;
                    tMoments.averageSpeedWhileShooting[validRoundIndex] += playerRetakeState.sumSpeedDuringShooting;
                    if (explosionTickId != INVALID_ID && playerRetakeState.lastTickAlive + 1 >= explosionTickId) {
                        tMoments.numPlayersAliveTickBeforeExplosion[validRoundIndex]++;
                    }
                    tMoments.numPlayers[validRoundIndex]++;
                }
            }

            ctMoments.shotsPerKill[validRoundIndex] /= ctKills;
            ctMoments.averageSpeedWhileShooting[validRoundIndex] /= ctShots;
            tMoments.shotsPerKill[validRoundIndex] /= tKills;
            tMoments.averageSpeedWhileShooting[validRoundIndex] /= tShots;

            tickLength[validRoundIndex] = roundEndTickId[validRoundIndex] - plantTickId[validRoundIndex] + 1;
        }
    }
}
