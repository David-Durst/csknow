//
// Created by durst on 4/29/23.
//

#include "queries/moments/retakes_per_round_moments.h"
#include "signal.h"

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
        else if (demoName.find("rh_0") == 0) {
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
                                          const csknow::round_extractor::ExtractValidBotRetakesRounds & extractValidBotRetakesRounds,
                                          bool botData) {
        size_t numRounds = extractValidBotRetakesRounds.validRoundIds.size();
        gameId.resize(numRounds);
        plantSituationId.resize(numRounds);
        plantTickId.resize(numRounds);
        roundEndTickId.resize(numRounds);
        tickLength.resize(numRounds);
        roundId.resize(numRounds);
        plantId.resize(numRounds, INVALID_ID);
        defusalId.resize(numRounds, INVALID_ID);

        ctMoments.win.resize(numRounds);
        ctMoments.shotsPerTotalPlayers.resize(numRounds);
        ctMoments.killsPerTotalPlayers.resize(numRounds);
        ctMoments.distanceTraveledPerPlayer.resize(numRounds, 0.);
        ctMoments.maxDistanceFromStart.resize(numRounds, 0.);
        ctMoments.shotsPerKill.resize(numRounds, 0);
        ctMoments.averageSpeedWhileShooting.resize(numRounds, 0.);
        ctMoments.numPlayersAliveTickBeforeExplosion.resize(numRounds,
                                                            std::numeric_limits<double>::quiet_NaN());
        ctMoments.numPlayersAliveTickAfterExplosion.resize(numRounds,
                                                           std::numeric_limits<double>::quiet_NaN());
        ctMoments.botType.resize(numRounds);
        ctMoments.numPlayers.resize(numRounds, 0);

        tMoments.win.resize(numRounds);
        tMoments.shotsPerTotalPlayers.resize(numRounds);
        tMoments.killsPerTotalPlayers.resize(numRounds);
        tMoments.distanceTraveledPerPlayer.resize(numRounds, 0.);
        tMoments.maxDistanceFromStart.resize(numRounds, 0.);
        tMoments.shotsPerKill.resize(numRounds, 0.);
        tMoments.averageSpeedWhileShooting.resize(numRounds, 0.);
        tMoments.numPlayersAliveTickBeforeExplosion.resize(numRounds,
                                                           std::numeric_limits<double>::quiet_NaN());
        tMoments.numPlayersAliveTickAfterExplosion.resize(numRounds,
                                                          std::numeric_limits<double>::quiet_NaN());
        tMoments.botType.resize(numRounds);
        tMoments.numPlayers.resize(numRounds, 0);

//#pragma omp parallel for
        for (size_t validRoundIndex = 0; validRoundIndex < extractValidBotRetakesRounds.validRoundIds.size();
             validRoundIndex++) {
            plantSituationId[validRoundIndex] = extractValidBotRetakesRounds.plantIndex[validRoundIndex];
            //std::cout << "valid round index " << validRoundIndex << std::endl;
            //int threadNum = omp_get_thread_num();
            int64_t roundIndex = extractValidBotRetakesRounds.validRoundIds[validRoundIndex];
            roundId[validRoundIndex] = roundIndex;
            gameId[validRoundIndex] = rounds.gameId[roundIndex];
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

            if (ctMoments.botType[validRoundIndex] == RetakeBotType::CSKnowLearned &&
                tMoments.botType[validRoundIndex] == RetakeBotType::CSGODefault &&
                tMoments.win[validRoundIndex]) {
                std::cout << "t defualt beats ct learned " << games.demoFile[gameId] << " start game tick number " <<
                    ticks.gameTickNumber[rounds.ticksPerRound[roundIndex].minId];
            }

            map<int64_t, RetakeInternalPlayerState> playerToRetakeState;
            int64_t explosionTickId = INVALID_ID;

            bool foundFirstPlantInRound = false, foundFirstDefusalInRound;
            int64_t firstDeathByWorldTick = INVALID_ID;
            int64_t firstTeleportTickIndex = INVALID_ID;
            //int numTeleportTicks = 0;
            for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
                 tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {
                // can't use first death by world, not all rounds involve death by world
                // but they all involve a teleport - clear all positions if first time teleported
                // sometimes can teleport for next round before end of cur round
                //bool kills = false;
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

                for (const auto & unused :
                    ticks.explosionsPerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex)) {
                    (void) unused;
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
                    playersKillingThisTick.insert(kills.killer[killIndex]);
                    if (firstDeathByWorldTick == INVALID_ID &&
                        static_cast<DemoEquipmentType>(kills.weapon[killIndex]) == DemoEquipmentType::EqWorld) {
                        firstDeathByWorldTick = tickIndex;
                    }
                }

                if (curTickIsFirstPlant) {
                    foundFirstPlantInRound = true;
                }
                if (curTickIsFirstDefusal) {
                    foundFirstDefusalInRound = true;
                }

                /*
                bool pastSetupDeaths = firstDeathByWorldTick != INVALID_ID && firstDeathByWorldTick < tickIndex;

                if (firstDeathByWorldTick != INVALID_ID && tickIndex >= firstDeathByWorldTick &&
                    tickIndex < firstDeathByWorldTick + 2) {
                    std::cout << "demoName " << games.demoFile[gameId] << " tickIndex " << tickIndex << " firstDeathByWorldTick " << firstDeathByWorldTick <<
                        " c4 pos " << ticks.bombX[tickIndex] << "," << ticks.bombY[tickIndex] << "," << ticks.bombZ[tickIndex] << std::endl;
                }
                 */


                if (botData || foundFirstPlantInRound) {
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
                            double teleportDistance = computeDistance(playerRetakeState.lastPos, curPos);
                            if (tickIndex != rounds.startTick[roundIndex] && teleportDistance > 50. &&
                                computeMagnitude({playerAtTick.velX[patIndex], playerAtTick.velY[patIndex], playerAtTick.velZ[patIndex]}) == 0.) {
                                if (firstTeleportTickIndex == INVALID_ID) {
                                    firstTeleportTickIndex = tickIndex;
                                }
                                /*
                                if (!teleportTick) {
                                    numTeleportTicks++;
                                    std::cout << "teleport tick " << tickIndex << " game tick number " << ticks.gameTickNumber[tickIndex] << " distance " << teleportDistance << std::endl;
                                }
                                teleportTick = true;
                                 */
                            }
                            playerRetakeState.lastPos = curPos;
                        }
                    }
                }
                if (firstTeleportTickIndex != INVALID_ID && tickIndex >= firstTeleportTickIndex &&
                    tickIndex <= firstTeleportTickIndex + 3) {
                    playerToRetakeState.clear();
                }
            }
            /*
            if (numTeleportTicks != 1) {
                std::cout << "num teleport ticks: " << numTeleportTicks << std::endl;
            }
             */

            // compute overall team stats per round
            if (explosionTickId != INVALID_ID) {
                ctMoments.numPlayersAliveTickBeforeExplosion[validRoundIndex] = 0;
                ctMoments.numPlayersAliveTickAfterExplosion[validRoundIndex] = 0;
                tMoments.numPlayersAliveTickBeforeExplosion[validRoundIndex] = 0;
                tMoments.numPlayersAliveTickAfterExplosion[validRoundIndex] = 0;
            }
            double ctKills = 0., tKills = 0., ctShots = 0., tShots = 0.;
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
                    if (explosionTickId != INVALID_ID && playerRetakeState.lastTickAlive > explosionTickId) {
                        ctMoments.numPlayersAliveTickAfterExplosion[validRoundIndex]++;
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
                    if (explosionTickId != INVALID_ID && playerRetakeState.lastTickAlive > explosionTickId) {
                        tMoments.numPlayersAliveTickAfterExplosion[validRoundIndex]++;
                    }
                    tMoments.numPlayers[validRoundIndex]++;
                }
            }

            ctMoments.shotsPerTotalPlayers[validRoundIndex] = ctShots /
                (ctMoments.numPlayers[validRoundIndex] + tMoments.numPlayers[validRoundIndex]);
            ctMoments.killsPerTotalPlayers[validRoundIndex] = ctKills /
                (ctMoments.numPlayers[validRoundIndex] + tMoments.numPlayers[validRoundIndex]);
            ctMoments.shotsPerKill[validRoundIndex] /= ctKills;
            ctMoments.averageSpeedWhileShooting[validRoundIndex] /= ctShots;
            tMoments.shotsPerTotalPlayers[validRoundIndex] = tShots /
                (ctMoments.numPlayers[validRoundIndex] + tMoments.numPlayers[validRoundIndex]);
            tMoments.killsPerTotalPlayers[validRoundIndex] = tKills /
                (ctMoments.numPlayers[validRoundIndex] + tMoments.numPlayers[validRoundIndex]);
            tMoments.shotsPerKill[validRoundIndex] /= tKills;
            tMoments.averageSpeedWhileShooting[validRoundIndex] /= tShots;

            tickLength[validRoundIndex] = roundEndTickId[validRoundIndex] - plantTickId[validRoundIndex] + 1;
        }
        size = roundId.size();
    }
}
