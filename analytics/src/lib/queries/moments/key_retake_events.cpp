//
// Created by durst on 5/4/23.
//

#include "queries/moments/key_retake_events.h"

namespace csknow::key_retake_events {
    KeyRetakeEvents::KeyRetakeEvents(const Rounds &rounds, const Ticks &ticks, const PlayerAtTick & playerAtTick,
                                     const Plants & plants, const Defusals & defusals, const Kills & kills) {
        firedBeforeOrDuringThisTick.resize(ticks.size, false);
        plantFinishedBeforeOrDuringThisTick.resize(ticks.size, false);
        defusalFinishedBeforeOrDuringThisTick.resize(ticks.size, false);
        explosionBeforeOrDuringThisTick.resize(ticks.size, false);
        ctAliveAfterExplosion.resize(ticks.size, false);
        tAliveAfterDefusal.resize(ticks.size, false);
        roundHasPlant.resize(ticks.size,  false);
        roundHasDefusal.resize(ticks.size,  false);
        roundHasRetakeCTSave.resize(ticks.size,  false);
        roundHasRetakeTSave.resize(ticks.size,  false);
        roundHasRetakeSave.resize(ticks.size,  false);
        roundC4Deaths.resize(ticks.size, 0);
        roundNonC4PostPlantWorldDeaths.resize(ticks.size, 0);
#pragma omp parallel for
        for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
            bool foundFirstFireInRound = false, foundFirstPlantInRound = false, foundFirstDefusalInRound = false,
                foundExplosionInRound = false;
            int64_t firstExplosionTick = INVALID_ID;
            for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
                 tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {
                // fire event
                if (!foundFirstFireInRound) {
                    for ([[maybe_unused]] const auto & _ :
                            ticks.weaponFirePerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex)) {
                        foundFirstFireInRound = true;
                        break;
                    }
                }
                firedBeforeOrDuringThisTick[tickIndex] = foundFirstFireInRound;

                // plant event
                if (!foundFirstPlantInRound) {
                    for (const auto & [_0, _1, plantIndex] :
                            ticks.plantsEndPerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex)) {
                        if (plants.succesful[plantIndex]) {
                            foundFirstPlantInRound = true;
                            break;
                        }
                    }
                }
                plantFinishedBeforeOrDuringThisTick[tickIndex] = foundFirstPlantInRound;

                // defusal event
                if (!foundFirstDefusalInRound) {
                    for (const auto & [_0, _1, defusalIndex] :
                            ticks.defusalsEndPerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex)) {
                        if (defusals.succesful[defusalIndex]) {
                            foundFirstDefusalInRound = true;
                            break;
                        }
                    }
                }
                defusalFinishedBeforeOrDuringThisTick[tickIndex] = foundFirstDefusalInRound;

                // explosion event
                if (!foundExplosionInRound) {
                    for ([[maybe_unused]] const auto & _ :
                            ticks.explosionsPerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex)) {
                        foundExplosionInRound = true;
                        firstExplosionTick = tickIndex;
                        break;
                    }
                }
                explosionBeforeOrDuringThisTick[tickIndex] = foundExplosionInRound;

                for (const auto & [_0, _1, killIndex] :
                        ticks.killsPerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex)) {
                    if (kills.killer[killIndex] == INVALID_ID && foundFirstPlantInRound) {
                        if (firstExplosionTick == tickIndex) {
                            roundC4Deaths[roundIndex]++;
                        }
                        else {
                            roundNonC4PostPlantWorldDeaths[roundIndex]++;
                        }
                    }
                }
            }


            for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].maxId;
                 tickIndex >= rounds.ticksPerRound[roundIndex].minId; tickIndex--) {
                bool ctAlive = false, tAlive = false;
                for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
                     patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
                    if (playerAtTick.isAlive[patIndex]) {
                        if (playerAtTick.team[patIndex] == ENGINE_TEAM_CT) {
                            ctAlive = true;
                        }
                        else if (playerAtTick.team[patIndex] == ENGINE_TEAM_T) {
                            tAlive = true;
                        }
                    }
                }

                if (ctAlive && explosionBeforeOrDuringThisTick[tickIndex]) {
                    ctAliveAfterExplosion[tickIndex] = true;
                }
                if (tAlive && defusalFinishedBeforeOrDuringThisTick[tickIndex]) {
                    tAliveAfterDefusal[tickIndex] = true;
                }

                if (tickIndex < rounds.ticksPerRound[roundIndex].maxId) {
                    ctAliveAfterExplosion[tickIndex] =
                            ctAliveAfterExplosion[tickIndex] || ctAliveAfterExplosion[tickIndex+1];
                    tAliveAfterDefusal[tickIndex] = tAliveAfterDefusal[tickIndex] || tAliveAfterDefusal[tickIndex+1];
                }
            }

            int64_t lastTickInRound = rounds.ticksPerRound[roundIndex].maxId;
            roundHasPlant[roundIndex] = plantFinishedBeforeOrDuringThisTick[lastTickInRound];
            roundHasDefusal[roundIndex] = defusalFinishedBeforeOrDuringThisTick[lastTickInRound];
            roundHasRetakeCTSave[roundIndex] = ctAliveAfterExplosion[lastTickInRound];
            roundHasRetakeTSave[roundIndex] = tAliveAfterDefusal[lastTickInRound];
            roundHasRetakeSave[roundIndex] = roundHasRetakeCTSave[roundIndex] || roundHasRetakeTSave[roundIndex];
        }
        int numRetakeRounds = 0, numRetakeNonSaveRounds = 0, numRetakeCTSaveRounds = 0, numRetakeTSaveRounds = 0,
            numC4Deaths = 0, numNonC4PostPlantWorldDeaths = 0;
        double pctRetakeNonSaveRounds = 0., pctRetakeCTSaveRounds = 0., pctRetakeTSaveRounds = 0.;

        for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
            if (roundHasPlant[roundIndex]) {
                numRetakeRounds++;
                if (roundHasRetakeCTSave[roundIndex]) {
                    numRetakeCTSaveRounds++;
                }
                else if (roundHasRetakeTSave[roundIndex]) {
                    numRetakeTSaveRounds++;
                }
                else {
                    numRetakeNonSaveRounds++;
                }
            }
            numC4Deaths += roundC4Deaths[roundIndex];
            numNonC4PostPlantWorldDeaths += roundNonC4PostPlantWorldDeaths[roundIndex];
        }
        pctRetakeNonSaveRounds = static_cast<double>(numRetakeNonSaveRounds) / static_cast<double>(numRetakeRounds);
        pctRetakeCTSaveRounds = static_cast<double>(numRetakeCTSaveRounds) / static_cast<double>(numRetakeRounds);
        pctRetakeTSaveRounds = static_cast<double>(numRetakeTSaveRounds) / static_cast<double>(numRetakeRounds);

        std::cout << "num retake rounds: " << numRetakeRounds
            << ", num retake non-save rounds: " << numRetakeNonSaveRounds
            << ", pct retake non-save rounds: " << pctRetakeNonSaveRounds
            << ", num retake CT save rounds: " << numRetakeCTSaveRounds
            << ", pct retake CT save rounds: " << pctRetakeCTSaveRounds
            << ", num retake T save rounds: " << numRetakeTSaveRounds
            << ", pct retake T save rounds: " << pctRetakeTSaveRounds
            << ", num C4 deaths: " << numC4Deaths
            << ", num non-c4 post plant world deaths: " << numNonC4PostPlantWorldDeaths
            << std::endl;
    }

}
