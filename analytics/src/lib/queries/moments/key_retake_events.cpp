//
// Created by durst on 5/4/23.
//

#include "queries/moments/key_retake_events.h"
#include "bots/testing/command.h"
#include "file_helpers.h"

namespace csknow::key_retake_events {
    KeyRetakeEvents::KeyRetakeEvents(const Rounds &rounds, const Ticks &ticks, const PlayerAtTick & playerAtTick,
                                     const Plants & plants, const Defusals & defusals, const Kills & kills,
                                     const Say & say) {
        firedBeforeOrDuringThisTick.resize(ticks.size, false);
        plantFinishedBeforeOrDuringThisTick.resize(ticks.size, false);
        defusalFinishedBeforeOrDuringThisTick.resize(ticks.size, false);
        explosionBeforeOrDuringThisTick.resize(ticks.size, false);
        ctAliveAfterExplosion.resize(ticks.size, false);
        tAliveAfterDefusal.resize(ticks.size, false);
        testStartBeforeOrDuringThisTick.resize(ticks.size, false);
        testEndBeforeOrDuringThisTick.resize(ticks.size, false);

        roundHasPlant.resize(rounds.size,  false);
        roundCTAliveOnPlant.resize(rounds.size, 0);
        roundTAliveOnPlant.resize(rounds.size, 0);
        roundHasDefusal.resize(rounds.size,  false);
        roundHasRetakeCTSave.resize(rounds.size,  false);
        roundHasRetakeTSave.resize(rounds.size,  false);
        roundHasRetakeSave.resize(rounds.size,  false);
        roundC4Deaths.resize(rounds.size, 0);
        roundNonC4PostPlantWorldDeaths.resize(rounds.size, 0);
        roundTestName.resize(rounds.size, "INVALID");
        roundTestIndex.resize(rounds.size, INVALID_ID);
        roundNumTests.resize(rounds.size, INVALID_ID);
        roundHasCompleteTest.resize(rounds.size, false);
        roundHasFailedTest.resize(rounds.size, false);
        roundBaiters.resize(rounds.size, false);
#pragma omp parallel for
        for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
            bool foundFirstFireInRound = false, foundFirstPlantInRound = false, foundFirstDefusalInRound = false,
                foundExplosionInRound = false, foundTestStartInRound = false, foundTestFinishInRound = false;
            int64_t firstPlantTick = INVALID_ID;
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
                            firstPlantTick = tickIndex;
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

                // say event
                for (const auto & [_0, _1, sayIndex] :
                        ticks.sayPerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex)) {
                    std::string sayMessage = say.message[sayIndex];
                    if (sayMessage.find(test_ready_string) != std::string::npos) {
                        foundTestStartInRound = true;
                        std::vector<std::string> parsedMessage = parseString(sayMessage, '_');
                        roundTestName[roundIndex] = parsedMessage[1];
                        roundTestIndex[roundIndex] = std::stoi(parsedMessage[2]);
                        roundNumTests[roundIndex] = std::stoi(parsedMessage[3]);
                    }
                    else if (sayMessage.find(test_finished_string) != std::string::npos) {
                        foundTestFinishInRound = true;
                        roundHasCompleteTest[roundIndex] = foundTestStartInRound && foundTestFinishInRound;
                        if (sayMessage.find("Bait") != std::string::npos) {
                            roundBaiters[roundIndex] = true;
                        }
                    }
                    else if (sayMessage.find(test_failed_string) != std::string::npos) {
                        foundTestFinishInRound = true;
                        roundHasFailedTest[roundIndex] = foundTestStartInRound && foundTestFinishInRound;
                        if (sayMessage.find("Bait") != std::string::npos) {
                            roundBaiters[roundIndex] = true;
                        }
                    }
                }
                testStartBeforeOrDuringThisTick[tickIndex] = foundTestStartInRound;
                testEndBeforeOrDuringThisTick[tickIndex] = foundTestFinishInRound;
            }


            for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].maxId;
                 tickIndex >= rounds.ticksPerRound[roundIndex].minId; tickIndex--) {
                bool ctAlive = false, tAlive = false;
                for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
                     patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
                    if (playerAtTick.isAlive[patIndex]) {
                        if (playerAtTick.team[patIndex] == ENGINE_TEAM_CT) {
                            ctAlive = true;
                            if (firstPlantTick == tickIndex) {
                                roundCTAliveOnPlant[roundIndex]++;
                            }
                        }
                        else if (playerAtTick.team[patIndex] == ENGINE_TEAM_T) {
                            tAlive = true;
                            if (firstPlantTick == tickIndex) {
                                roundTAliveOnPlant[roundIndex]++;
                            }
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
            numC4Deaths [[maybe_unused]] = 0, numNonC4PostPlantWorldDeaths [[maybe_unused]] = 0;
        double pctRetakeNonSaveRounds [[maybe_unused]] = 0., pctRetakeCTSaveRounds [[maybe_unused]] = 0.,
            pctRetakeTSaveRounds [[maybe_unused]] = 0.;

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

        /*
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

        std::cout << "round id, num CT alive on plant, num T alive on plant" << std::endl;
        for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
            if (roundHasPlant[roundIndex]) {
                std::cout << roundIndex << "," << roundCTAliveOnPlant[roundIndex] << "," << roundTAliveOnPlant[roundIndex] << std::endl;
            }
        }
         */
    }

}
