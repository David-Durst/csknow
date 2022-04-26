//
// Created by durst on 3/24/22.
//

#include "queries/bot_train_dataset/navmesh_trajectory.h"
#include "queries/lookback.h"
#include "geometryNavConversions.h"
#include "bots/thinker.h"
#include <utility>
#include <cassert>

void
computeTrajectoryPerRound(const Rounds & rounds, const Ticks & ticks, const PlayerAtTick & playerAtTick,
                          const nav_mesh::nav_file & navFile, const int64_t roundId,
                          map<int64_t, NavmeshTrajectoryResult::Trajectory> perRoundPlayerTrajectory) {
    // first key is shooter, second is target
    for (int64_t tickIndex = rounds.ticksPerRound[roundId].minId;
         tickIndex != -1 && tickIndex <= rounds.ticksPerRound[roundId].maxId; tickIndex++) {
        int64_t gameTickNumber = ticks.gameTickNumber[tickIndex];
        for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
             patIndex != -1 && patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
            int64_t playerId = playerAtTick.playerId[patIndex];
            const nav_mesh::nav_area & navMeshArea = navFile.get_nearest_area_by_position(vec3Conv({
                playerAtTick.posX[patIndex],
                playerAtTick.posY[patIndex],
                playerAtTick.posZ[patIndex]
            }));
            // update end every tick, so when die no need to do further processing
            if (playerAtTick.isAlive[patIndex]) {
                if (perRoundPlayerTrajectory.find(playerId) == perRoundPlayerTrajectory.end()) {
                    perRoundPlayerTrajectory[playerId].target = NavmeshTrajectoryResult::TrajectoryTarget::NOT_YET_KNOWN;
                    perRoundPlayerTrajectory[playerId].startEndTickIds = {tickIndex, tickIndex};
                    perRoundPlayerTrajectory[playerId].startEndGameTickNumbers = {gameTickNumber, gameTickNumber};
                    perRoundPlayerTrajectory[playerId].navMeshArea = {navMeshArea.get_id()};
                    perRoundPlayerTrajectory[playerId].navMeshPlace = {navMeshArea.m_place};
                    perRoundPlayerTrajectory[playerId].areaEntryPATId = {patIndex};
                }
                else {
                    perRoundPlayerTrajectory[playerId].startEndTickIds.maxId = tickIndex;
                    perRoundPlayerTrajectory[playerId].startEndGameTickNumbers.maxId = gameTickNumber;
                    if (perRoundPlayerTrajectory[playerId].navMeshArea.back() != navMeshArea.get_id()) {
                        perRoundPlayerTrajectory[playerId].navMeshArea.push_back(navMeshArea.get_id());
                        perRoundPlayerTrajectory[playerId].navMeshPlace.push_back(navMeshArea.m_place);
                        perRoundPlayerTrajectory[playerId].areaEntryPATId.push_back(patIndex);
                    }
                }
            }

        }
    }
}


NavmeshTrajectoryResult queryNavmeshTrajectoryDataset(const Games & games, const Rounds & rounds,
                                                      const Ticks & ticks, const Players & players,
                                                      const PlayerAtTick & playerAtTick,
                                                      const std::map<std::string, const nav_mesh::nav_file> & mapNavs) {
    int numThreads = omp_get_max_threads();
    // stored per trajectory
    vector<int64_t> tmpSourcePlayerId[numThreads];
    vector<string> tmpSourcePlayerName[numThreads];
    vector<string> tmpDemoName[numThreads];
    vector<NavmeshTrajectoryResult::Trajectory> tmpTrajectory[numThreads];
    // stored once for all tracjectories in a round
    vector<int64_t> tmpRoundIds[numThreads];
    vector<int64_t> tmpRoundStarts[numThreads];
    vector<int64_t> tmpRoundSizes[numThreads];

//#pragma omp parallel for
    for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
        int threadNum = omp_get_thread_num();
        string mapName = games.mapName[rounds.gameId[roundIndex]];
        const nav_mesh::nav_file & navFile = mapNavs.at(mapName);

        // find all the bombsite A and B locations
        vector<uint32_t> aLocations, bLocations;
        for (const auto & navMeshArea : navFile.m_areas) {
            if (navFile.m_places[navMeshArea.m_place] == "BombsiteA") {
                aLocations.push_back(navMeshArea.get_id());
            }
            else if (navFile.m_places[navMeshArea.m_place] == "BombsiteB") {
                bLocations.push_back(navMeshArea.get_id());
            }
        }

        // compute all the per round trajectories
        map<int64_t, NavmeshTrajectoryResult::Trajectory> perRoundPlayerTrajectory;
        computeTrajectoryPerRound(rounds, ticks, playerAtTick, navFile, roundIndex, perRoundPlayerTrajectory);
        // each engagement is ENGAGEMENT_SECONDS_RADIUS seconds before first shot (or round start, whichever later)
        // until ENGAGEMENT_SECONDS_RADIUS after last shot at same target (or death or round end or next engagement, whichever soonest)
        // can have multiple engagements if multiple targets
        // if shooting at a target, only switch to no target engagement after target engagement ends
        vector<EngagementIds> engagementIds;
        computeEngagementsPerRound(rounds, ticks, playerAtTick, weaponFire, hurt, roundIndex, engagementIds,
                                   RADIUS_GAME_TICKS, tickRates);

        for (int i = 0; i < engagementIds.size(); i++) {
            if (engagementIds[i].startTickId == engagementIds[i].endTickId) {
                int x = 1;
            }
        }

        vector<EngagementIds> validEngagementIds;
        set<int64_t> validWeapons = {
                //0, //UNKNOWN
                102, //MP9
                206, //Negev
                309, //AWP
                //401, //Zeus x27
                10, //R8 Revolver
                106, //P90
                201, //Sawed-Off
                302, //FAMAS
                305, //M4A1
                //501, //Decoy Grenade
                //-1, //empty
                3, //P250
                8, //CZ75 Auto
                104, //MAC-10
                310, //SCAR-20
                6, //Dual Berettas
                203, //MAG-7
                303, //AK-47
                //403, //Kevlar + Helmet
                205, //M249
                307, //SG 553
                //406, //Defuse Kit
                2, //Glock-18
                4, //Desert Eagle
                5, //Five-SeveN
                103, //PP-Bizon
                105, //UMP-45
                //506, //HE Grenade
                311, //G3SG1
                //402, //Kevlar Vest
                //404, //C4
                101, //MP7
                202, //Nova
                204, //XM1014
                304, //M4A4
                308, //AUG
                //405, //Knife
                //503, //Incendiary Grenade
                //505, //Smoke Grenade
                1, //P2000
                7, //Tec-9
                9, //USP-S
                306, //SSG 08
                502, //Molotov
                107, //MP5-SD
                301, //Galil AR
                //407, //World
                //504, //Flashbang
        };

        for (const auto & engagementId : engagementIds) {
            int64_t numValidFires = 0;
            for (const auto & engagementWeaponFireId : engagementId.shooterWeaponFireIds) {
                int64_t weaponNumber = weaponFire.weapon[engagementWeaponFireId];
                if (validWeapons.find(weaponNumber) != validWeapons.end() &&
                        (weaponNumber == 302 || weaponNumber == 305 || weaponNumber == 303 || weaponNumber == 304)) {
                    numValidFires++;
                }
            }
            if (numValidFires >= 3 && engagementId.targetHurtIds.size() >= 1) {
                validEngagementIds.push_back(engagementId);
            }
        }

        // engagementIndex 127
        map<int64_t, map<int64_t, vector<int64_t>>> tickToShooterToEngagementIds;
        for (int i = 0; i < validEngagementIds.size(); i++) {
            const EngagementIds &oneEngagementIds = validEngagementIds[i];
            // end on tick before last tick of engagement so for all ticks know alive for next one
            // when computing results
            if (validEngagementIds[i].startTickId == validEngagementIds[i].endTickId) {
                int x = 1;
            }
            if (validEngagementIds[i].startTickId == 353300 &&
                validEngagementIds[i].shooterId == 7) {
                int x = 1;
            }
            for (int64_t tickId = oneEngagementIds.startTickId; tickId < oneEngagementIds.endTickId; tickId++) {
                tickToShooterToEngagementIds[tickId][oneEngagementIds.shooterId].push_back(i);
            }
        }

        map<int64_t, vector<int64_t>> shooterToWeaponFireGameTicks;
        for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
             tickIndex != -1 && tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {
            if (ticks.weaponFirePerTick.find(tickIndex) != ticks.weaponFirePerTick.end()) {
                for (const auto & weaponFireIndex : ticks.weaponFirePerTick.at(tickIndex)) {
                    // ensure inserting unique values in increasing order
                    assert(shooterToWeaponFireGameTicks[weaponFire.shooter[weaponFireIndex]].empty() ||
                                   shooterToWeaponFireGameTicks[weaponFire.shooter[weaponFireIndex]].back() < ticks.gameTickNumber[tickIndex]);
                    shooterToWeaponFireGameTicks[weaponFire.shooter[weaponFireIndex]].push_back(ticks.gameTickNumber[tickIndex]);
                }
            }
        }

        computeEngagementResults(rounds, ticks, playerAtTick, roundIndex, validEngagementIds,
                                 tickToShooterToEngagementIds, shooterToWeaponFireGameTicks,
                                 RADIUS_GAME_TICKS, tickRates,tmpStates[threadNum], tmpActions[threadNum]);
    }


    NavmeshTrajectoryResult result(navFile);
    vector<int64_t> roundsProcessedPerThread(numThreads, 0);
    while (true) {
        bool roundToProcess = false;
        int64_t minThreadId = -1;
        int64_t minRoundId = -1;
        for (int64_t threadId = 0; threadId < numThreads; threadId++) {
            if (roundsProcessedPerThread[threadId] < tmpRoundIds[threadId].size()) {
                roundToProcess = true;
                if (minThreadId == -1 || tmpRoundIds[threadId][roundsProcessedPerThread[threadId]] < minRoundId) {
                    minThreadId = threadId;
                    minRoundId = tmpRoundIds[minThreadId][roundsProcessedPerThread[minThreadId]];
                }

            }
        }
        if (!roundToProcess) {
            break;
        }
        int64_t startTrajectoryEntry = result.roundId.size();
        int64_t roundStart = tmpRoundStarts[minThreadId][roundsProcessedPerThread[minThreadId]];
        int64_t roundEnd = roundStart + tmpRoundSizes[minThreadId][roundsProcessedPerThread[minThreadId]];
        for (int tmpRowId = roundStart; tmpRowId < roundEnd; tmpRowId++) {
            result.roundId.push_back(minRoundId);
            result.sourcePlayerId.push_back(tmpSourcePlayerId[minThreadId][tmpRowId]);
            result.sourcePlayerName.push_back(tmpSourcePlayerName[minThreadId][tmpRowId]);
            result.demoName.push_back(tmpDemoName[minThreadId][tmpRowId]);
            result.trajectory.push_back(tmpTrajectory[minThreadId][tmpRowId]);
        }
        result.trajectoryPerRound.push_back({startTrajectoryEntry, static_cast<int64_t>(result.roundId.size())});
        roundsProcessedPerThread[minThreadId]++;
    }
    result.size = result.tickId.size();
    return result;
    return result;
}
 */
