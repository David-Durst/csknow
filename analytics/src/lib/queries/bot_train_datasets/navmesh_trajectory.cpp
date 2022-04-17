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
computeEngagementsPerRound(const Rounds & rounds, const Ticks & ticks, const PlayerAtTick & playerAtTick,
                           const WeaponFire & weaponFire, const Hurt & hurt, const int64_t roundId,
                           vector<EngagementIds> & engagementIds, const int64_t RADIUS_GAME_TICKS,
                           const TickRates & tickRates) {
    // first key is shooter, second is target
    map<int64_t, map<int64_t, EngagementIds>> activeEngagementIds;
    for (int64_t tickIndex = rounds.ticksPerRound[roundId].minId;
         tickIndex != -1 && tickIndex <= rounds.ticksPerRound[roundId].maxId; tickIndex++) {

        // first precompute who is alive, so know target's state when computing for shooter
        map<int64_t, bool> playerAlive;
        for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
             patIndex != -1 && patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
            playerAlive[playerAtTick.playerId[patIndex]] = playerAtTick.isAlive[patIndex];
        }
        // add the players who were hurt this tick to those alive, as you get hurt on tick you die
        if (ticks.hurtPerTick.find(tickIndex) != ticks.hurtPerTick.end()) {
            for (const auto & hurtId : ticks.hurtPerTick.at(tickIndex)) {
                // ignoring nade damage, so have to use same weapon as fired
                playerAlive[hurt.victim[hurtId]] = true;
            }
        }

        // if any of player's engagements are over, remove them
        // remove player if no active engagements
        vector<int64_t> shooterEngagementsToErase;
        for (const auto & shooterToSubmap : activeEngagementIds) {
            vector<int64_t> targetEngagementsToErase;
            for (const auto targetToEngagementIds : shooterToSubmap.second) {
                if (targetToEngagementIds.second.endTickId < tickIndex ||
                    // engagement ends when shooter dies or disconnects
                    !(playerAlive.find(shooterToSubmap.first) != playerAlive.end() &&
                      playerAlive[shooterToSubmap.first]) ||
                    // engagement ends when target dies or disconnects (if target is valid)
                    !(targetToEngagementIds.first == INVALID_ID ||
                      (playerAlive.find(targetToEngagementIds.first) != playerAlive.end() &&
                       playerAlive[targetToEngagementIds.first]))) {
                    EngagementIds & finishedIds = activeEngagementIds[shooterToSubmap.first][targetToEngagementIds.first];
                    finishedIds.endTickId = std::min(finishedIds.endTickId, tickIndex);
                    if ((finishedIds.startTickId == 353300 || finishedIds.startTickId == 353314) && finishedIds.shooterId == 7 && finishedIds.targetId == -1) {
                        int x = 1;
                    }
                    finishedIds.startTickId = weaponFire.tickId[finishedIds.shooterWeaponFireIds.front()];
                    finishedIds.endTickId = weaponFire.tickId[finishedIds.shooterWeaponFireIds.back()];
                    engagementIds.push_back(finishedIds);
                    targetEngagementsToErase.push_back(targetToEngagementIds.first);
                }
            }
            for (const auto & target : targetEngagementsToErase) {
                EngagementIds e = activeEngagementIds[shooterToSubmap.first][target];
                if ((e.startTickId == 353300 || e.startTickId == 353314) && e.shooterId == 7 && e.targetId == -1) {
                    int x = 1;
                }
                activeEngagementIds[shooterToSubmap.first].erase(target);
            }
            if (activeEngagementIds[shooterToSubmap.first].empty()) {
                shooterEngagementsToErase.push_back(shooterToSubmap.first);
            }
        }
        for (const auto & shooter : shooterEngagementsToErase) {
            activeEngagementIds.erase(shooter);
        }

        // now add engagements
        for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
             patIndex != -1 && patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
            int64_t shooterPlayerId = playerAtTick.playerId[patIndex];

            // skip if not alive and not on CT or T
            if (!playerAtTick.isAlive[patIndex] ||
                !(playerAtTick.team[patIndex] == CT_TEAM || playerAtTick.team[patIndex] == T_TEAM)) {
                continue;
            }

            // get if player shot
            int64_t engagementFireId = INVALID_ID;
            if (ticks.weaponFirePerTick.find(tickIndex) != ticks.weaponFirePerTick.end()) {
                for (const auto & fireId : ticks.weaponFirePerTick.at(tickIndex)) {
                    if (weaponFire.shooter[fireId] == shooterPlayerId) {
                        engagementFireId = fireId;
                        break;
                    }
                }
            }
            // if didn't shoot, then continue as can't change engagement state if didn't shoot
            if (engagementFireId == INVALID_ID) {
                continue;
            }

            // get if player hurt anyone
            vector<int64_t> engagementHurtIds;
            vector<int64_t> engagementTargetIds;
            if (ticks.hurtPerTick.find(tickIndex) != ticks.hurtPerTick.end()) {
                for (const auto & hurtId : ticks.hurtPerTick.at(tickIndex)) {
                    // ignoring nade damage, so have to use same weapon as fired
                    if (hurt.attacker[hurtId] == shooterPlayerId &&
                        hurt.weapon[hurtId] == weaponFire.weapon[engagementFireId]) {
                        engagementHurtIds.push_back(hurtId);
                        engagementTargetIds.push_back(hurt.victim[hurtId]);
                    }
                }
            }

            // always track when shooting for a no-target engagement
            // if no shooting at nothing engagement, then add a new one, no target to check
            if (activeEngagementIds[shooterPlayerId].find(INVALID_ID) == activeEngagementIds[shooterPlayerId].end()) {
                activeEngagementIds[shooterPlayerId][INVALID_ID].startTickId = tickIndex -
                        getLookbackDemoTick(rounds, ticks, playerAtTick, tickIndex, tickRates, RADIUS_GAME_TICKS, 1000);
                activeEngagementIds[shooterPlayerId][INVALID_ID].endTickId = tickIndex +
                        getLookforwardDemoTick(rounds, ticks, playerAtTick, tickIndex, tickRates, RADIUS_GAME_TICKS, 1000);
                int64_t newStartTickId = activeEngagementIds[shooterPlayerId][INVALID_ID].startTickId;
                if ((newStartTickId == 353300 || newStartTickId == 353314) && shooterPlayerId == 7) {
                    int x = 1;
                }
                activeEngagementIds[shooterPlayerId][INVALID_ID].firstHurtTick = INVALID_ID;
                activeEngagementIds[shooterPlayerId][INVALID_ID].lastHurtTick = INVALID_ID;
                activeEngagementIds[shooterPlayerId][INVALID_ID].shooterId = shooterPlayerId;
                activeEngagementIds[shooterPlayerId][INVALID_ID].targetId = INVALID_ID;
                // will add weapon fires below
                // activeEngagementIds[shooterPlayerId][INVALID_ID].shooterWeaponFireIds.push_back(engagementFireId);
                activeEngagementIds[shooterPlayerId][INVALID_ID].numHits = 0;
            }
            // else extend the shooting at no target engagmeent
            else {
                activeEngagementIds[shooterPlayerId][INVALID_ID].endTickId = tickIndex +
                        getLookforwardDemoTick(rounds, ticks, playerAtTick, tickIndex, tickRates, RADIUS_GAME_TICKS, 1000);
                // activeEngagementIds[shooterPlayerId][INVALID_ID].shooterWeaponFireIds.push_back(engagementFireId);
            }

            // if shot someone, then start/continue those engagements
            for (size_t i = 0; i < engagementHurtIds.size(); i++) {
                int64_t targetId = engagementTargetIds[i];
                int64_t newStartTickId = tickIndex -
                        getLookbackDemoTick(rounds, ticks, playerAtTick, tickIndex, tickRates, RADIUS_GAME_TICKS, 1000);
                int64_t newEndTickId = tickIndex +
                        getLookforwardDemoTick(rounds, ticks, playerAtTick, tickIndex, tickRates, RADIUS_GAME_TICKS, 1000);
                if (newStartTickId == newEndTickId) {
                    int x = 1;
                }
                // new engagement
                if (activeEngagementIds[shooterPlayerId].find(targetId) == activeEngagementIds[shooterPlayerId].end()) {
                    activeEngagementIds[shooterPlayerId][targetId].startTickId = newStartTickId;
                    activeEngagementIds[shooterPlayerId][targetId].endTickId = newEndTickId;
                    activeEngagementIds[shooterPlayerId][targetId].firstHurtTick = tickIndex;
                    activeEngagementIds[shooterPlayerId][targetId].lastHurtTick = tickIndex;
                    activeEngagementIds[shooterPlayerId][targetId].shooterId = shooterPlayerId;
                    activeEngagementIds[shooterPlayerId][targetId].targetId = targetId;
                    //activeEngagementIds[shooterPlayerId][targetId].shooterWeaponFireIds.push_back(engagementFireId);
                    activeEngagementIds[shooterPlayerId][targetId].targetHurtIds.push_back(engagementHurtIds[i]);
                    activeEngagementIds[shooterPlayerId][targetId].numHits = 1;
                }
                // continuing old engagement
                else {
                    activeEngagementIds[shooterPlayerId][targetId].endTickId = newEndTickId;
                    activeEngagementIds[shooterPlayerId][targetId].lastHurtTick = tickIndex;
                    // no need to add the weapon fire again, did it before this for loop
                    // activeEngagementIds[shooterPlayerId][targetId].shooterWeaponFireIds.push_back(engagementFireId);
                    activeEngagementIds[shooterPlayerId][targetId].targetHurtIds.push_back(engagementHurtIds[i]);
                    activeEngagementIds[shooterPlayerId][targetId].numHits++;
                }
                if (activeEngagementIds[shooterPlayerId][targetId].startTickId == 353300 &&
                    activeEngagementIds[shooterPlayerId][targetId].shooterId == 7) {
                    int x = 1;
                }
            }

            // add the weapon fire to all existing engagements with targets
            for (const auto & [targetId, engagementIdsToUpdate] : activeEngagementIds[shooterPlayerId]) {
                activeEngagementIds[shooterPlayerId][targetId].shooterWeaponFireIds.push_back(engagementFireId);
            }
        }
    }

    // if round ends abruptly, terminate all engagements
    // no need to remove from activeEngagementIds as that structure goes out of scope after this line
    for (const auto & shooterToSubmap : activeEngagementIds) {
        for (const auto targetToEngagementIds : shooterToSubmap.second) {
            EngagementIds finishedIds = targetToEngagementIds.second;
            finishedIds.startTickId = weaponFire.tickId[finishedIds.shooterWeaponFireIds.front()];
            finishedIds.endTickId = weaponFire.tickId[finishedIds.shooterWeaponFireIds.back()];
            engagementIds.push_back(finishedIds);
        }
    }
}


NavmeshTrajectoryResult queryNavmeshTrajectoryDataset(const Games & games, const Rounds & rounds,
                                                      const Ticks & ticks, const Players & players,
                                                      const PlayerAtTick & playerAtTick, const nav_mesh::nav_file & navFile) {
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

        // first compute all the engagements
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
