//
// Created by durst on 3/24/22.
//

#include "queries/bot_train_dataset/engagement.h"
#include "queries/lookback.h"
#include "geometryNavConversions.h"
#include "bots/thinker.h"
#include <utility>
#include <cassert>

struct EngagementIds {
    int64_t startTickId;
    int64_t endTickId;
    int64_t firstHurtTick;
    int64_t lastHurtTick;
    int64_t shooterId;
    int64_t targetId;
    vector<int64_t> shooterWeaponFireIds;
    int32_t numHits;
};

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

        // if any of player's engagements are over, remove them
        // remove player if no active engagements
        vector<int64_t> shooterEngagementsToErase;
        for (const auto & shooterToSubmap : activeEngagementIds) {
            vector<int64_t> targetEngagementsToErase;
            for (const auto targetToEngagementIds : shooterToSubmap.second) {
                if (targetToEngagementIds.second.endTickId < tickIndex ||
                    // engagement ends when shooter dies or disconnects
                    !(playerAlive.find(shooterToSubmap.first) != playerAlive.end() &&
                      playerAlive[targetToEngagementIds.first]) ||
                    // engagement ends when target dies or disconnects (if target is valid)
                    !(targetToEngagementIds.first == INVALID_ID ||
                      (playerAlive.find(targetToEngagementIds.first) != playerAlive.end() &&
                       playerAlive[targetToEngagementIds.first]))) {
                    engagementIds.push_back(activeEngagementIds[shooterToSubmap.first][targetToEngagementIds.first]);
                    targetEngagementsToErase.push_back(targetToEngagementIds.first);
                }
            }
            for (const auto & target : targetEngagementsToErase) {
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

            // if shot but didn't hit anyone, only start/continue engagement is no active engagement with a target
            if (engagementHurtIds.empty()) {
                // if no engagements, then add a new one, no target to check
                if (activeEngagementIds[shooterPlayerId].empty()) {
                    activeEngagementIds[shooterPlayerId][INVALID_ID].startTickId = tickIndex -
                            getLookbackDemoTick(rounds, ticks, playerAtTick, tickIndex, tickRates, RADIUS_GAME_TICKS, 1000);
                    activeEngagementIds[shooterPlayerId][INVALID_ID].endTickId = tickIndex +
                            getLookforwardDemoTick(rounds, ticks, playerAtTick, tickIndex, tickRates, RADIUS_GAME_TICKS, 1000);
                    activeEngagementIds[shooterPlayerId][INVALID_ID].firstHurtTick = INVALID_ID;
                    activeEngagementIds[shooterPlayerId][INVALID_ID].lastHurtTick = INVALID_ID;
                    activeEngagementIds[shooterPlayerId][INVALID_ID].shooterId = shooterPlayerId;
                    activeEngagementIds[shooterPlayerId][INVALID_ID].targetId = INVALID_ID;
                    activeEngagementIds[shooterPlayerId][INVALID_ID].numHits = 0;
                }
                // else if have engagements, if only one without a target, then extend it
                else if (activeEngagementIds[shooterPlayerId].size() == 1 &&
                         activeEngagementIds[shooterPlayerId].find(INVALID_ID) != activeEngagementIds[shooterPlayerId].end()){
                    activeEngagementIds[shooterPlayerId][INVALID_ID].endTickId = tickIndex +
                            getLookforwardDemoTick(rounds, ticks, playerAtTick, tickIndex, tickRates, RADIUS_GAME_TICKS, 1000);
                    activeEngagementIds[shooterPlayerId][INVALID_ID].lastHurtTick = tickIndex;
                }
            }
            // if shot someone, then start/continue those engagements
            else {
                for (size_t i = 0; i < engagementHurtIds.size(); i++) {
                    int64_t targetId = engagementTargetIds[i];
                    int64_t newStartTickId = tickIndex -
                            getLookbackDemoTick(rounds, ticks, playerAtTick, tickIndex, tickRates, RADIUS_GAME_TICKS, 1000);
                    int64_t newEndTickId = tickIndex +
                            getLookforwardDemoTick(rounds, ticks, playerAtTick, tickIndex, tickRates, RADIUS_GAME_TICKS, 1000);
                    // if existing engagement with no target, remove it as replaced with engagement with target
                    if (activeEngagementIds[shooterPlayerId].find(INVALID_ID) == activeEngagementIds[shooterPlayerId].end()) {
                        activeEngagementIds[shooterPlayerId].erase(INVALID_ID);
                    }

                    // new engagement
                    if (activeEngagementIds[shooterPlayerId].find(targetId) == activeEngagementIds[shooterPlayerId].end()) {
                        activeEngagementIds[shooterPlayerId][targetId].startTickId = newStartTickId;
                        activeEngagementIds[shooterPlayerId][targetId].endTickId = newEndTickId;
                        activeEngagementIds[shooterPlayerId][targetId].firstHurtTick = tickIndex;
                        activeEngagementIds[shooterPlayerId][targetId].lastHurtTick = tickIndex;
                        activeEngagementIds[shooterPlayerId][targetId].shooterId = shooterPlayerId;
                        activeEngagementIds[shooterPlayerId][targetId].targetId = targetId;
                        activeEngagementIds[shooterPlayerId][targetId].numHits = 1;
                    }
                    // continuing old engagement
                    else {
                        activeEngagementIds[shooterPlayerId][targetId].endTickId = newEndTickId;
                        activeEngagementIds[shooterPlayerId][targetId].lastHurtTick = tickIndex;
                        activeEngagementIds[shooterPlayerId][targetId].numHits++;
                    }
                }
            }
        }
    }

    // if round ends abruptly, terminate all engagements
    // no need to remove from activeEngagementIds as that structure goes out of scope after this line
    for (const auto & shooterToSubmap : activeEngagementIds) {
        for (const auto targetToEngagementIds : shooterToSubmap.second) {
            engagementIds.push_back(targetToEngagementIds.second);
        }
    }
}

EngagementResult::PosState
computePosState(const PlayerAtTick & playerAtTick, Vec3 shooterOrigin,
                const RotationMatrix3D & shooterRotationMatrix, int64_t targetPATId, int64_t shooterPATId) {
    EngagementResult::PosState result;
    result.eyePosRelativeToShooter = translateThenRotate(shooterOrigin, shooterRotationMatrix,
                                                         {playerAtTick.posX[targetPATId],
                                                          playerAtTick.posY[targetPATId],
                                                          playerAtTick.eyePosZ[targetPATId]});

    result.velocityRelativeToShooter = {
            playerAtTick.velX[targetPATId] - playerAtTick.velX[shooterPATId],
            playerAtTick.velY[targetPATId] - playerAtTick.velY[shooterPATId],
            playerAtTick.velZ[targetPATId] - playerAtTick.velZ[shooterPATId],
            };

    result.viewAngleRelativeToShooter = {
            playerAtTick.viewX[targetPATId] - playerAtTick.viewX[shooterPATId],
            playerAtTick.viewY[targetPATId] - playerAtTick.viewY[shooterPATId],
            };

    result.isCrouching = playerAtTick.isCrouching[targetPATId];
    result.isWalking = playerAtTick.isWalking[targetPATId];
    result.isScoped = playerAtTick.isScoped[targetPATId];
    result.isAirborne = playerAtTick.isAirborne[targetPATId];
    result.remainingFlashTime = playerAtTick.remainingFlashTime[targetPATId];
    return result;
}

void computeEngagementResults(const Rounds & rounds, const Ticks & ticks, const PlayerAtTick & playerAtTick, const int64_t roundId,
                              const vector<EngagementIds> & engagementIds, map<int64_t, map<int64_t, vector<int64_t>>> tickToShooterToEngagementIds,
                              const map<int64_t, vector<int64_t>> & shooterToWeaponFireGameTicks, const int64_t RADIUS_GAME_TICKS,
                              const TickRates & tickRates,
                              vector<EngagementResult::TimeStepState> & states, vector<EngagementResult::TimeStepAction> & actions) {
    for (int64_t tickIndex = rounds.ticksPerRound[roundId].minId;
         tickIndex != -1 && tickIndex <= rounds.ticksPerRound[roundId].maxId; tickIndex++) {
        if (tickToShooterToEngagementIds.find(tickIndex) != tickToShooterToEngagementIds.end()) {
            vector<int64_t> activePlayers;
            map<int64_t, int64_t> activePlayerPATIds;
            map<int64_t, int16_t> activePlayerTeams;
            // get all players on CT or T and their PAT ids in this tick
            for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
                 patIndex != -1 && patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
                if (playerAtTick.team[patIndex] == CT_TEAM || playerAtTick.team[patIndex] == T_TEAM) {
                    activePlayers.push_back(playerAtTick.playerId[patIndex]);
                    activePlayerPATIds[playerAtTick.playerId[patIndex]] = patIndex;
                    activePlayerTeams[playerAtTick.playerId[patIndex]] = playerAtTick.team[patIndex];
                }
            }
            assert(activePlayerPATIds.size() <= NUM_PLAYERS);
            std::sort(activePlayers.begin(), activePlayers.end());


            for (const auto &[shooter, engagementIdIndices] : tickToShooterToEngagementIds[tickIndex]) {
                assert(!engagementIds.empty());

                EngagementResult::TimeStepState state;
                int64_t shooterPATId = activePlayerPATIds[shooter];
                state.team = playerAtTick.team[shooterPATId];
                state.globalShooterEyePos = {playerAtTick.posX[shooterPATId], playerAtTick.posY[shooterPATId],
                                             playerAtTick.eyePosZ[shooterPATId]};
                state.globalShooterVelocity = {playerAtTick.velX[shooterPATId], playerAtTick.velY[shooterPATId],
                                               playerAtTick.velZ[shooterPATId]};
                state.globalShooterViewAngle = {playerAtTick.viewX[shooterPATId], playerAtTick.viewY[shooterPATId]};
                Vec2 aimPunchAngle = {playerAtTick.aimPunchX[shooterPATId], playerAtTick.aimPunchY[shooterPATId]};
                Vec2 viewPunchAngle = {playerAtTick.viewPunchX[shooterPATId],
                                       playerAtTick.viewPunchY[shooterPATId]};
                state.viewAngleWithActualRecoil =
                        state.globalShooterViewAngle + aimPunchAngle * WEAPON_RECOIL_SCALE;
                state.viewAngleWithVisualRecoil = state.globalShooterViewAngle + viewPunchAngle +
                                                  aimPunchAngle * WEAPON_RECOIL_SCALE * VIEW_RECOIL_TRACKING;


                int64_t priorFireGameTick = ticks.gameTickNumber[tickIndex] - RADIUS_GAME_TICKS,
                        nextFireGameTick = ticks.gameTickNumber[tickIndex] + RADIUS_GAME_TICKS;
                if (shooterToWeaponFireGameTicks.find(shooter) != shooterToWeaponFireGameTicks.end()) {
                    for (size_t i = 0; i < shooterToWeaponFireGameTicks.at(shooter).size(); i++) {
                        if (shooterToWeaponFireGameTicks.at(shooter)[i] < ticks.gameTickNumber[tickIndex] &&
                            shooterToWeaponFireGameTicks.at(shooter)[i] > priorFireGameTick) {
                            priorFireGameTick = shooterToWeaponFireGameTicks.at(shooter)[i];
                        }
                        if (shooterToWeaponFireGameTicks.at(shooter)[i] > ticks.gameTickNumber[tickIndex] &&
                            shooterToWeaponFireGameTicks.at(shooter)[i] < nextFireGameTick) {
                            nextFireGameTick = shooterToWeaponFireGameTicks.at(shooter)[i];
                        }
                    }
                }
                state.secondsSinceLastFire = (ticks.gameTickNumber[tickIndex] - priorFireGameTick) /
                                             static_cast<double>(tickRates.gameTickRate);
                double secondsUntilNextFire = (nextFireGameTick - ticks.gameTickNumber[tickIndex]) /
                                              static_cast<double>(tickRates.gameTickRate);

                state.roundId = roundId;
                state.tickId = tickIndex;
                state.shooterPatId = shooterPATId;
                // start with target inactive and set to 0
                state.target = {false, false, false,
                                {{ 0., 0., 0.}, { 0., 0., 0.}, {0., 0.}, false, false, false, false, 0. }, false, 0};

                RotationMatrix3D shooterRotationMatrix(
                        {state.globalShooterViewAngle.x, state.globalShooterViewAngle.y}, true);

                // if more than 1 engagement for this shooter on this tick, ranking is
                // 1. actively shooting at target
                // 2. number of hits landed during engagement
                int64_t bestEngagementIndex = 0;
                std::set<int64_t> engagedTargets{engagementIds[engagementIdIndices[0]].targetId};
                if (engagementIdIndices.size() > 1) {
                    int32_t maxHits = -1;
                    bool maxInShootingRange = false;
                    for (size_t i = 0; i < engagementIdIndices.size(); i++) {
                        engagedTargets.insert(engagementIds[engagementIdIndices[i]].targetId);
                        int32_t curHits = engagementIds[engagementIdIndices[i]].numHits;
                        bool curInShootingRange = engagementIds[engagementIdIndices[i]].firstHurtTick != INVALID_ID &&
                                engagementIds[engagementIdIndices[i]].firstHurtTick <= tickIndex &&
                                engagementIds[engagementIdIndices[i]].lastHurtTick >= tickIndex;
                        if ((!maxInShootingRange && curInShootingRange) ||
                            (!(maxInShootingRange && !curInShootingRange) && maxHits < curHits)) {
                            maxHits = curHits;
                            maxInShootingRange = curInShootingRange;
                            bestEngagementIndex = engagementIdIndices[i];
                        }
                    }
                }

                int8_t friendlyIndex = 0;
                int8_t enemyIndex = 0;
                int64_t targetPATId;
                for (const auto &activePlayerId : activePlayers) {
                    int64_t activePATId = activePlayerPATIds[activePlayerId];
                    if (playerAtTick.team[activePATId] == playerAtTick.team[shooterPATId]) {
                        state.friendlyPlayerStates[friendlyIndex].slotFilled = true;
                        state.friendlyPlayerStates[friendlyIndex].alive = playerAtTick.isAlive[activePATId];
                        state.friendlyPlayerStates[friendlyIndex].posState = computePosState(playerAtTick,
                                                                                             state.globalShooterEyePos,
                                                                                             shooterRotationMatrix,
                                                                                             activePATId,
                                                                                             shooterPATId);
                        state.friendlyPlayerStates[friendlyIndex].money = playerAtTick.money[activePATId];
                        state.friendlyPlayerStates[friendlyIndex].activeWeapon = playerAtTick.activeWeapon[activePATId];
                        state.friendlyPlayerStates[friendlyIndex].primaryWeapon = playerAtTick.primaryWeapon[activePATId];
                        state.friendlyPlayerStates[friendlyIndex].secondaryWeapon = playerAtTick.secondaryWeapon[activePATId];
                        if (playerAtTick.activeWeapon[activePATId] == playerAtTick.primaryWeapon[activePATId]) {
                            state.friendlyPlayerStates[friendlyIndex].currentClipBullets = playerAtTick.primaryBulletsClip[activePATId];
                        } else if (playerAtTick.activeWeapon[activePATId] ==
                                   playerAtTick.secondaryWeapon[activePATId]) {
                            state.friendlyPlayerStates[friendlyIndex].currentClipBullets = playerAtTick.secondaryBulletsClip[activePATId];
                        } else {
                            state.friendlyPlayerStates[friendlyIndex].currentClipBullets = -1;
                        }
                        state.friendlyPlayerStates[friendlyIndex].primaryClipBullets = playerAtTick.primaryBulletsClip[activePATId];
                        state.friendlyPlayerStates[friendlyIndex].secondaryClipBullets = playerAtTick.secondaryBulletsClip[activePATId];
                        state.friendlyPlayerStates[friendlyIndex].health = playerAtTick.health[activePATId];
                        state.friendlyPlayerStates[friendlyIndex].armor = playerAtTick.armor[activePATId];
                        if (activePATId == shooterPATId) {
                            state.shooter = state.friendlyPlayerStates[friendlyIndex];
                        }
                        friendlyIndex++;
                    } else {
                        state.enemyPlayerStates[enemyIndex].slotFilled = true;
                        state.enemyPlayerStates[enemyIndex].alive = playerAtTick.isAlive[activePATId];
                        state.enemyPlayerStates[enemyIndex].engaged =
                                engagedTargets.find(activePlayerId) != engagedTargets.end();
                        state.enemyPlayerStates[enemyIndex].posState = computePosState(playerAtTick,
                                                                                       state.globalShooterEyePos,
                                                                                       shooterRotationMatrix,
                                                                                       activePATId, shooterPATId);
                        // save if no primary weapon
                        state.enemyPlayerStates[enemyIndex].saveRound =
                                playerAtTick.primaryWeapon[activePATId] != INVALID_ID;
                        state.enemyPlayerStates[enemyIndex].activeWeapon = playerAtTick.activeWeapon[activePATId];
                        if (activePlayerId == engagementIds[bestEngagementIndex].targetId) {
                            targetPATId = activePATId;
                            state.target = state.enemyPlayerStates[enemyIndex];
                        }
                        enemyIndex++;
                    }
                }

                EngagementResult::TimeStepAction action;
                action.secondsUntilEngagementOver =
                        (ticks.gameTickNumber[engagementIds[bestEngagementIndex].endTickId] -
                         ticks.gameTickNumber[tickIndex]) /
                        static_cast<double>(tickRates.gameTickRate);
                // know alive next tick as tickToShooterToEngagementIds ends one tick early for each engagement

                vector<int64_t> nextShooterPATIdsFour, nextShooterPATIdsEight;
                int64_t fourOffset = std::min(4L, rounds.ticksPerRound[roundId].maxId - tickIndex + 1);
                int64_t eightOffset = std::min(8L, rounds.ticksPerRound[roundId].maxId - tickIndex + 1);
                for (int64_t tickOffset = 1; tickOffset <= eightOffset; tickOffset++) {
                    for (int64_t patIndex = ticks.patPerTick[tickIndex + tickOffset].minId;
                         patIndex != -1 && patIndex <= ticks.patPerTick[tickIndex + tickOffset].maxId; patIndex++) {
                        if (playerAtTick.playerId[patIndex] == shooter) {
                            if (tickOffset <= fourOffset) {
                                nextShooterPATIdsFour.push_back(patIndex);
                            }
                            nextShooterPATIdsEight.push_back(patIndex);
                            break;
                        }
                    }
                }
                assert(!nextShooterPATIdsEight.empty());
                int64_t nextPATId = nextShooterPATIdsEight[0];
                action.deltaPos = Vec3{playerAtTick.posX[nextPATId], playerAtTick.posY[nextPATId], playerAtTick.posZ[nextPATId]} -
                        Vec3{playerAtTick.posX[shooterPATId], playerAtTick.posY[shooterPATId], playerAtTick.posZ[shooterPATId]};
                action.deltaView = (Vec2{playerAtTick.viewX[nextPATId], playerAtTick.viewY[nextPATId]} -
                                  Vec2{playerAtTick.viewX[shooterPATId], playerAtTick.viewY[shooterPATId]});
                action.deltaView4 = (Vec2{playerAtTick.viewX[nextShooterPATIdsFour.back()], playerAtTick.viewY[nextShooterPATIdsFour.back()]} -
                                    Vec2{playerAtTick.viewX[shooterPATId], playerAtTick.viewY[shooterPATId]}) / nextShooterPATIdsFour.size();
                action.deltaView8 = (Vec2{playerAtTick.viewX[nextShooterPATIdsEight.back()], playerAtTick.viewY[nextShooterPATIdsEight.back()]} -
                                    Vec2{playerAtTick.viewX[shooterPATId], playerAtTick.viewY[shooterPATId]}) / nextShooterPATIdsEight.size();
                action.nextFireTimeSeconds = secondsUntilNextFire;
                action.crouch = playerAtTick.isCrouching[nextPATId];
                action.walk = playerAtTick.isWalking[nextPATId];
                action.scope = playerAtTick.isScoped[nextPATId];
                action.newlyAirborne = playerAtTick.isAirborne[nextPATId] && !playerAtTick.isAirborne[shooterPATId];
                // filter out time steps with weird, way tooi great view angle changes
                // remove non-target events for now
                if (state.target.slotFilled) { // && std::abs(action.deltaView.x) <= 2. && std::abs(action.deltaView.y) <= 1) {
                    states.push_back(state);
                    actions.push_back(action);
                }
            }
        }
    }
}

EngagementResult queryEngagementDataset(const Equipment & equipment, const Games & games, const Rounds & rounds,
                                        const WeaponFire & weaponFire, const Hurt & hurt,
                                        const Ticks & ticks, const Players & players, const PlayerAtTick & playerAtTick) {
    int numThreads = omp_get_max_threads();
    vector<EngagementResult::TimeStepState> tmpStates[numThreads];
    vector<EngagementResult::TimeStepAction> tmpActions[numThreads];
    vector<int64_t> numNavAreas(numThreads, 0);

//#pragma omp parallel for
    for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
        if (rounds.gameId[roundIndex] > 1) {
            continue;
        }
        int threadNum = omp_get_thread_num();
        TickRates tickRates = computeTickRates(games, rounds, roundIndex);
        const int64_t RADIUS_GAME_TICKS = ENGAGEMENT_SECONDS_RADIUS * tickRates.gameTickRate;

        // first compute all the engagements
        // each engagement is ENGAGEMENT_SECONDS_RADIUS seconds before first shot (or round start, whichever later)
        // until ENGAGEMENT_SECONDS_RADIUS after last shot at same target (or death or round end or next engagement, whichever soonest)
        // can have multiple engagements if multiple targets
        // if shooting at a target, only switch to no target engagement after target engagement ends
        vector<EngagementIds> engagementIds;
        computeEngagementsPerRound(rounds, ticks, playerAtTick, weaponFire, hurt, roundIndex, engagementIds,
                                   RADIUS_GAME_TICKS, tickRates);

        map<int64_t, map<int64_t, vector<int64_t>>> tickToShooterToEngagementIds;
        for (int i = 0; i < engagementIds.size(); i++) {
            const EngagementIds &oneEngagementIds = engagementIds[i];
            // end on tick before last tick of engagement so for all ticks know alive for next one
            // when computing results
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

        computeEngagementResults(rounds, ticks, playerAtTick, roundIndex, engagementIds,
                                 tickToShooterToEngagementIds, shooterToWeaponFireGameTicks,
                                 RADIUS_GAME_TICKS, tickRates,tmpStates[threadNum], tmpActions[threadNum]);
    }


    EngagementResult result(equipment);
    for (int i = 0; i < numThreads; i++) {
        for (int j = 0; j < tmpStates[i].size(); j++) {
            result.tickId.push_back(tmpStates[i][j].tickId);
            result.roundId.push_back(tmpStates[i][j].roundId);
            result.sourcePlayerId.push_back(playerAtTick.playerId[tmpStates[i][j].shooterPatId]);
            result.sourcePlayerName.push_back(players.name[result.sourcePlayerId.back() + players.idOffset]);
            result.demoName.push_back(games.demoFile[rounds.gameId[tmpStates[i][j].roundId]]);
            result.states.push_back(tmpStates[i][j]);
            result.actions.push_back(tmpActions[i][j]);
        }
    }
    result.size = result.tickId.size();
    return result;
}
