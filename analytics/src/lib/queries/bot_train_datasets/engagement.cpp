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
    int64_t shooterId;
    int64_t targetId;
};

void
computeEngagementsPerRound(const Rounds & rounds, const Ticks & ticks, const PlayerAtTick & playerAtTick,
                           const WeaponFire & weaponFire, const Hurt & hurt, const int64_t roundId,
                           vector<EngagementIds> & engagementIds, const int64_t RADIUS_GAME_TICKS,
                           const TickRates & tickRates) {
    // track remaining engagement time if no key events occur
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
            for (const auto & fireId : ticks.weaponFirePerTick.at(tickIndex)) {
                if (weaponFire.shooter[fireId] == shooterPlayerId) {
                    engagementFireId = fireId;
                    break;
                }
            }
            // if didn't shoot, then continue as can't change engagement state if didn't shoot
            if (engagementFireId == INVALID_ID) {
                continue;
            }

            // get if player hurt anyone
            vector<int64_t> engagementHurtIds;
            vector<int64_t> engagementTargetIds;
            for (const auto & hurtId : ticks.hurtPerTick.at(tickIndex)) {
                // ignoring nade damage, so have to use same weapon as fired
                if (hurt.attacker[hurtId] == shooterPlayerId &&
                    hurt.weapon[hurtId] == weaponFire.weapon[engagementFireId]) {
                    engagementHurtIds.push_back(hurtId);
                    engagementTargetIds.push_back(hurt.victim[hurtId]);
                }
            }

            // if shot but didn't hit anyone, only start/continue engagement is no active engagement with a target
            if (engagementHurtIds.empty()) {
                bool haveOtherEngagements = false;

                // if no engagements, then add a new one, no target to check
                if (activeEngagementIds[shooterPlayerId].empty()) {
                    activeEngagementIds[shooterPlayerId][INVALID_ID].startTickId = tickIndex -
                            getLookbackDemoTick(rounds, ticks, playerAtTick, tickIndex, tickRates, RADIUS_GAME_TICKS, 1000);
                    activeEngagementIds[shooterPlayerId][INVALID_ID].endTickId = tickIndex +
                            getLookforwardDemoTick(rounds, ticks, playerAtTick, tickIndex, tickRates, RADIUS_GAME_TICKS, 1000);
                    activeEngagementIds[shooterPlayerId][INVALID_ID].shooterId = shooterPlayerId;
                    activeEngagementIds[shooterPlayerId][INVALID_ID].targetId = INVALID_ID;
                }
                // else if have engagements, if only one without a target, then extend it
                else if (activeEngagementIds[shooterPlayerId].size() == 1 &&
                         activeEngagementIds[shooterPlayerId].find(INVALID_ID) != activeEngagementIds[shooterPlayerId].end()){
                    activeEngagementIds[shooterPlayerId][INVALID_ID].endTickId = tickIndex +
                            getLookforwardDemoTick(rounds, ticks, playerAtTick, tickIndex, tickRates, RADIUS_GAME_TICKS, 1000);
                }
            }
            // if shot someone, then start/continue those engagements
            else {
                for (size_t i = 0; i < engagementHurtIds.size(); i++) {
                    int64_t targetId = engagementTargetIds[i];
                    // new engagement
                    if (activeEngagementIds[shooterPlayerId].find(targetId) == activeEngagementIds[shooterPlayerId].end()) {
                        activeEngagementIds[shooterPlayerId][targetId].startTickId = tickIndex -
                                getLookbackDemoTick(rounds, ticks, playerAtTick, tickIndex, tickRates, RADIUS_GAME_TICKS, 1000);
                        activeEngagementIds[shooterPlayerId][targetId].endTickId = tickIndex +
                                getLookforwardDemoTick(rounds, ticks, playerAtTick, tickIndex, tickRates, RADIUS_GAME_TICKS, 1000);
                        activeEngagementIds[shooterPlayerId][targetId].shooterId = shooterPlayerId;
                        activeEngagementIds[shooterPlayerId][targetId].targetId = targetId;
                    }
                    // continuing old engagement
                    else {
                        activeEngagementIds[shooterPlayerId][targetId].endTickId = tickIndex +
                                getLookforwardDemoTick(rounds, ticks, playerAtTick, tickIndex, tickRates, RADIUS_GAME_TICKS, 1000);
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

void
computeEngagementsPerRound(const Rounds & rounds, const Ticks & ticks, const PlayerAtTick & playerAtTick,
                           const WeaponFire & weaponFire, const Hurt & hurt, const int64_t roundId,
                           vector<EngagementIds> & engagementIds, const int64_t RADIUS_GAME_TICKS,
                           const TickRates & tickRates) {

EngagementResult queryEngagementDataset(const Equipment & equipment, const Games & games, const Rounds & rounds,
                                        const WeaponFire & weaponFire, const Hurt & hurt,
                                        const Ticks & ticks, const Players & players, const PlayerAtTick & playerAtTick) {
    int numThreads = omp_get_max_threads();
    vector<NextNavmeshResult::TimeStepState> tmpState[numThreads];
    vector<NextNavmeshResult::TimeStepPlan> tmpPlan[numThreads];
    vector<int64_t> numNavAreas(numThreads, 0);

//#pragma omp parallel for
    for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
        if (strcmp(games.mapName[rounds.gameId[roundIndex]], games.mapName[0]) != 0) {
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


        // track who is in
        int64_t lastTickInDataset = rounds.ticksPerRound[roundIndex].minId;
        for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
             tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {

            // only store every DECISION_SECONDS, not continually making a decision
            // make sure at least 2 DECISION_SECONDS from start so can look into future
            if (secondsBetweenTicks(ticks, tickRates, lastTickInDataset, tickIndex) < DECISION_SECONDS ||
                secondsBetweenTicks(ticks, tickRates, rounds.ticksPerRound[roundIndex].minId, tickIndex) < 2*DECISION_SECONDS) {
                continue;
            }
            lastTickInDataset = tickIndex;

            int64_t nextDemoTickId = tickIndex;
            int64_t curDemoTickId = tickIndex - getLookbackDemoTick(rounds, ticks, playerAtTick, tickIndex, tickRates, lookbackGameTicks, 1000);
            int64_t lastDemoTickId = curDemoTickId - 1;
            int64_t oldDemoTickId = tickIndex - getLookbackDemoTick(rounds, ticks, playerAtTick, tickIndex, tickRates, 2 * lookbackGameTicks, 1000);
            auto nextStepStates = addStepStatesForTick(ticks, playerAtTick, rounds.gameId[roundIndex], roundIndex, nextDemoTickId,
                                                       navFile, defaultTimeStepState);
            auto curStepStates = addStepStatesForTick(ticks, playerAtTick, rounds.gameId[roundIndex], roundIndex, curDemoTickId,
                                                      navFile, defaultTimeStepState);
            auto lastStepStates = addStepStatesForTick(ticks, playerAtTick, rounds.gameId[roundIndex], roundIndex, lastDemoTickId,
                                                       navFile, defaultTimeStepState);
            auto oldStepStates = addStepStatesForTick(ticks, playerAtTick, rounds.gameId[roundIndex], roundIndex, oldDemoTickId,
                                                      navFile, defaultTimeStepState);

            // players could leave in between frames, don't store any situations where all clients aren't same
            if (nextStepStates.size() == curStepStates.size() && nextStepStates.size() == lastStepStates.size() &&
                nextStepStates.size() == oldStepStates.size()) {
                bool samePlayersInSamePositions = true;
                for (size_t i = 0; i < nextStepStates.size(); i++) {
                    if (playerAtTick.playerId[nextStepStates[i].patId] != playerAtTick.playerId[curStepStates[i].patId] ||
                        playerAtTick.playerId[nextStepStates[i].patId] != playerAtTick.playerId[lastStepStates[i].patId] ||
                        playerAtTick.playerId[nextStepStates[i].patId] != playerAtTick.playerId[oldStepStates[i].patId]) {
                        samePlayersInSamePositions = false;
                        break;
                    }
                }

                if (samePlayersInSamePositions) {
                    tmpNextState[threadNum].insert(tmpNextState[threadNum].end(), nextStepStates.begin(), nextStepStates.end());
                    tmpCurState[threadNum].insert(tmpCurState[threadNum].end(), curStepStates.begin(), curStepStates.end());
                    tmpLastState[threadNum].insert(tmpLastState[threadNum].end(), lastStepStates.begin(), lastStepStates.end());
                    tmpOldState[threadNum].insert(tmpOldState[threadNum].end(), oldStepStates.begin(), oldStepStates.end());
                }
            }
        }

        for (int64_t planIndex = planStartIndex; planIndex < tmpCurState[threadNum].size(); planIndex++) {
            NextNavmeshResult::TimeStepPlan plan;


            plan.deltaX = tmpNextState[threadNum][planIndex].pos.x - tmpCurState[threadNum][planIndex].pos.x;
            plan.deltaY = tmpNextState[threadNum][planIndex].pos.y - tmpCurState[threadNum][planIndex].pos.y;
            double deltaZ = tmpNextState[threadNum][planIndex].pos.z - tmpCurState[threadNum][planIndex].pos.z;

            size_t curAreaId = tmpCurState[threadNum][planIndex].curArea,
                    nextAreaId = tmpNextState[threadNum][planIndex].curArea;
            Vec3 curAreaCenter(vec3tConv(navFile.m_areas[curAreaId].get_center()));
            Vec2 movementDir{plan.deltaX, plan.deltaY},
                    curAreaDir{curAreaCenter.x - tmpCurState[threadNum][planIndex].pos.x,
                               curAreaCenter.y - tmpCurState[threadNum][planIndex].pos.y};
            Ray movementRay(tmpCurState[threadNum][planIndex].pos, {movementDir.x, movementDir.y, deltaZ});
            if (curAreaId != nextAreaId) {
                plan.navTargetArea = nextAreaId;
            }
            else if (computeDistance(tmpCurState[threadNum][planIndex].pos, curAreaCenter) >= MIN_DISTANCE_TO_NAV_POINT &&
                     cosineSimilarity(movementDir, curAreaDir) > COSINE_SIMILARITY_THRESHOLD) {
                plan.navTargetArea = curAreaId;
            }
            else {
                const nav_mesh::nav_area & startArea = navFile.m_areas[curAreaId];
                auto& areaConnections = startArea.get_connections( );
                // default to not hitting anything, so just stay in current area
                // this captures not moving
                uint32_t nearestArea = curAreaId;
                double nearestDistance = std::numeric_limits<double>::max();
                for ( auto& connection : areaConnections ) {
                    auto &connection_area = navFile.get_area_by_id(connection.id);
                    // create some z
                    AABB areaAABB{vec3tConv(connection_area.m_nw_corner), vec3tConv(connection_area.m_se_corner)};
                    // create some extra vertical space for too small AABB
                    areaAABB.min.z -= 5.;
                    areaAABB.max.z += 5.;
                    double newT0, newT1;
                    if(intersectP(areaAABB, movementRay, newT0, newT1) &&
                       newT0 * computeMagnitude(movementRay.dir) < nearestDistance) {
                        nearestArea = navFile.m_area_ids_to_indices.at(connection.id);
                        nearestDistance = newT0 * computeMagnitude(movementRay.dir);
                    }
                }
                plan.navTargetArea = nearestArea;
            }

            plan.shootDuringNextThink = playerAtTick.primaryBulletsClip[tmpNextState[threadNum][planIndex].patId] !=
                                        playerAtTick.primaryBulletsClip[tmpCurState[threadNum][planIndex].patId];
            plan.crouchDuringNextThink = playerAtTick.isCrouching[tmpNextState[threadNum][planIndex].patId];

            tmpPlan[threadNum].push_back(plan);
        }
    }

    NextNavmeshResult result;
    result.numNavAreas = 0;
    for (int i = 0; i < numThreads; i++) {
        result.numNavAreas = std::max(result.numNavAreas, numNavAreas[i]);
        for (int j = 0; j < tmpCurState[i].size(); j++) {
            result.tickId.push_back(tmpCurState[i][j].tickId);
            result.roundId.push_back(tmpCurState[i][j].roundId);
            result.sourcePlayerId.push_back(playerAtTick.playerId[tmpCurState[i][j].patId]);
            result.sourcePlayerName.push_back(players.name[result.sourcePlayerId.back() + players.idOffset]);
            result.demoName.push_back(games.demoFile[tmpCurState[i][j].gameId]);
            result.curState.push_back(tmpCurState[i][j]);
            result.lastState.push_back(tmpLastState[i][j]);
            result.oldState.push_back(tmpOldState[i][j]);
            result.plan.push_back(tmpPlan[i][j]);
        }
    }
    result.size = result.tickId.size();
    return result;
}
