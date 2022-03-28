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
                    activeEngagementIds[shooterPlayerId][INVALID_ID].firstHurtTick = tickIndex;
                    activeEngagementIds[shooterPlayerId][INVALID_ID].lastHurtTick = tickIndex;
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
                    // new engagement
                    if (activeEngagementIds[shooterPlayerId].find(targetId) == activeEngagementIds[shooterPlayerId].end()) {
                        activeEngagementIds[shooterPlayerId][targetId].startTickId = tickIndex -
                                getLookbackDemoTick(rounds, ticks, playerAtTick, tickIndex, tickRates, RADIUS_GAME_TICKS, 1000);
                        activeEngagementIds[shooterPlayerId][targetId].endTickId = tickIndex +
                                getLookforwardDemoTick(rounds, ticks, playerAtTick, tickIndex, tickRates, RADIUS_GAME_TICKS, 1000);
                        activeEngagementIds[shooterPlayerId][targetId].firstHurtTick = tickIndex;
                        activeEngagementIds[shooterPlayerId][targetId].lastHurtTick = tickIndex;
                        activeEngagementIds[shooterPlayerId][targetId].shooterId = shooterPlayerId;
                        activeEngagementIds[shooterPlayerId][targetId].targetId = targetId;
                        activeEngagementIds[shooterPlayerId][targetId].numHits = 1;
                    }
                    // continuing old engagement
                    else {
                        activeEngagementIds[shooterPlayerId][targetId].endTickId = tickIndex +
                                getLookforwardDemoTick(rounds, ticks, playerAtTick, tickIndex, tickRates, RADIUS_GAME_TICKS, 1000);
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

void computeEngagementResults(const Rounds & rounds, const Ticks & ticks, const PlayerAtTick & playerAtTick, const int64_t roundId,
                              const vector<EngagementIds> & engagementIds, map<int64_t, map<int64_t, vector<int64_t>>> tickToShooterToEngagementIds,
                              vector<EngagementResult::TimeStepState> states, vector<EngagementResult::TimeStepAction> actions) {
    for (int64_t tickIndex = rounds.ticksPerRound[roundId].minId;
         tickIndex != -1 && tickIndex <= rounds.ticksPerRound[roundId].maxId; tickIndex++) {
        if (tickToShooterToEngagementIds.find(tickIndex) != tickToShooterToEngagementIds.end()) {
            for (const auto & [shooter, engagementIdIndices] : tickToShooterToEngagementIds[tickIndex]) {
                assert(!engagementIds.empty());

                // if more than 1 engagement for this shooter on this tick, ranking is
                // 1. actively shooting at target
                // 2. number of hits landed during engagement
                int64_t bestEngagement = 0;
                if (engagementIdIndices.size() > 1) {
                    int32_t maxHits = -1;
                    bool maxInShootingRange = false;
                    for (size_t i = 0; i < engagementIdIndices.size(); i++) {
                        int32_t curHits = engagementIds[engagementIdIndices[i]].numHits;
                        bool curInShootingRange = engagementIds[engagementIdIndices[0]].firstHurtTick <= tickIndex &&
                                engagementIds[engagementIdIndices[0]].lastHurtTick >= tickIndex;
                        if ((!maxInShootingRange && curInShootingRange) ||
                            (!(maxInShootingRange && !curInShootingRange) && maxHits < curHits)) {
                            maxHits = curHits;
                            maxInShootingRange = curInShootingRange;
                            bestEngagement = i;
                        }
                    }
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
            for (int64_t tickId = oneEngagementIds.startTickId; tickId <= oneEngagementIds.endTickId; tickId++) {
                tickToShooterToEngagementIds[tickId][oneEngagementIds.shooterId].push_back(i);
            }
        }

        computeEngagementResults(rounds, ticks, playerAtTick, roundIndex, engagementIds, tickToShooterToEngagementIds,
                                 tmpStates[threadNum], tmpActions[threadNum]);
    }


    EngagementResult result(equipment);
    for (int i = 0; i < numThreads; i++) {
        for (int j = 0; j < tmpStates[i].size(); j++) {
            result.tickId.push_back(tmpStates[i][j].tickId);
            result.roundId.push_back(tmpStates[i][j].roundId);
            result.sourcePlayerId.push_back(playerAtTick.playerId[tmpStates[i][j].patId]);
            result.sourcePlayerName.push_back(players.name[result.sourcePlayerId.back() + players.idOffset]);
            result.demoName.push_back(games.demoFile[tmpStates[i][j].gameId]);
            result.states.push_back(tmpStates[i][j]);
            result.actions.push_back(tmpActions[i][j]);
        }
    }
    result.size = result.tickId.size();
    return result;
}
