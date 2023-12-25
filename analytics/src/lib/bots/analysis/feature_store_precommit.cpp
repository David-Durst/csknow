//
// Created by durst on 4/6/23.
//
#include "bots/analysis/feature_store_precommit.h"

namespace csknow::feature_store {
    void FeatureStorePreCommitBuffer::updateFeatureStoreBufferPlayers(const ServerState &state, bool newRound) {
        // keep ids constant as much as possible inside rounds as players may join and leave, don't want to shuffle
        // as alive players will never leave and rejoin alive, will rejoin dead and come back next round
        // when continuity irrelevant
        set<int> usedTIndices, usedCTIndices;
        if (newRound) {
            tPlayerIdToIndex.clear();
            ctPlayerIdToIndex.clear();
        }
        else {
            // remove players who left server in last frame
            vector<int64_t> tPlayerIdsToRemove;
            for (const auto & [csgoId, tIndex] : tPlayerIdToIndex) {
                if (state.csgoIds.count(csgoId)) {
                    usedTIndices.insert(tIndex);
                }
                else {
                    tPlayerIdsToRemove.push_back(csgoId);
                }
            }
            for (const auto & tPlayerId : tPlayerIdsToRemove) {
                tPlayerIdToIndex.erase(tPlayerId);
            }

            vector<int64_t> ctPlayerIdsToRemove;
            for (const auto & [csgoId, ctIndex] : ctPlayerIdToIndex) {
                if (state.csgoIds.count(csgoId)) {
                    usedTIndices.insert(ctIndex);
                }
                else {
                    ctPlayerIdsToRemove.push_back(csgoId);
                }
            }
            for (const auto & ctPlayerId : ctPlayerIdsToRemove) {
                ctPlayerIdToIndex.erase(ctPlayerId);
            }
        }

        for (const auto &client: state.clients) {
            if (client.team == ENGINE_TEAM_T) {
                if (!tPlayerIdToIndex.count(client.csgoId)) {
                    for (int tIndex = 0; tIndex < max_enemies; tIndex++) {
                        if (!usedTIndices.count(tIndex)) {
                            tPlayerIdToIndex[client.csgoId] = tIndex;
                            usedTIndices.insert(tIndex);
                            break;
                        }
                    }
                }
            }
            else if (client.team == ENGINE_TEAM_CT) {
                if (!ctPlayerIdToIndex.count(client.csgoId)) {
                    for (int ctIndex = 0; ctIndex < max_enemies; ctIndex++) {
                        if (!usedCTIndices.count(ctIndex)) {
                            ctPlayerIdToIndex[client.csgoId] = ctIndex;
                            usedCTIndices.insert(ctIndex);
                            break;
                        }
                    }
                }
            }
        }
    }

    void FeatureStorePreCommitBuffer::addEngagementPossibleEnemy(
        const EngagementPossibleEnemy &engagementPossibleEnemy) {
        engagementPossibleEnemyBuffer.push_back(engagementPossibleEnemy);
    }

    void FeatureStorePreCommitBuffer::addEngagementLabel(bool hitEngagement, bool visibleEngagement) {
        hitEngagementBuffer = hitEngagement;
        visibleEngagementBuffer = visibleEngagement;
    }

    void
    FeatureStorePreCommitBuffer::addTargetPossibleEnemyLabel(const TargetPossibleEnemyLabel &targetPossibleEnemyLabel) {
        targetPossibleEnemyLabelBuffer.push_back(targetPossibleEnemyLabel);
    }

    void FeatureStorePreCommitBuffer::addEngagementTeammate(const EngagementTeammate &engagementTeammate) {
        engagementTeammateBuffer.push_back(engagementTeammate);
    }

    void FeatureStorePreCommitBuffer::updatePlayerTickCounters(const ServerState & state) {
        // collect state for counter updates
        set<CSGOId> victimsThisTick;
        set<CSGOId> attackersWhoHitEnemiesThisTick;
        for (const auto & hurtEvent : state.hurtEvents) {
            // filter out non-line of sight hurt events
            if (sourcemodNonGunWeaponNames.count(hurtEvent.weapon) == 0 &&
                demoNonGunWeaponNames.count(hurtEvent.weapon) == 0) {
                victimsThisTick.insert(hurtEvent.victimId);
                if (state.getClient(hurtEvent.victimId).team != state.getClient(hurtEvent.attackerId).team) {
                    attackersWhoHitEnemiesThisTick.insert(hurtEvent.attackerId);
                }
            }
        }

        set<CSGOId> attackersThisTick;
        for (const auto & weaponFireEvent : state.weaponFireEvents) {
            // filter out non-line of sight hurt events
            if (sourcemodNonGunWeaponNames.count(weaponFireEvent.weapon) == 0 &&
                demoNonGunWeaponNames.count(weaponFireEvent.weapon) == 0) {
                attackersThisTick.insert(weaponFireEvent.shooter);
            }
        }

        set<CSGOId> noFOVEnemiesVisible;
        set<CSGOId> fovEnemiesVisible;
        for (const auto & client : state.clients) {
            bool a = false, b = false;
            if (!state.getVisibleEnemies(client.csgoId, false).empty()) {
                noFOVEnemiesVisible.insert(client.csgoId);
                a = true;
            }
            if (!state.getVisibleEnemies(client.csgoId, true).empty()) {
                fovEnemiesVisible.insert(client.csgoId);
                b = true;
            }
            if (!a && b) {
                std::cout << "bad" << std::endl;
            }
        }

        // fill in tick counters for players that don't have them
        // and remove those for players that don't need them
        set<CSGOId> alivePlayers;
        for (const auto & client : state.clients) {
            if (client.isAlive && playerTickCounters.count(client.csgoId) == 0) {
                playerTickCounters[client.csgoId] = default_tick_counters;
            }
            if (client.isAlive) {
                alivePlayers.insert(client.csgoId);
            }
        }
        vector<CSGOId> playerCountersToRemove;
        for (const auto & [playerId, playerCounter] : playerTickCounters) {
            if (alivePlayers.count(playerId) == 0) {
                playerCountersToRemove.push_back(playerId);
            }
        }
        for (const auto & playerId : playerCountersToRemove) {
            playerTickCounters.erase(playerId);
        }

        // now update all counters
        for (auto & [playerId, playerCounter] : playerTickCounters) {
            if (victimsThisTick.count(playerId) > 0) {
                playerCounter.ticksSinceHurt = 0;
            }
            else {
                playerCounter.ticksSinceHurt = std::min(playerCounter.ticksSinceHurt + 1, default_many_ticks);
            }

            if (attackersWhoHitEnemiesThisTick.count(playerId) > 0) {
                playerCounter.ticksSinceHitEnemy = 0;
            }
            else {
                playerCounter.ticksSinceHitEnemy = std::min(playerCounter.ticksSinceHitEnemy + 1, default_many_ticks);
            }

            if (attackersThisTick.count(playerId) > 0) {
                playerCounter.ticksSinceFire = 0;
            }
            else {
                playerCounter.ticksSinceFire = std::min(playerCounter.ticksSinceFire + 1, default_many_ticks);
            }

            if (noFOVEnemiesVisible.count(playerId) > 0) {
                playerCounter.ticksSinceNoFOVEnemyVisible = 0;
            }
            else {
                playerCounter.ticksSinceNoFOVEnemyVisible =
                        std::min(playerCounter.ticksSinceNoFOVEnemyVisible + 1, default_many_ticks);
            }

            if (fovEnemiesVisible.count(playerId) > 0) {
                playerCounter.ticksSinceFOVEnemyVisible = 0;
            }
            else {
                playerCounter.ticksSinceFOVEnemyVisible =
                        std::min(playerCounter.ticksSinceFOVEnemyVisible + 1, default_many_ticks);
            }
        }
    }

    map<CSGOId, double> getNearestCrosshairDistanceToEnemy(const ServerState & state) {
        map<CSGOId, double> result;
        for (const auto & attacker : state.clients) {
            if (!attacker.isAlive) {
                continue;
            }
            result[attacker.csgoId] = std::numeric_limits<double>::max();
            for (const auto & victim : state.clients) {
                if (!victim.isAlive || victim.team == attacker.team) {
                    continue;
                }
                Vec3 victimHeadPos = getCenterHeadCoordinatesForPlayer(victim.getFootPosForPlayer(), victim.getCurrentViewAngles(), victim.duckAmount);
                Vec2 deltaViewAngle = deltaViewFromOriginToDest(attacker.getEyePosForPlayer(), victimHeadPos, attacker.getCurrentViewAngles());
                double newDeltaViewAngle = computeMagnitude(deltaViewAngle);
                result[attacker.csgoId] = std::min(result[attacker.csgoId], newDeltaViewAngle);
            }
        }
        return result;
    }

    map<CSGOId, double> getNearestWorldDistanceToTeammateOrEnemy(const ServerState & state, bool toEnemy) {
        map<CSGOId, double> result;
        for (const auto & player : state.clients) {
            if (!player.isAlive) {
                continue;
            }
            result[player.csgoId] = std::numeric_limits<double>::max();
            for (const auto & other : state.clients) {
                bool teamCondition;
                if (toEnemy) {
                    teamCondition = player.team != other.team;
                }
                else {
                    teamCondition = player.team == other.team;
                }
                if (!other.isAlive || !teamCondition) {
                    continue;
                }
                result[player.csgoId] = std::min(result[player.csgoId], computeDistance(player.getFootPosForPlayer(),
                                                                                        other.getFootPosForPlayer()));
            }
        }
        return result;
    }

    void FeatureStorePreCommitBuffer::updateCurTeamData(const ServerState & state, const nav_mesh::nav_file & navFile) {
        updatePlayerTickCounters(state);
        map<CSGOId, double> nearestCrosshairDistanceToEnemy = getNearestCrosshairDistanceToEnemy(state);
        map<CSGOId, double> nearestWorldDistanceToEnemy = getNearestWorldDistanceToTeammateOrEnemy(state, true);
        map<CSGOId, double> nearestWorldDistanceToTeammate = getNearestWorldDistanceToTeammateOrEnemy(state, false);
        btTeamPlayerData.clear();

        for (const auto & client : state.clients) {
            if (!client.isAlive) {
                continue;
            }
            AreaId curAreaId = navFile
                    .get_nearest_area_by_position(vec3Conv(client.getFootPosForPlayer()))
                    .get_id();
            //AreaId curAreaId = nearestNavCell.getNearestArea(client.getFootPosForPlayer());
            int64_t curAreaIndex = navFile.m_area_ids_to_indices.at(curAreaId);
            if (curAreaIndex == -1) {
                std::cout << "bad pos " << client.getFootPosForPlayer().toString() << std::endl;
            }
            btTeamPlayerData.push_back({client.csgoId, client.team, curAreaId, curAreaIndex,
                                        client.getCurrentViewAngles(),
                                        client.getFootPosForPlayer(), client.getVelocity(),
                                        nearestCrosshairDistanceToEnemy[client.csgoId],
                                        nearestWorldDistanceToEnemy[client.csgoId],
                                        nearestWorldDistanceToTeammate[client.csgoId],
                                        client.health, client.armor, client.hasHelmet,
                                        static_cast<EngineWeaponId>(client.currentWeaponId),
                                        client.isScoped, client.isAirborne, client.isWalking, client.duckKeyPressed});
        }
        appendPlayerHistory();
        AreaId c4AreaId = navFile
                .get_nearest_area_by_position(vec3Conv(state.getC4Pos()))
                .get_id();
        int64_t c4AreaIndex = navFile.m_area_ids_to_indices.at(c4AreaId);
        c4MapData = {
                state.getC4Pos(),
                state.c4IsPlanted,
                state.ticksSinceLastPlant,
                c4AreaId,
                c4AreaIndex
        };
    }

    void FeatureStorePreCommitBuffer::appendPlayerHistory() {
        std::map<int64_t, BTTeamPlayerData> newEntryHistoricalPlayerDataBuffer;
        for (const auto & playerData : btTeamPlayerData) {
            newEntryHistoricalPlayerDataBuffer[playerData.playerId] = playerData;
        }
        historicalPlayerDataBuffer.enqueue(newEntryHistoricalPlayerDataBuffer);
    }

    void FeatureStorePreCommitBuffer::clearHistory() {
        historicalPlayerDataBuffer.clear();
        // needed since I carry over tick counters between frames
        btTeamPlayerData.clear();
        playerTickCounters.clear();
    }

    int64_t FeatureStorePreCommitBuffer::getPlayerOldestContiguousHistoryIndex(int64_t playerId) {
        int64_t result = INVALID_ID;
        for (int64_t i = 0; i < historicalPlayerDataBuffer.getCurSize(); i++) {
            if (historicalPlayerDataBuffer.fromNewest(i).find(playerId) !=
                historicalPlayerDataBuffer.fromNewest(i).end()) {
                result = i;
            }
            else {
                break;
            }
        }
        if (result == INVALID_ID) {
            throw std::runtime_error("player " + std::to_string(playerId) + " not in history with cur size " +
                std::to_string(historicalPlayerDataBuffer.getCurSize()));
        }
        return result;
    }
}