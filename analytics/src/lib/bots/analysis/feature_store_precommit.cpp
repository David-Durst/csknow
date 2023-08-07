//
// Created by durst on 4/6/23.
//
#include "bots/analysis/feature_store_precommit.h"

namespace csknow::feature_store {
    void FeatureStorePreCommitBuffer::updateFeatureStoreBufferPlayers(const ServerState &state) {
        tPlayerIdToIndex.clear();
        ctPlayerIdToIndex.clear();
        int tIndex = 0, ctIndex = 0;
        for (const auto &client: state.clients) {
            if (client.team == ENGINE_TEAM_T) {
                tPlayerIdToIndex[client.csgoId] = tIndex;
                tIndex++;
            } else if (client.team == ENGINE_TEAM_CT) {
                ctPlayerIdToIndex[client.csgoId] = ctIndex;
                ctIndex++;
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
        for (const auto & hurtEvent : state.hurtEvents) {
            // filter out non-line of sight hurt events
            if (sourcemodNonGunWeaponNames.count(hurtEvent.weapon) == 0 &&
                demoNonGunWeaponNames.count(hurtEvent.weapon) == 0) {
                victimsThisTick.insert(hurtEvent.victimId);
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

        set<CSGOId> enemiesVisible;
        for (const auto & client : state.clients) {
            if (state.getVisibleEnemies(client.csgoId, true).size() > 0) {
                enemiesVisible.insert(client.csgoId);
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

            if (attackersThisTick.count(playerId) > 0) {
                playerCounter.ticksSinceFire = 0;
            }
            else {
                playerCounter.ticksSinceFire = std::min(playerCounter.ticksSinceFire + 1, default_many_ticks);
            }

            if (enemiesVisible.count(playerId) > 0) {
                playerCounter.ticksSinceEnemyVisible = 0;
            }
            else {
                playerCounter.ticksSinceEnemyVisible =
                        std::min(playerCounter.ticksSinceEnemyVisible + 1, default_many_ticks);
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
                if (!victim.isAlive) {
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
    /*
     */

    void FeatureStorePreCommitBuffer::updateCurTeamData(const ServerState & state, const nav_mesh::nav_file & navFile) {
        updatePlayerTickCounters(state);
        map<CSGOId, double> nearestCrosshairDistanceToEnemy = getNearestCrosshairDistanceToEnemy(state);
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
            btTeamPlayerData.push_back({client.csgoId, client.team, curAreaId, curAreaIndex,
                                        client.getCurrentViewAngles(),
                                        client.getFootPosForPlayer(), client.getVelocity(),
                                        nearestCrosshairDistanceToEnemy[client.csgoId],
                                        client.health, client.armor,
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