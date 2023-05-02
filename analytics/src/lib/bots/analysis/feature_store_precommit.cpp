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

    void FeatureStorePreCommitBuffer::clearHistory() {
        historicalPlayerDataBuffer.clear();
    }

    void FeatureStorePreCommitBuffer::appendPlayerHistory() {
        std::map<int64_t, BTTeamPlayerData> newEntryHistoricalPlayerDataBuffer;
        for (const auto & playerData : btTeamPlayerData) {
            newEntryHistoricalPlayerDataBuffer[playerData.playerId] = playerData;
        }
        historicalPlayerDataBuffer.enqueue(newEntryHistoricalPlayerDataBuffer);
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