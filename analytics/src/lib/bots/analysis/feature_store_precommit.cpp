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

}