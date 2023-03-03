//
// Created by durst on 3/3/23.
//

#include "bots/analysis/feature_store.h"

namespace csknow::feature_store {
    FeatureStoreResult::FeatureStoreResult(bool training) : training(training) {
        if (!training) {
            for (int i = 0; i < maxEnemies; i++) {
                columnEnemyData[i].enemyEngagementStates.resize(1);
                columnEnemyData[i].timeSinceLastVisible.resize(1);
                columnEnemyData[i].timeToBecomeVisible.resize(1);
                columnEnemyData[i].worldDistanceToEnemy.resize(1);
                columnEnemyData[i].crosshairDistanceToEnemy.resize(1);
            }
            hitEngagement.resize(1);
            visibleEngagement.resize(1);
        }
    }

    void FeatureStoreResult::addEngagementPossibleEnemy(
        const EngagementPossibleEnemy & engagementPossibleEnemy) {
        engagementPossibleEnemyBuffer.push_back(engagementPossibleEnemy);
    }

    void FeatureStoreResult::addTargetPossibleEnemy(const TargetPossibleEnemy & targetPossibleEnemy) {
        targetPossibleEnemyBuffer.push_back(targetPossibleEnemy);
    }

    void FeatureStoreResult::addEngagementLabel(bool hitEngagement, bool visibleEngagement) {
        hitEngagementBuffer = hitEngagement;
        visibleEngagementBuffer = visibleEngagement;
    }

    void FeatureStoreResult::addTargetPossibleEnemyLabel(const TargetPossibleEnemyLabel & targetPossibleEnemyLabel) {
        targetPossibleEnemyLabelBuffer.push_back(targetPossibleEnemyLabel);
    }

    void FeatureStoreResult::commitRow() {
        std::sort(engagementPossibleEnemyBuffer.begin(), engagementPossibleEnemyBuffer.end(),
                  [](const EngagementPossibleEnemy & a, const EngagementPossibleEnemy & b) {
            return a.playerId < b.playerId;
        });
        std::sort(targetPossibleEnemyBuffer.begin(), targetPossibleEnemyBuffer.end(),
                  [](const TargetPossibleEnemy & a, const TargetPossibleEnemy & b) {
                      return a.playerId < b.playerId;
        });
        std::sort(targetPossibleEnemyLabelBuffer.begin(), targetPossibleEnemyLabelBuffer.end(),
                  [](const TargetPossibleEnemyLabel & a, const TargetPossibleEnemyLabel & b) {
                      return a.playerId < b.playerId;
        });

        if (engagementPossibleEnemyBuffer.size() != maxEnemies) {
            throw std::runtime_error("committing row with wrong number of engagement players");
        }
        if (targetPossibleEnemyBuffer.size() != maxEnemies) {
            throw std::runtime_error("committing row with wrong number of target players");
        }
        if (targetPossibleEnemyLabelBuffer.size() != maxEnemies) {
            throw std::runtime_error("committing row with wrong number of target player labels");
        }
        for (size_t i = 0; i < engagementPossibleEnemyBuffer.size(); i++) {
            if (engagementPossibleEnemyBuffer[i].playerId != targetPossibleEnemyBuffer[i].playerId ||
                engagementPossibleEnemyBuffer[i].playerId != targetPossibleEnemyLabelBuffer[i].playerId) {
                throw std::runtime_error("committing row with different player ids");
            }

            if (training) {
                columnEnemyData[i].playerId.push_back(engagementPossibleEnemyBuffer[i].playerId);
                columnEnemyData[i].enemyEngagementStates.push_back(engagementPossibleEnemyBuffer[i].enemyState);
                columnEnemyData[i].timeSinceLastVisible.push_back(engagementPossibleEnemyBuffer[i].timeSinceLastVisible);
                columnEnemyData[i].timeToBecomeVisible.push_back(engagementPossibleEnemyBuffer[i].timeToBecomeVisible);
                columnEnemyData[i].worldDistanceToEnemy.push_back(targetPossibleEnemyBuffer[i].worldDistanceToEnemy);
                columnEnemyData[i].crosshairDistanceToEnemy.push_back(targetPossibleEnemyBuffer[i].crosshairDistanceToEnemyHead);
                columnEnemyData[i].nearestTargetEnemy.push_back(targetPossibleEnemyLabelBuffer[i].nearestTargetEnemy);
                columnEnemyData[i].hitTargetEnemy.push_back(targetPossibleEnemyLabelBuffer[i].hitTargetEnemy);
            }
            else {
                columnEnemyData[i].playerId[0] = engagementPossibleEnemyBuffer[i].playerId;
                columnEnemyData[i].enemyEngagementStates[0] = engagementPossibleEnemyBuffer[i].enemyState;
                columnEnemyData[i].timeSinceLastVisible[0] = engagementPossibleEnemyBuffer[i].timeSinceLastVisible;
                columnEnemyData[i].timeToBecomeVisible[0] = engagementPossibleEnemyBuffer[i].timeToBecomeVisible;
                columnEnemyData[i].worldDistanceToEnemy[0] = targetPossibleEnemyBuffer[i].worldDistanceToEnemy;
                columnEnemyData[i].crosshairDistanceToEnemy[0] = targetPossibleEnemyBuffer[i].crosshairDistanceToEnemyHead;
                columnEnemyData[i].nearestTargetEnemy[0] = targetPossibleEnemyLabelBuffer[i].nearestTargetEnemy;
                columnEnemyData[i].hitTargetEnemy[0] = targetPossibleEnemyLabelBuffer[i].hitTargetEnemy;
            }
        }

        if (training) {
            hitEngagement.push_back(hitEngagementBuffer);
            visibleEngagement.push_back(visibleEngagementBuffer);
        }
        else {
            hitEngagement[0] = hitEngagementBuffer;
            visibleEngagement[0] = visibleEngagementBuffer;
        }
    }
}
