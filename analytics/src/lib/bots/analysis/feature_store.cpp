//
// Created by durst on 3/3/23.
//

#include "bots/analysis/feature_store.h"

namespace csknow::feature_store {
    void FeatureStoreResult::init(size_t size) {
        for (int i = 0; i < maxEnemies; i++) {
            columnEnemyData[i].enemyEngagementStates.resize(size);
            columnEnemyData[i].timeSinceLastVisible.resize(size);
            columnEnemyData[i].timeToBecomeVisible.resize(size);
            columnEnemyData[i].worldDistanceToEnemy.resize(size);
            columnEnemyData[i].crosshairDistanceToEnemy.resize(size);
        }
        hitEngagement.resize(size);
        visibleEngagement.resize(size);
    }

    FeatureStoreResult::FeatureStoreResult() {
        training = false;
        init(1);
    }

    FeatureStoreResult::FeatureStoreResult(size_t size) {
        training = true;
        init(size);
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
        if (training && targetPossibleEnemyLabelBuffer.size() != maxEnemies) {
            throw std::runtime_error("committing row with wrong number of target player labels");
        }
        for (size_t i = 0; i < engagementPossibleEnemyBuffer.size(); i++) {
            if (engagementPossibleEnemyBuffer[i].playerId != targetPossibleEnemyBuffer[i].playerId ||
                (training && engagementPossibleEnemyBuffer[i].playerId != targetPossibleEnemyLabelBuffer[i].playerId)) {
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
            }
        }

        if (training) {
            hitEngagement.push_back(hitEngagementBuffer);
            visibleEngagement.push_back(visibleEngagementBuffer);
        }
    }

    void FeatureStoreResult::toHDF5Inner(HighFive::File & file) {
        HighFive::DataSetCreateProps hdf5FlatCreateProps;
        hdf5FlatCreateProps.add(HighFive::Deflate(6));
        hdf5FlatCreateProps.add(HighFive::Chunking(columnEnemyData[0].crosshairDistanceToEnemy.size()));

        for (size_t i = 0; i < columnEnemyData.size(); i++) {
            string iStr = std::to_string(i);
            file.createDataSet("/data/player id " + iStr, columnEnemyData[i].playerId, hdf5FlatCreateProps);
            file.createDataSet("/data/enemy engagement states" + iStr,
                               vectorOfEnumsToVectorOfInts(columnEnemyData[i].enemyEngagementStates),
                               hdf5FlatCreateProps);
            file.createDataSet("/data/time since last visible " + iStr,
                               columnEnemyData[i].timeSinceLastVisible, hdf5FlatCreateProps);
            file.createDataSet("/data/time to become visible " + iStr,
                               columnEnemyData[i].timeToBecomeVisible, hdf5FlatCreateProps);
            file.createDataSet("/data/world distance to enemy " + iStr,
                               columnEnemyData[i].worldDistanceToEnemy, hdf5FlatCreateProps);
            file.createDataSet("/data/crosshair distance to enemy " + iStr,
                               columnEnemyData[i].crosshairDistanceToEnemy, hdf5FlatCreateProps);
            file.createDataSet("/data/nearest target enemy " + iStr,
                               columnEnemyData[i].nearestTargetEnemy, hdf5FlatCreateProps);
            file.createDataSet("/data/hit target enemy " + iStr,
                               columnEnemyData[i].hitTargetEnemy, hdf5FlatCreateProps);
        }
        file.createDataSet("/data/hit engagement", hitEngagement, hdf5FlatCreateProps);
        file.createDataSet("/data/visible engagement", visibleEngagement, hdf5FlatCreateProps);
    }
}
