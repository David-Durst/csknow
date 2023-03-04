//
// Created by durst on 3/3/23.
//

#include <omp.h>
#include "bots/analysis/feature_store.h"

namespace csknow::feature_store {
    void FeatureStorePreCommitBuffer::addEngagementPossibleEnemy(
        const EngagementPossibleEnemy & engagementPossibleEnemy) {
        engagementPossibleEnemyBuffer.push_back(engagementPossibleEnemy);
    }

    void FeatureStorePreCommitBuffer::addTargetPossibleEnemy(const TargetPossibleEnemy & targetPossibleEnemy) {
        targetPossibleEnemyBuffer.push_back(targetPossibleEnemy);
    }

    void FeatureStorePreCommitBuffer::addEngagementLabel(bool hitEngagement, bool visibleEngagement) {
        hitEngagementBuffer = hitEngagement;
        visibleEngagementBuffer = visibleEngagement;
    }

    void FeatureStorePreCommitBuffer::addTargetPossibleEnemyLabel(const TargetPossibleEnemyLabel & targetPossibleEnemyLabel) {
        targetPossibleEnemyLabelBuffer.push_back(targetPossibleEnemyLabel);
    }

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
        valid.resize(size, false);
    }

    FeatureStoreResult::FeatureStoreResult() {
        training = false;
        init(1);
    }

    FeatureStoreResult::FeatureStoreResult(size_t size) {
        training = true;
        init(size);
    }

    void FeatureStoreResult::commitRow(FeatureStorePreCommitBuffer & buffer, size_t rowIndex) {
        std::sort(buffer.engagementPossibleEnemyBuffer.begin(), buffer.engagementPossibleEnemyBuffer.end(),
                  [](const EngagementPossibleEnemy & a, const EngagementPossibleEnemy & b) {
            return a.playerId < b.playerId;
        });
        std::sort(buffer.targetPossibleEnemyBuffer.begin(), buffer.targetPossibleEnemyBuffer.end(),
                  [](const TargetPossibleEnemy & a, const TargetPossibleEnemy & b) {
                      return a.playerId < b.playerId;
        });
        std::sort(buffer.targetPossibleEnemyLabelBuffer.begin(), buffer.targetPossibleEnemyLabelBuffer.end(),
                  [](const TargetPossibleEnemyLabel & a, const TargetPossibleEnemyLabel & b) {
                      return a.playerId < b.playerId;
        });

        if (buffer.engagementPossibleEnemyBuffer.size() != maxEnemies) {
            throw std::runtime_error("committing row with wrong number of engagement players");
        }
        if (buffer.targetPossibleEnemyBuffer.size() != maxEnemies) {
            throw std::runtime_error("committing row with wrong number of target players");
        }
        if (training && buffer.targetPossibleEnemyLabelBuffer.size() != maxEnemies) {
            throw std::runtime_error("committing row with wrong number of target player labels");
        }
        for (size_t i = 0; i < buffer.engagementPossibleEnemyBuffer.size(); i++) {
            if (buffer.engagementPossibleEnemyBuffer[i].playerId != buffer.targetPossibleEnemyBuffer[i].playerId ||
                (training && buffer.engagementPossibleEnemyBuffer[i].playerId != buffer.targetPossibleEnemyLabelBuffer[i].playerId)) {
                throw std::runtime_error("committing row with different player ids");
            }

            columnEnemyData[i].playerId[rowIndex] = buffer.engagementPossibleEnemyBuffer[i].playerId;
            columnEnemyData[i].enemyEngagementStates[rowIndex] =
                buffer.engagementPossibleEnemyBuffer[i].enemyState;
            columnEnemyData[i].timeSinceLastVisible[rowIndex] =
                buffer.engagementPossibleEnemyBuffer[i].timeSinceLastVisible;
            columnEnemyData[i].timeToBecomeVisible[rowIndex] =
                buffer.engagementPossibleEnemyBuffer[i].timeToBecomeVisible;
            columnEnemyData[i].worldDistanceToEnemy[rowIndex] =
                buffer.targetPossibleEnemyBuffer[i].worldDistanceToEnemy;
            columnEnemyData[i].crosshairDistanceToEnemy[rowIndex] =
                buffer.targetPossibleEnemyBuffer[i].crosshairDistanceToEnemyHead;
        }

        if (training) {
            hitEngagement[rowIndex] = buffer.hitEngagementBuffer;
            visibleEngagement[rowIndex] = buffer.visibleEngagementBuffer;
        }

        valid[rowIndex] = true;
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
