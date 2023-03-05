//
// Created by durst on 3/3/23.
//

#include <omp.h>
#include <map>
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
            columnEnemyData[i].enemyEngagementStates.resize(size, EngagementEnemyState::None);
            columnEnemyData[i].timeSinceLastVisibleOrToBecomeVisible.resize(size, maxTimeToVis);
            columnEnemyData[i].worldDistanceToEnemy.resize(size, INVALID_ID);
            columnEnemyData[i].crosshairDistanceToEnemy.resize(size, INVALID_ID);
        }
        hitEngagement.resize(size, false);
        visibleEngagement.resize(size, false);
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

        // fewer target labels than active players, so record those that exist and make rest non-target
        std::map<int64_t, TargetPossibleEnemyLabel> playerToTargetLabel;
        for (const auto targetLabel : buffer.targetPossibleEnemyLabelBuffer) {
            playerToTargetLabel[targetLabel.playerId] = targetLabel;
        }

        for (size_t i = 0; i < buffer.engagementPossibleEnemyBuffer.size(); i++) {
            if (buffer.engagementPossibleEnemyBuffer[i].playerId != buffer.targetPossibleEnemyBuffer[i].playerId) {
                throw std::runtime_error("committing row with different player ids");
            }

            int64_t curPlayerId = buffer.engagementPossibleEnemyBuffer[i].playerId;
            columnEnemyData[i].playerId[rowIndex] = curPlayerId;
            columnEnemyData[i].enemyEngagementStates[rowIndex] =
                buffer.engagementPossibleEnemyBuffer[i].enemyState;
            columnEnemyData[i].timeSinceLastVisibleOrToBecomeVisible[rowIndex] =
                buffer.engagementPossibleEnemyBuffer[i].timeSinceLastVisibleOrToBecomeVisible;
            columnEnemyData[i].worldDistanceToEnemy[rowIndex] =
                buffer.targetPossibleEnemyBuffer[i].worldDistanceToEnemy;
            columnEnemyData[i].crosshairDistanceToEnemy[rowIndex] =
                buffer.targetPossibleEnemyBuffer[i].crosshairDistanceToEnemyHead;

            if (playerToTargetLabel.find(curPlayerId) == playerToTargetLabel.end()) {
                columnEnemyData[i].nearestTargetEnemy[rowIndex] = false;
                columnEnemyData[i].hitTargetEnemy[rowIndex] = false;
            }
            else {
                columnEnemyData[i].nearestTargetEnemy[rowIndex] =
                    playerToTargetLabel[curPlayerId].nearestTargetEnemy;
                columnEnemyData[i].hitTargetEnemy[rowIndex] = playerToTargetLabel[curPlayerId].hitTargetEnemy;
            }
        }

        if (training) {
            hitEngagement[rowIndex] = buffer.hitEngagementBuffer;
            visibleEngagement[rowIndex] = buffer.visibleEngagementBuffer;
        }

        valid[rowIndex] = true;

        buffer.engagementPossibleEnemyBuffer.clear();
        buffer.targetPossibleEnemyBuffer.clear();
        buffer.targetPossibleEnemyLabelBuffer.clear();
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
            file.createDataSet("/data/time since last visible or to become visible " + iStr,
                               columnEnemyData[i].timeSinceLastVisibleOrToBecomeVisible, hdf5FlatCreateProps);
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
