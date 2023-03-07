//
// Created by durst on 3/3/23.
//

#include <omp.h>
#include <map>
#include "bots/analysis/feature_store.h"

namespace csknow::feature_store {
    void FeatureStorePreCommitBuffer::updateFeatureStoreBufferPlayers(const ServerState & state) {
        tPlayerIdToIndex.clear();
        ctPlayerIdToIndex.clear();
        int tIndex = 0, ctIndex = 0;
        for (const auto & client : state.clients) {
            if (client.team == ENGINE_TEAM_T) {
                tPlayerIdToIndex[client.csgoId] = tIndex;
                tIndex++;
            }
            else if (client.team == ENGINE_TEAM_CT) {
                ctPlayerIdToIndex[client.csgoId] = ctIndex;
                ctIndex++;
            }
        }
    }

    void FeatureStorePreCommitBuffer::addEngagementPossibleEnemy(
        const EngagementPossibleEnemy & engagementPossibleEnemy) {
        engagementPossibleEnemyBuffer.push_back(engagementPossibleEnemy);
    }

    void FeatureStorePreCommitBuffer::addEngagementLabel(bool hitEngagement, bool visibleEngagement) {
        hitEngagementBuffer = hitEngagement;
        visibleEngagementBuffer = visibleEngagement;
    }

    void FeatureStorePreCommitBuffer::addTargetPossibleEnemyLabel(const TargetPossibleEnemyLabel & targetPossibleEnemyLabel) {
        targetPossibleEnemyLabelBuffer.push_back(targetPossibleEnemyLabel);
    }

    void FeatureStoreResult::init(size_t size) {
        roundId.resize(size, INVALID_ID);
        tickId.resize(size, INVALID_ID);
        playerId.resize(size, INVALID_ID);
        for (int i = 0; i < maxEnemies; i++) {
            columnEnemyData[i].playerId.resize(size, INVALID_ID);
            columnEnemyData[i].enemyEngagementStates.resize(size, EngagementEnemyState::None);
            columnEnemyData[i].timeSinceLastVisibleOrToBecomeVisible.resize(size, maxTimeToVis);
            columnEnemyData[i].worldDistanceToEnemy.resize(size, maxWorldDistance);
            columnEnemyData[i].crosshairDistanceToEnemy.resize(size, maxCrosshairDistance);
            columnEnemyData[i].nearestTargetEnemy.resize(size, false);
            columnEnemyData[i].hitTargetEnemy.resize(size, false);
        }
        hitEngagement.resize(size, false);
        visibleEngagement.resize(size, false);
        valid.resize(size, false);
        this->size = size;
    }

    FeatureStoreResult::FeatureStoreResult() {
        training = false;
        init(1);
    }

    FeatureStoreResult::FeatureStoreResult(size_t size) {
        training = true;
        init(size);
    }

    void FeatureStoreResult::commitRow(FeatureStorePreCommitBuffer & buffer, size_t rowIndex,
                                       int64_t roundIndex, int64_t tickIndex, int64_t playerIndex) {
        roundId[rowIndex] = roundIndex;
        tickId[rowIndex] = tickIndex;
        playerId[rowIndex] = playerIndex;

        if (buffer.engagementPossibleEnemyBuffer.size() > maxEnemies) {
            std::cerr << "committing row with wrong number of engagement players" << std::endl;
            throw std::runtime_error("committing row with wrong number of engagement players");
        }

        // fewer target labels than active players, so record those that exist and make rest non-target
        std::map<int64_t, TargetPossibleEnemyLabel> playerToTargetLabel;
        for (const auto targetLabel : buffer.targetPossibleEnemyLabelBuffer) {
            playerToTargetLabel[targetLabel.playerId] = targetLabel;
        }

        bool tEnemies = false;
        for (size_t i = 0; i < buffer.engagementPossibleEnemyBuffer.size(); i++) {
            int64_t curPlayerId = buffer.engagementPossibleEnemyBuffer[i].playerId;
            if (i == 0 && buffer.tPlayerIdToIndex.find(curPlayerId) != curPlayerId) {
                tEnemies = true;
            }
            size_t columnIndex = tEnemies ? buffer.tPlayerIdToIndex[curPlayerId] : buffer.ctPlayerIdToIndex[curPlayerId];
            columnEnemyData[columnIndex].playerId[rowIndex] = curPlayerId;
            columnEnemyData[columnIndex].enemyEngagementStates[rowIndex] =
                buffer.engagementPossibleEnemyBuffer[i].enemyState;
            columnEnemyData[columnIndex].timeSinceLastVisibleOrToBecomeVisible[rowIndex] =
                buffer.engagementPossibleEnemyBuffer[i].timeSinceLastVisibleOrToBecomeVisible;
            columnEnemyData[columnIndex].worldDistanceToEnemy[rowIndex] =
                buffer.engagementPossibleEnemyBuffer[i].worldDistanceToEnemy;
            columnEnemyData[columnIndex].crosshairDistanceToEnemy[rowIndex] =
                buffer.engagementPossibleEnemyBuffer[i].crosshairDistanceToEnemyHead;

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
        buffer.targetPossibleEnemyLabelBuffer.clear();
    }

    void FeatureStoreResult::toHDF5Inner(HighFive::File & file) {
        HighFive::DataSetCreateProps hdf5FlatCreateProps;
        hdf5FlatCreateProps.add(HighFive::Deflate(6));
        hdf5FlatCreateProps.add(HighFive::Chunking(columnEnemyData[0].crosshairDistanceToEnemy.size()));

        file.createDataSet("/data/round id", roundId, hdf5FlatCreateProps);
        file.createDataSet("/data/tick id", tickId, hdf5FlatCreateProps);
        file.createDataSet("/data/player id", playerId, hdf5FlatCreateProps);
        for (size_t i = 0; i < columnEnemyData.size(); i++) {
            string iStr = std::to_string(i);
            file.createDataSet("/data/enemy player id " + iStr, columnEnemyData[i].playerId, hdf5FlatCreateProps);
            file.createDataSet("/data/enemy engagement states " + iStr,
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
