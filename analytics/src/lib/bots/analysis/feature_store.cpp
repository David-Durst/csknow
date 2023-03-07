//
// Created by durst on 3/3/23.
//

#include <omp.h>
#include <map>
#include "bots/analysis/feature_store.h"
#include "queries/lookback.h"

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
            columnEnemyData[i].visibleIn1s.resize(size, false);
            columnEnemyData[i].visibleIn2s.resize(size, false);
            columnEnemyData[i].visibleIn5s.resize(size, false);
            columnEnemyData[i].visibleIn10s.resize(size, false);
        }
        hitEngagement.resize(size, false);
        visibleEngagement.resize(size, false);
        nearestCrosshairEnemy500ms.resize(size, false);
        nearestCrosshairEnemy1s.resize(size, false);
        nearestCrosshairEnemy2s.resize(size, false);
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
            if (i == 0 && buffer.tPlayerIdToIndex.find(curPlayerId) != buffer.tPlayerIdToIndex.end()) {
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

    void FeatureStoreResult::computeAcausalLabels(const Games & games, const Rounds & rounds,
                                                  const Ticks & ticks, const PlayerAtTick & playerAtTick) {
//#pragma omp parallel for
        for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
            TickRates tickRates = computeTickRates(games, rounds, roundIndex);
            // start at end and work backwards to compute future
            std::map<int64_t, std::map<int64_t, int64_t>> nextVisibleTickId;
            std::map<int64_t, std::map<int, int64_t>> playerToEnemyNumToNumTicksNearestCrosshair500ms;
            std::map<int64_t, std::map<int, int64_t>> playerToEnemyNumToNumTicksNearestCrosshair1s;
            std::map<int64_t, std::map<int, int64_t>> playerToEnemyNumToNumTicksNearestCrosshair2s;
            std::map<int64_t, std::map<int64_t, int>> playerToTickToNearest;
            int64_t futureTickIndex500ms = rounds.ticksPerRound[roundIndex].maxId,
                futureTickIndex1s = rounds.ticksPerRound[roundIndex].maxId,
                futureTickIndex2s = rounds.ticksPerRound[roundIndex].maxId;
            for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].maxId;
                 tickIndex >= rounds.ticksPerRound[roundIndex].minId; tickIndex--) {
                vector<int64_t> removed500msTicks;
                while (secondsBetweenTicks(ticks, tickRates, tickIndex, futureTickIndex500ms) > 0.5) {
                    removed500msTicks.push_back(futureTickIndex500ms);
                    futureTickIndex500ms--;
                }
                vector<int64_t> removed1sTicks;
                while (secondsBetweenTicks(ticks, tickRates, tickIndex, futureTickIndex1s) > 1.) {
                    removed1sTicks.push_back(futureTickIndex1s);
                    futureTickIndex1s--;
                }
                vector<int64_t> removed2sTicks;
                while (secondsBetweenTicks(ticks, tickRates, tickIndex, futureTickIndex2s) > 2.) {
                    removed2sTicks.push_back(futureTickIndex2s);
                    futureTickIndex2s--;
                }

                for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
                     patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
                    if (!valid[patIndex]) {
                        continue;
                    }
                    const int64_t & curPlayerId = playerAtTick.playerId[patIndex];

                    double minCrosshairDistanceToEnemy = maxCrosshairDistance;
                    int nearestEnemy = maxEnemies;
                    for (size_t columnIndex = 0; columnIndex < maxEnemies; columnIndex++) {
                        int64_t enemyPlayerId = columnEnemyData[columnIndex].playerId[patIndex];
                        // compute visibility for this frame
                        if (columnEnemyData[columnIndex].enemyEngagementStates[patIndex] == EngagementEnemyState::Visible) {
                            nextVisibleTickId[curPlayerId][enemyPlayerId] = tickIndex;
                        }
                        // update visibiity for this and future frames
                        if (nextVisibleTickId[curPlayerId].find(enemyPlayerId) != nextVisibleTickId[curPlayerId].end()) {
                            double timeUntilVisible = secondsBetweenTicks(ticks, tickRates, tickIndex,
                                                                          nextVisibleTickId[curPlayerId][enemyPlayerId]);
                            columnEnemyData[columnIndex].visibleIn1s[patIndex] = timeUntilVisible < 1.;
                            columnEnemyData[columnIndex].visibleIn2s[patIndex] = timeUntilVisible < 2.;
                            columnEnemyData[columnIndex].visibleIn5s[patIndex] = timeUntilVisible < 5.;
                            columnEnemyData[columnIndex].visibleIn10s[patIndex] = timeUntilVisible < 10.;
                        }
                        // record nearest enemy for collection later
                        if (columnEnemyData[columnIndex].crosshairDistanceToEnemy[patIndex] < minCrosshairDistanceToEnemy) {
                            nearestEnemy = columnIndex;
                            minCrosshairDistanceToEnemy =
                                columnEnemyData[columnIndex].crosshairDistanceToEnemy[patIndex];
                        }
                    }

                    // add to rolling trackers
                    playerToTickToNearest[curPlayerId][tickIndex] = nearestEnemy;
                    if (playerToEnemyNumToNumTicksNearestCrosshair500ms[curPlayerId].find(nearestEnemy) ==
                        playerToEnemyNumToNumTicksNearestCrosshair500ms[curPlayerId].end()) {
                        playerToEnemyNumToNumTicksNearestCrosshair500ms[curPlayerId][nearestEnemy] = 0;
                    }
                    playerToEnemyNumToNumTicksNearestCrosshair500ms[curPlayerId][nearestEnemy]++;
                    if (playerToEnemyNumToNumTicksNearestCrosshair1s[curPlayerId].find(nearestEnemy) ==
                        playerToEnemyNumToNumTicksNearestCrosshair1s[curPlayerId].end()) {
                        playerToEnemyNumToNumTicksNearestCrosshair1s[curPlayerId][nearestEnemy] = 0;
                    }
                    playerToEnemyNumToNumTicksNearestCrosshair1s[curPlayerId][nearestEnemy]++;
                    if (playerToEnemyNumToNumTicksNearestCrosshair2s[curPlayerId].find(nearestEnemy) ==
                        playerToEnemyNumToNumTicksNearestCrosshair2s[curPlayerId].end()) {
                        playerToEnemyNumToNumTicksNearestCrosshair2s[curPlayerId][nearestEnemy] = 0;
                    }
                    playerToEnemyNumToNumTicksNearestCrosshair2s[curPlayerId][nearestEnemy]++;

                    // subtract from rolling trackers
                    for (const auto & removedTick : removed500msTicks) {
                        int enemyToDecrease = playerToTickToNearest[curPlayerId][removedTick];
                        playerToEnemyNumToNumTicksNearestCrosshair500ms[curPlayerId][enemyToDecrease]--;
                    }
                    for (const auto & removedTick : removed1sTicks) {
                        int enemyToDecrease = playerToTickToNearest[curPlayerId][removedTick];
                        playerToEnemyNumToNumTicksNearestCrosshair1s[curPlayerId][enemyToDecrease]--;
                    }
                    for (const auto & removedTick : removed2sTicks) {
                        int enemyToDecrease = playerToTickToNearest[curPlayerId][removedTick];
                        playerToEnemyNumToNumTicksNearestCrosshair2s[curPlayerId][enemyToDecrease]--;
                        // remove the tick from the specific tick trackers once it's past longest window
                        playerToTickToNearest[curPlayerId].erase(removedTick);
                    }

                    // compute labels based on future data
                    nearestCrosshairCurTick[patIndex] = nearestEnemy;
                    int64_t maxTicksNearest = 0;
                    int nearestEnemyOverWindow = maxEnemies;
                    for (const auto & [enemyNum, numTicksNearest] :
                        playerToEnemyNumToNumTicksNearestCrosshair500ms[curPlayerId]) {
                        if (numTicksNearest > maxTicksNearest) {
                            maxTicksNearest = numTicksNearest;
                            nearestEnemyOverWindow = enemyNum;
                        }
                    }
                    nearestCrosshairEnemy500ms[patIndex] = nearestEnemyOverWindow;
                    maxTicksNearest = 0;
                    nearestEnemyOverWindow = maxEnemies;
                    for (const auto & [enemyNum, numTicksNearest] :
                        playerToEnemyNumToNumTicksNearestCrosshair1s[curPlayerId]) {
                        if (numTicksNearest > maxTicksNearest) {
                            maxTicksNearest = numTicksNearest;
                            nearestEnemyOverWindow = enemyNum;
                        }
                    }
                    nearestCrosshairEnemy1s[patIndex] = nearestEnemyOverWindow;
                    maxTicksNearest = 0;
                    nearestEnemyOverWindow = maxEnemies;
                    for (const auto & [enemyNum, numTicksNearest] :
                        playerToEnemyNumToNumTicksNearestCrosshair2s[curPlayerId]) {
                        if (numTicksNearest > maxTicksNearest) {
                            maxTicksNearest = numTicksNearest;
                            nearestEnemyOverWindow = enemyNum;
                        }
                    }
                    nearestCrosshairEnemy2s[patIndex] = nearestEnemyOverWindow;
                }
            }
        }
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
            file.createDataSet("/data/visible in 1s " + iStr,
                               columnEnemyData[i].visibleIn1s, hdf5FlatCreateProps);
            file.createDataSet("/data/visible in 2s " + iStr,
                               columnEnemyData[i].visibleIn2s, hdf5FlatCreateProps);
            file.createDataSet("/data/visible in 5s " + iStr,
                               columnEnemyData[i].visibleIn5s, hdf5FlatCreateProps);
            file.createDataSet("/data/visible in 10s " + iStr,
                               columnEnemyData[i].visibleIn10s, hdf5FlatCreateProps);
        }
        file.createDataSet("/data/hit engagement", hitEngagement, hdf5FlatCreateProps);
        file.createDataSet("/data/visible engagement", visibleEngagement, hdf5FlatCreateProps);
        file.createDataSet("/data/nearest crosshair enemy 500ms", nearestCrosshairEnemy500ms, hdf5FlatCreateProps);
        file.createDataSet("/data/nearest crosshair enemy 1s", nearestCrosshairEnemy1s, hdf5FlatCreateProps);
        file.createDataSet("/data/nearest crosshair enemy 2s", nearestCrosshairEnemy2s, hdf5FlatCreateProps);
    }
}
