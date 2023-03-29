//
// Created by durst on 3/3/23.
//

#include <omp.h>
#include <map>
#include "bots/analysis/feature_store.h"
#include "queries/lookback.h"
#include "file_helpers.h"
#include <atomic>
#include "circular_buffer.h"

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

    void FeatureStorePreCommitBuffer::addEngagementTeammate(const EngagementTeammate &engagementTeammate) {
        engagementTeammateBuffer.push_back(engagementTeammate);
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
            columnTeammateData[i].playerId.resize(size, INVALID_ID);
            columnTeammateData[i].teammateWorldDistance.resize(size, maxWorldDistance);
            columnTeammateData[i].crosshairDistanceToTeammate.resize(size, maxCrosshairDistance);
        }
        fireCurTick.resize(size, false);
        hitEngagement.resize(size, false);
        visibleEngagement.resize(size, false);
        nearestCrosshairCurTick.resize(size, maxEnemies);
        nearestCrosshairEnemy500ms.resize(size, maxEnemies);
        nearestCrosshairEnemy1s.resize(size, maxEnemies);
        nearestCrosshairEnemy2s.resize(size, maxEnemies);
        positionOffset2sUpToThreshold.resize(size, 1.);
        viewAngleOffset2sUpToThreshold.resize(size, 1.);
        negPositionOffset2sUpToThreshold.resize(size, 0.);
        negViewAngleOffset2sUpToThreshold.resize(size, 0.);
        for (int i = 0; i <= maxEnemies; i++) {
            pctNearestCrosshairEnemy2s[i].resize(size, 0.);
        }
        visibleEnemy2s.resize(size, 0.);
        negVisibleEnemy2s.resize(size, 1.);
        fireNext2s.resize(size, 0.);
        negFireNext2s.resize(size, 1.);
        for (size_t i = 0; i < numNearestEnemyState; i++) {
            pctNearestEnemyChange2s[i].resize(size, 0.);
        }
        nextPATId2s.resize(size, INVALID_ID);
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
                columnEnemyData[columnIndex].nearestTargetEnemy[rowIndex] = false;
                columnEnemyData[columnIndex].hitTargetEnemy[rowIndex] = false;
            }
            else {
                columnEnemyData[columnIndex].nearestTargetEnemy[rowIndex] =
                    playerToTargetLabel[curPlayerId].nearestTargetEnemy;
                columnEnemyData[columnIndex].hitTargetEnemy[rowIndex] = playerToTargetLabel[curPlayerId].hitTargetEnemy;
            }
        }

        for (size_t i = 0; i < buffer.engagementTeammateBuffer.size(); i++) {
            int64_t curPlayerId = buffer.engagementTeammateBuffer[i].playerId;
            size_t columnIndex = tEnemies ? buffer.ctPlayerIdToIndex[curPlayerId] : buffer.tPlayerIdToIndex[curPlayerId];
            columnTeammateData[columnIndex].teammateWorldDistance[rowIndex] =
                buffer.engagementTeammateBuffer[i].worldDistanceToTeammate;
            columnTeammateData[columnIndex].crosshairDistanceToTeammate[rowIndex] =
                buffer.engagementTeammateBuffer[i].crosshairDistanceToTeammateHead;
        }

        if (training) {
            hitEngagement[rowIndex] = buffer.hitEngagementBuffer;
            visibleEngagement[rowIndex] = buffer.visibleEngagementBuffer;
        }

        valid[rowIndex] = true;

        buffer.engagementPossibleEnemyBuffer.clear();
        buffer.engagementTeammateBuffer.clear();
        buffer.targetPossibleEnemyLabelBuffer.clear();
    }

    void FeatureStoreResult::computeAcausalLabels(const Games & games, const Rounds & rounds,
                                                  const Ticks & ticks, const PlayerAtTick & playerAtTick) {
        std::atomic<int64_t> roundsProcessed = 0;
        patId = playerAtTick.id;
#pragma omp parallel for
        for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
            TickRates tickRates = computeTickRates(games, rounds, roundIndex);
            int twoSecondTicks = secondsToDemoTicks(tickRates, 2.);
            // start at end and work backwards to compute future
            std::map<int64_t, std::map<int64_t, int64_t>> nextVisibleTickId;
            std::map<int64_t, std::map<int, int64_t>> playerToEnemyNumToNumTicksNearestCrosshair500ms;
            std::map<int64_t, std::map<int, int64_t>> playerToEnemyNumToNumTicksNearestCrosshair1s;
            std::map<int64_t, std::map<int, int64_t>> playerToEnemyNumToNumTicksNearestCrosshair2s;
            std::map<int64_t, std::map<int64_t, int>> playerToTickToNearestCrosshair;
            std::map<int64_t, CircularBuffer<double>> playerToTickToNearestWorldDistance;
            std::map<int64_t, CircularBuffer<double>> playerToTickToFireNext2s;
            std::map<int64_t, CircularBuffer<double>> playerToTickToVisibleEnemy2s;
            std::map<int64_t, std::map<int64_t, Vec3>> playerToTickToPos;
            std::map<int64_t, std::map<int64_t, Vec2>> playerToTickToViewAngle;
            std::map<int64_t, std::map<int64_t, int64_t>> tickToPlayerToPATId;
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
                //int64_t oldFutureTickIndex2s = futureTickIndex2s;
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
                    tickToPlayerToPATId[tickIndex][curPlayerId] = patIndex;

                    std::array<bool, maxEnemies+1> validEnemyNums{};
                    validEnemyNums[maxEnemies] = true;

                    double minCrosshairDistanceToEnemy = maxCrosshairDistance;
                    int nearestCrosshairEnemy = maxEnemies;
                    double minWorldDistanceToEnemy = maxWorldDistance;
                    bool visibleEnemyThisFrame = false;
                    for (size_t columnIndex = 0; columnIndex < maxEnemies; columnIndex++) {
                        int64_t enemyPlayerId = columnEnemyData[columnIndex].playerId[patIndex];
                        validEnemyNums[columnIndex] = enemyPlayerId != INVALID_ID;
                        if (!validEnemyNums[columnIndex]) {
                            continue;
                        }
                        // compute visibility for this frame
                        if (columnEnemyData[columnIndex].enemyEngagementStates[patIndex] == EngagementEnemyState::Visible) {
                            nextVisibleTickId[curPlayerId][enemyPlayerId] = tickIndex;
                            visibleEnemyThisFrame = true;
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
                        // these default to false, so no need for else clause setting them to false
                        // record nearest enemy for collection later
                        if (columnEnemyData[columnIndex].crosshairDistanceToEnemy[patIndex] < minCrosshairDistanceToEnemy) {
                            nearestCrosshairEnemy = columnIndex;
                            minCrosshairDistanceToEnemy =
                                columnEnemyData[columnIndex].crosshairDistanceToEnemy[patIndex];
                        }
                        if (columnEnemyData[columnIndex].worldDistanceToEnemy[patIndex] < minWorldDistanceToEnemy) {
                            minWorldDistanceToEnemy =
                                columnEnemyData[columnIndex].worldDistanceToEnemy[patIndex];
                        }
                    }

                    // add to rolling trackers
                    playerToTickToNearestCrosshair[curPlayerId][tickIndex] = nearestCrosshairEnemy;

                    // manage circular buffers
                    if (playerToTickToNearestWorldDistance.find(curPlayerId) == playerToTickToNearestWorldDistance.end()) {
                        playerToTickToNearestWorldDistance.insert({curPlayerId,
                                                                   CircularBuffer<double>(twoSecondTicks)});
                    }
                    playerToTickToNearestWorldDistance.at(curPlayerId).enqueue(minWorldDistanceToEnemy);
                    if (playerToTickToFireNext2s.find(curPlayerId) == playerToTickToFireNext2s.end()) {
                        playerToTickToFireNext2s.insert({curPlayerId,
                                                         CircularBuffer<double>(twoSecondTicks)});
                    }
                    playerToTickToFireNext2s.at(curPlayerId).enqueue(fireCurTick[patIndex] ? 1. : 0.);
                    if (playerToTickToVisibleEnemy2s.find(curPlayerId) == playerToTickToVisibleEnemy2s.end()) {
                        playerToTickToVisibleEnemy2s.insert({curPlayerId,
                                                         CircularBuffer<double>(twoSecondTicks)});
                    }
                    playerToTickToFireNext2s.at(curPlayerId).enqueue(visibleEnemyThisFrame ? 1. : 0.);

                    playerToTickToPos[curPlayerId][tickIndex] = {
                        playerAtTick.posX[patIndex], playerAtTick.posY[patIndex], playerAtTick.posZ[patIndex]
                    };
                    playerToTickToViewAngle[curPlayerId][tickIndex] = {
                        playerAtTick.viewX[patIndex], playerAtTick.viewY[patIndex]
                    };
                    if (playerToEnemyNumToNumTicksNearestCrosshair500ms[curPlayerId].find(nearestCrosshairEnemy) ==
                        playerToEnemyNumToNumTicksNearestCrosshair500ms[curPlayerId].end()) {
                        playerToEnemyNumToNumTicksNearestCrosshair500ms[curPlayerId][nearestCrosshairEnemy] = 0;
                    }
                    playerToEnemyNumToNumTicksNearestCrosshair500ms[curPlayerId][nearestCrosshairEnemy]++;
                    if (playerToEnemyNumToNumTicksNearestCrosshair1s[curPlayerId].find(nearestCrosshairEnemy) ==
                        playerToEnemyNumToNumTicksNearestCrosshair1s[curPlayerId].end()) {
                        playerToEnemyNumToNumTicksNearestCrosshair1s[curPlayerId][nearestCrosshairEnemy] = 0;
                    }
                    playerToEnemyNumToNumTicksNearestCrosshair1s[curPlayerId][nearestCrosshairEnemy]++;
                    if (playerToEnemyNumToNumTicksNearestCrosshair2s[curPlayerId].find(nearestCrosshairEnemy) ==
                        playerToEnemyNumToNumTicksNearestCrosshair2s[curPlayerId].end()) {
                        playerToEnemyNumToNumTicksNearestCrosshair2s[curPlayerId][nearestCrosshairEnemy] = 0;
                    }
                    playerToEnemyNumToNumTicksNearestCrosshair2s[curPlayerId][nearestCrosshairEnemy]++;

                    // subtract from rolling trackers
                    for (const auto & removedTick : removed500msTicks) {
                        int enemyToDecrease = playerToTickToNearestCrosshair[curPlayerId][removedTick];
                        playerToEnemyNumToNumTicksNearestCrosshair500ms[curPlayerId][enemyToDecrease]--;
                    }
                    for (const auto & removedTick : removed1sTicks) {
                        int enemyToDecrease = playerToTickToNearestCrosshair[curPlayerId][removedTick];
                        playerToEnemyNumToNumTicksNearestCrosshair1s[curPlayerId][enemyToDecrease]--;
                    }
                    for (const auto & removedTick : removed2sTicks) {
                        int enemyToDecrease = playerToTickToNearestCrosshair[curPlayerId][removedTick];
                        playerToEnemyNumToNumTicksNearestCrosshair2s[curPlayerId][enemyToDecrease]--;
                        // remove the tick from the specific tick trackers once it's past longest window
                        playerToTickToNearestCrosshair[curPlayerId].erase(removedTick);
                        playerToTickToPos[curPlayerId].erase(removedTick);
                        playerToTickToViewAngle[curPlayerId].erase(removedTick);
                        tickToPlayerToPATId.erase(removedTick);
                    }

                    // compute labels based on future data
                    nearestCrosshairCurTick[patIndex] = nearestCrosshairEnemy;
                    int64_t maxTicksNearest = 0;
                    int nearestEnemyOverWindow = maxEnemies;
                    for (const auto & [enemyNum, numTicksNearest] :
                        playerToEnemyNumToNumTicksNearestCrosshair500ms[curPlayerId]) {
                        if (numTicksNearest > maxTicksNearest && validEnemyNums[enemyNum]) {
                            maxTicksNearest = numTicksNearest;
                            nearestEnemyOverWindow = enemyNum;
                        }
                    }
                    nearestCrosshairEnemy500ms[patIndex] = nearestEnemyOverWindow;
                    maxTicksNearest = 0;
                    nearestEnemyOverWindow = maxEnemies;
                    for (const auto & [enemyNum, numTicksNearest] :
                        playerToEnemyNumToNumTicksNearestCrosshair1s[curPlayerId]) {
                        if (numTicksNearest > maxTicksNearest && validEnemyNums[enemyNum]) {
                            maxTicksNearest = numTicksNearest;
                            nearestEnemyOverWindow = enemyNum;
                        }
                    }
                    nearestCrosshairEnemy1s[patIndex] = nearestEnemyOverWindow;
                    maxTicksNearest = 0;
                    nearestEnemyOverWindow = maxEnemies;
                    int64_t totalNumTicksNearest = 0;
                    for (const auto & [enemyNum, numTicksNearest] :
                        playerToEnemyNumToNumTicksNearestCrosshair2s[curPlayerId]) {
                        if (validEnemyNums[enemyNum]) {
                            if (numTicksNearest > maxTicksNearest) {
                                maxTicksNearest = numTicksNearest;
                                nearestEnemyOverWindow = enemyNum;
                            }
                            totalNumTicksNearest += numTicksNearest;
                        }
                    }
                    nearestCrosshairEnemy2s[patIndex] = nearestEnemyOverWindow;
                    for (const auto & [enemyNum, numTicksNearest] :
                        playerToEnemyNumToNumTicksNearestCrosshair2s[curPlayerId]) {
                        if (validEnemyNums[enemyNum]) {
                            /*
                            if (futureTickIndex2s == tickIndex) {
                                if (enemyNum == maxEnemies) {
                                    pctNearestCrosshairEnemy2s[enemyNum][patIndex] = 1.;
                                }
                                else {
                                    pctNearestCrosshairEnemy2s[enemyNum][patIndex] = 0.;
                                }
                            }
                            else {
                             */
                                pctNearestCrosshairEnemy2s[enemyNum][patIndex] =
                                    static_cast<double>(numTicksNearest) / static_cast<double>(totalNumTicksNearest);
                                /*
                                if (totalNumTicksNearest == 0) {
                                    std::cout << "creating an infinity" << std::endl;
                                }
                                 */
                            //}
                        }
                    }

                    array<size_t, numNearestEnemyState> numNearestEnemyChange{0};
                    int64_t numNearestWorldDistanceTicks =
                        playerToTickToNearestWorldDistance.at(curPlayerId).getCurSize();
                    for (int64_t i = 0; i < numNearestWorldDistanceTicks; i++) {
                        double futureMinDistance = playerToTickToNearestWorldDistance.at(curPlayerId).fromNewest(i);
                        if (futureMinDistance < minWorldDistanceToEnemy - nearestEnemyChangeThreshold) {
                            numNearestEnemyChange[enumAsInt(NearestEnemyState::Decrease)]++;
                        }
                        else if (futureMinDistance > minWorldDistanceToEnemy - nearestEnemyChangeThreshold) {
                            numNearestEnemyChange[enumAsInt(NearestEnemyState::Increase)]++;
                        }
                        else {
                            numNearestEnemyChange[enumAsInt(NearestEnemyState::Constant)]++;
                        }
                    }
                    pctNearestEnemyChange2s[enumAsInt(NearestEnemyState::Decrease)][patIndex] =
                        static_cast<double>(numNearestEnemyChange[enumAsInt(NearestEnemyState::Decrease)]) /
                        static_cast<double>(numNearestWorldDistanceTicks);
                    pctNearestEnemyChange2s[enumAsInt(NearestEnemyState::Increase)][patIndex] =
                        static_cast<double>(numNearestEnemyChange[enumAsInt(NearestEnemyState::Increase)]) /
                        static_cast<double>(numNearestWorldDistanceTicks);
                    pctNearestEnemyChange2s[enumAsInt(NearestEnemyState::Constant)][patIndex] =
                        static_cast<double>(numNearestEnemyChange[enumAsInt(NearestEnemyState::Constant)]) /
                        static_cast<double>(numNearestWorldDistanceTicks);

                    bool firedIn2sWindow = false;
                    int64_t numFiredTicks = playerToTickToFireNext2s.at(curPlayerId).getCurSize();
                    for (int64_t i = 0; i < numFiredTicks; i++) {
                        if (playerToTickToFireNext2s.at(curPlayerId).fromNewest(i) > 0.) {
                            firedIn2sWindow = true;
                            break;
                        }
                    }
                    if (firedIn2sWindow) {
                        fireNext2s[patIndex] = 1.;
                        negFireNext2s[patIndex] = 0.;
                    }

                    double numTicksEnemyVisible = 0.;
                    int64_t numVisibleOptionTicks = playerToTickToVisibleEnemy2s.at(curPlayerId).getCurSize();
                    for (int64_t i = 0; i < numVisibleOptionTicks; i++) {
                        numTicksEnemyVisible += playerToTickToVisibleEnemy2s.at(curPlayerId).fromNewest(i);
                    }
                    visibleEnemy2s[patIndex] = numTicksEnemyVisible / numVisibleOptionTicks;
                    negVisibleEnemy2s[patIndex] = 1 - visibleEnemy2s[patIndex];



                    /*
                    if (tickIndex == 224486 && curPlayerId == 5) {
                        std::cout << "nearestCrosshairEnemy2s " << nearestCrosshairEnemy2s[patIndex]
                            << ", future tick index 2s " << futureTickIndex2s
                            << ", tick index " << tickIndex << ", total num ticks nearest " << totalNumTicksNearest;
                        std::cout << ", pct nearest ";
                        for (size_t enemyN = 0; enemyN < maxEnemies + 1; enemyN++) {
                            std::cout << pctNearestCrosshairEnemy2s[enemyN][patIndex] << ", ";
                        }
                        std::cout << ", valid enemy nums ";
                        for (size_t enemyN = 0; enemyN < maxEnemies + 1; enemyN++) {
                            std::cout << validEnemyNums[enemyN] << ", ";
                        }
                        std::cout << ", num ticks nearest crosshair ";
                        for (size_t enemyN = 0; enemyN < maxEnemies + 1; enemyN++) {
                            if (playerToEnemyNumToNumTicksNearestCrosshair2s[curPlayerId].find(enemyN) !=
                                playerToEnemyNumToNumTicksNearestCrosshair2s[curPlayerId].end()) {
                                std::cout << playerToEnemyNumToNumTicksNearestCrosshair2s[curPlayerId][enemyN] << ", ";
                            }
                            else {
                                std::cout << 0 << ", ";
                            }
                        }
                        std::cout << std::endl;
                    }
                     */

                    positionOffset2sUpToThreshold[patIndex] = std::min(maxPositionDelta,
                        computeDistance(playerToTickToPos[curPlayerId][tickIndex],
                                        playerToTickToPos[curPlayerId][futureTickIndex2s])) / maxPositionDelta;
                    viewAngleOffset2sUpToThreshold[patIndex] = std::min(maxViewAngleDelta,
                        computeDistance(playerToTickToPos[curPlayerId][tickIndex],
                                        playerToTickToPos[curPlayerId][futureTickIndex2s])) / maxViewAngleDelta;
                    negPositionOffset2sUpToThreshold[patIndex] = 1. - positionOffset2sUpToThreshold[patIndex];
                    negViewAngleOffset2sUpToThreshold[patIndex] = 1. - viewAngleOffset2sUpToThreshold[patIndex];
                    /*
                    if (tickToPlayerToPATId.find(futureTickIndex2s) == tickToPlayerToPATId.end() ||
                        tickToPlayerToPATId[futureTickIndex2s].find(curPlayerId) == tickToPlayerToPATId[futureTickIndex2s].end()) {
                        std::cout << "something is wrong" << std::endl;
                        std::cout << "old future tick index 2s: " << oldFutureTickIndex2s << std::endl;
                        std::cout << "new future tick index 2s: " << futureTickIndex2s << std::endl;
                        std::cout << "tick index: " << tickIndex << std::endl;
                        std::cout << "removed ticks: ";
                        for (const auto & removedTick : removed2sTicks) {
                            std::cout << removedTick << ",";
                        }
                        if (tickToPlayerToPATId.find(futureTickIndex2s) == tickToPlayerToPATId.end()) {
                            std::cout << "missing entire tick" << std::endl;
                        }
                        else {
                            std::cout << "missing player" << std::endl;
                        }
                        std::cout << std::endl;
                        exit(1);
                    }
                     */
                    // in one round, a player drops out, need to ignore that part of round
                    if (tickToPlayerToPATId.find(futureTickIndex2s) != tickToPlayerToPATId.end() &&
                        tickToPlayerToPATId[futureTickIndex2s].find(curPlayerId) != tickToPlayerToPATId[futureTickIndex2s].end()) {
                        nextPATId2s[patIndex] = tickToPlayerToPATId[futureTickIndex2s][curPlayerId];
                    }
                }
            }
            roundsProcessed++;
            printProgress(roundsProcessed, rounds.size);
        }
    }

    void clone(const FeatureStoreResult & src, FeatureStoreResult & dst, size_t srcIndex, size_t dstIndex,
               bool invalidSubset) {
        dst.roundId[dstIndex] = src.roundId[srcIndex];
        dst.playerId[dstIndex] = src.playerId[srcIndex];
        dst.valid[dstIndex] = src.valid[srcIndex];
        if (invalidSubset) {
            return;
        }
        dst.tickId[dstIndex] = src.tickId[srcIndex];
        dst.patId[dstIndex] = src.patId[srcIndex];
        for (size_t i = 0; i < dst.columnEnemyData.size(); i++) {
            dst.columnEnemyData[i].playerId[dstIndex] =
                src.columnEnemyData[i].playerId[srcIndex];
            dst.columnEnemyData[i].enemyEngagementStates[dstIndex] =
                src.columnEnemyData[i].enemyEngagementStates[srcIndex];
            dst.columnEnemyData[i].timeSinceLastVisibleOrToBecomeVisible[dstIndex] =
                src.columnEnemyData[i].timeSinceLastVisibleOrToBecomeVisible[srcIndex];
            dst.columnEnemyData[i].worldDistanceToEnemy[dstIndex] =
                src.columnEnemyData[i].worldDistanceToEnemy[srcIndex];
            dst.columnEnemyData[i].crosshairDistanceToEnemy[dstIndex] =
                src.columnEnemyData[i].crosshairDistanceToEnemy[srcIndex];
            dst.columnEnemyData[i].nearestTargetEnemy[dstIndex] =
                src.columnEnemyData[i].nearestTargetEnemy[srcIndex];
            dst.columnEnemyData[i].hitTargetEnemy[dstIndex] =
                src.columnEnemyData[i].hitTargetEnemy[srcIndex];
            dst.columnEnemyData[i].visibleIn1s[dstIndex] =
                src.columnEnemyData[i].visibleIn1s[srcIndex];
            dst.columnEnemyData[i].visibleIn2s[dstIndex] =
                src.columnEnemyData[i].visibleIn2s[srcIndex];
            dst.columnEnemyData[i].visibleIn5s[dstIndex] =
                src.columnEnemyData[i].visibleIn5s[srcIndex];
            dst.columnEnemyData[i].visibleIn10s[dstIndex] =
                src.columnEnemyData[i].visibleIn10s[srcIndex];
            dst.columnTeammateData[i].playerId[dstIndex] =
                src.columnTeammateData[i].playerId[srcIndex];
            dst.columnTeammateData[i].teammateWorldDistance[dstIndex] =
                src.columnTeammateData[i].teammateWorldDistance[srcIndex];
            dst.columnTeammateData[i].crosshairDistanceToTeammate[dstIndex] =
                src.columnTeammateData[i].crosshairDistanceToTeammate[srcIndex];
        }
        dst.hitEngagement[dstIndex] = src.hitEngagement[srcIndex];
        dst.visibleEngagement[dstIndex] = src.visibleEngagement[srcIndex];
        dst.nearestCrosshairCurTick[dstIndex] = src.nearestCrosshairCurTick[srcIndex];
        dst.nearestCrosshairEnemy500ms[dstIndex] = src.nearestCrosshairEnemy500ms[srcIndex];
        dst.nearestCrosshairEnemy1s[dstIndex] = src.nearestCrosshairEnemy1s[srcIndex];
        dst.nearestCrosshairEnemy2s[dstIndex] = src.nearestCrosshairEnemy2s[srcIndex];
        dst.positionOffset2sUpToThreshold[dstIndex] = src.positionOffset2sUpToThreshold[srcIndex];
        dst.viewAngleOffset2sUpToThreshold[dstIndex] = src.viewAngleOffset2sUpToThreshold[srcIndex];
        dst.negPositionOffset2sUpToThreshold[dstIndex] = src.negPositionOffset2sUpToThreshold[srcIndex];
        dst.negViewAngleOffset2sUpToThreshold[dstIndex] = src.negViewAngleOffset2sUpToThreshold[srcIndex];
        for (size_t i = 0; i < dst.pctNearestCrosshairEnemy2s.size(); i++) {
            dst.pctNearestCrosshairEnemy2s[i][dstIndex] = src.pctNearestCrosshairEnemy2s[i][srcIndex];
        }
        dst.fireNext2s[dstIndex] = src.fireNext2s[srcIndex];
        dst.negFireNext2s[dstIndex] = src.negFireNext2s[srcIndex];
        dst.visibleEnemy2s[dstIndex] = src.visibleEnemy2s[srcIndex];
        dst.negVisibleEnemy2s[dstIndex] = src.negVisibleEnemy2s[srcIndex];
        for (size_t i = 0; i < dst.pctNearestEnemyChange2s.size(); i++) {
            dst.pctNearestEnemyChange2s[i][dstIndex] = src.pctNearestEnemyChange2s[i][srcIndex];
        }
        dst.nextPATId2s[dstIndex] = src.nextPATId2s[srcIndex];
        if (dst.nextPATId2s[dstIndex] == 0) {
            std::cout << "bad things are happening" << std::endl;
        }
    }

    FeatureStoreResult FeatureStoreResult::makeWindows() const {
        size_t numValid = 0;
        for (size_t i = 0; i < static_cast<size_t>(size); i++) {
            if (valid[i] && tickId[i] % 10 == 0) {
                numValid++;
            }
        }

        FeatureStoreResult windowResult(numValid * windowSize);
        windowResult.patId.resize(numValid * windowSize);
        size_t windowResultIndex = 0;
        for (size_t i = 0; i < static_cast<size_t>(size); i++) {
            if (!valid[i] || tickId[i] % 10 != 0) {
                continue;
            }

            size_t patPointer = i;
            bool curWindowValid = true;
            for (size_t j = 0; j < windowSize; j++) {
                clone(*this, windowResult, patPointer, windowResultIndex, !curWindowValid);
                if (curWindowValid) {
                    if (nextPATId2s[patPointer] != INVALID_ID) {
                        patPointer = static_cast<size_t>(nextPATId2s[patPointer]);
                    }
                    else {
                        curWindowValid = false;
                    }
                }
                windowResultIndex++;
            }
        }

        return windowResult;
    }

    void FeatureStoreResult::toHDF5Inner(HighFive::File & file) {
        HighFive::DataSetCreateProps hdf5FlatCreateProps;
        hdf5FlatCreateProps.add(HighFive::Deflate(6));
        hdf5FlatCreateProps.add(HighFive::Chunking(columnEnemyData[0].crosshairDistanceToEnemy.size()));

        file.createDataSet("/data/round id", roundId, hdf5FlatCreateProps);
        file.createDataSet("/data/tick id", tickId, hdf5FlatCreateProps);
        file.createDataSet("/data/player id", playerId, hdf5FlatCreateProps);
        file.createDataSet("/data/player at tick id", patId, hdf5FlatCreateProps);
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
            file.createDataSet("/data/teammate player id " + iStr, columnTeammateData[i].playerId, hdf5FlatCreateProps);
            file.createDataSet("/data/teammate world distance " + iStr, columnTeammateData[i].teammateWorldDistance, hdf5FlatCreateProps);
            file.createDataSet("/data/crosshair distance to teammate " + iStr, columnTeammateData[i].crosshairDistanceToTeammate, hdf5FlatCreateProps);
        }
        file.createDataSet("/data/hit engagement", hitEngagement, hdf5FlatCreateProps);
        file.createDataSet("/data/visible engagement", visibleEngagement, hdf5FlatCreateProps);
        file.createDataSet("/data/nearest crosshair enemy 500ms", nearestCrosshairEnemy500ms, hdf5FlatCreateProps);
        file.createDataSet("/data/nearest crosshair enemy 1s", nearestCrosshairEnemy1s, hdf5FlatCreateProps);
        file.createDataSet("/data/nearest crosshair enemy 2s", nearestCrosshairEnemy2s, hdf5FlatCreateProps);
        file.createDataSet("/data/position offset 2s up to threshold", positionOffset2sUpToThreshold, hdf5FlatCreateProps);
        file.createDataSet("/data/view angle offset 2s up to threshold", viewAngleOffset2sUpToThreshold, hdf5FlatCreateProps);
        file.createDataSet("/data/neg position offset 2s up to threshold", negPositionOffset2sUpToThreshold, hdf5FlatCreateProps);
        file.createDataSet("/data/neg view angle offset 2s up to threshold", negViewAngleOffset2sUpToThreshold, hdf5FlatCreateProps);
        for (size_t i = 0; i <= maxEnemies; i++) {
            file.createDataSet("/data/pct nearest crosshair enemy 2s " + std::to_string(i),
                               pctNearestCrosshairEnemy2s[i], hdf5FlatCreateProps);
        }
        file.createDataSet("/data/visible enemy 2s", visibleEnemy2s, hdf5FlatCreateProps);
        file.createDataSet("/data/neg visible enemy 2s", negVisibleEnemy2s, hdf5FlatCreateProps);
        file.createDataSet("/data/fire next 2s", fireNext2s, hdf5FlatCreateProps);
        file.createDataSet("/data/neg fire next 2s", negFireNext2s, hdf5FlatCreateProps);
        vector<string> nearestEnemyStateNames{"decrease", "constant", "increase"};
        for (size_t i = 0; i < numNearestEnemyState; i++) {
            file.createDataSet("/data/pct nearest enemy change 2s " + nearestEnemyStateNames[i],
                               pctNearestEnemyChange2s[i], hdf5FlatCreateProps);
        }
        file.createDataSet("/data/next player at tick id 2s", nextPATId2s, hdf5FlatCreateProps);
        file.createDataSet("/data/valid", valid, hdf5FlatCreateProps);
    }

    void FeatureStoreResult::checkInvalid() const {
        for (int64_t i = 0; i < size; i++) {
            for (size_t j = 0; j < maxEnemies; j++) {
                if (enumAsInt(columnEnemyData[j].enemyEngagementStates[i]) < 0 ||
                    enumAsInt(columnEnemyData[j].enemyEngagementStates[i]) > 5) {
                    std::cout << "invalid " << i << std::endl;
                    exit(1);
                }
            }
        }
    }
}
