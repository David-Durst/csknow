//
// Created by durst on 5/18/21.
//
#include "queries/a_cat_peekers.h"
#include "geometry.h"
#include "indices/spotted.h"
#include <omp.h>
#include <set>
#include <map>
#include <limits>
#include <algorithm>
using std::string;
using std::map;

ACatPeekers queryACatPeekers(const Rounds & rounds, const Ticks & ticks, const PlayerAtTick & playerAtTick) {
    int numThreads = omp_get_max_threads();
    vector<int64_t> tmpRoundId[numThreads];
    vector<int64_t> tmpPlayerAtTickId[numThreads];
    vector<int64_t> tmpPlayerId[numThreads];
    vector<double> tmpPosX[numThreads];
    vector<double> tmpPosY[numThreads];
    vector<double> tmpPosZ[numThreads];
    vector<double> tmpViewX[numThreads];
    vector<double> tmpViewY[numThreads];
    vector<int64_t> tmpWallId[numThreads];
    vector<double> tmpWallX[numThreads];
    vector<double> tmpWallY[numThreads];
    vector<double> tmpWallZ[numThreads];

    AABB aCatPositions{{217.0, 1327.34, -5.0}, {513.0, 2260.0, std::numeric_limits<double>::max()}};
    AABB frontCatWall{{507.0, 1353.0, std::numeric_limits<double>::min()}, {507.1, 2003.0, std::numeric_limits<double>::max()}};
    AABB oppositeElevatorWall{{507.0, 2003.0, std::numeric_limits<double>::min()}, {1215.0, 2003.1, std::numeric_limits<double>::max()}};
    AABB backCatWall{{224.0, 1301, std::numeric_limits<double>::min()}, {224.1, 2833.0, std::numeric_limits<double>::max()}};
    AABB ninjaWall{{224.0, 2833.0, std::numeric_limits<double>::min()}, {1010.0, 2833.1, std::numeric_limits<double>::max()}};
    AABB gooseWall{{1010.0, 2833.0, std::numeric_limits<double>::min()}, {1010.1, 3090.0, std::numeric_limits<double>::max()}};
    AABB topRampWall{{1010.0, 3090.0, std::numeric_limits<double>::min()}, {1588.0, 3090.1, std::numeric_limits<double>::max()}};
    AABB longWall{{1588.0, -1247.0, std::numeric_limits<double>::min()}, {1588.0, 3090.1, std::numeric_limits<double>::max()}};

    // these are wrapping containers to catch stray rays
    AABB leftContainer{{-2253.0, -900.0, std::numeric_limits<double>::min()}, {-2253.1, 3168.0, std::numeric_limits<double>::max()}};
    AABB topContainer{{-2253.0, 3168.0, std::numeric_limits<double>::min()}, {1897.0, 3168.1, std::numeric_limits<double>::max()}};
    AABB bottomContainer{{-2253.0, -900.0, std::numeric_limits<double>::min()}, {1897.0, -900.1, std::numeric_limits<double>::max()}};
    AABB rightContainer{{1897.0, -900.0, std::numeric_limits<double>::min()}, {1897.0, 3168.1, std::numeric_limits<double>::max()}};
    vector<AABB> walls{frontCatWall, oppositeElevatorWall, backCatWall, ninjaWall, gooseWall, topRampWall, longWall,
                       leftContainer, topContainer, bottomContainer, rightContainer};
    ACatPeekers result(walls);

#pragma omp parallel for
    for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
        int threadNum = omp_get_thread_num();
        // assuming first position is less than first kills
        for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
             tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {
            for (int64_t patIndex = ticks.patPerTick[tickIndex].minId; patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
                Vec3 playerPosition = {playerAtTick.posX[tickIndex], playerAtTick.posY[tickIndex], playerAtTick.posZ[tickIndex]};
                if (pointInRegion(aCatPositions, playerPosition)) {
                    tmpRoundId[threadNum].push_back(roundIndex);
                    tmpPlayerAtTickId[threadNum].push_back(playerAtTick.id[tickIndex]);
                    tmpPlayerId[threadNum].push_back(playerAtTick.playerId[tickIndex]);
                    tmpPosX[threadNum].push_back(playerAtTick.posX[tickIndex]);
                    tmpPosY[threadNum].push_back(playerAtTick.posY[tickIndex]);
                    tmpPosZ[threadNum].push_back(playerAtTick.posZ[tickIndex]);
                    tmpViewX[threadNum].push_back(playerAtTick.viewX[tickIndex]);
                    tmpViewY[threadNum].push_back(playerAtTick.viewY[tickIndex]);
                    bool hitAWall = false;
                    for (int wallIndex = 0; wallIndex < walls.size(); wallIndex++) {
                        Ray playerEyes = getEyeCoordinatesForPlayer(
                                playerPosition, {playerAtTick.viewX[tickIndex], playerAtTick.viewY[tickIndex]});
                        double hit0, hit1;
                        if (intersectP(walls[wallIndex], playerEyes, hit0, hit1)) {
                            tmpWallId[threadNum].push_back(wallIndex);
                            tmpWallX[threadNum].push_back(playerEyes.orig.x + hit0 * playerEyes.dir.x);
                            tmpWallY[threadNum].push_back(playerEyes.orig.y + hit0 * playerEyes.dir.y);
                            tmpWallZ[threadNum].push_back(playerEyes.orig.z + hit0 * playerEyes.dir.z);
                            hitAWall = true;
                            break;
                        }
                    }
                    if (!hitAWall) {
                        tmpWallId[threadNum].push_back(-1);
                        tmpWallX[threadNum].push_back(std::numeric_limits<double>::min());
                        tmpWallY[threadNum].push_back(std::numeric_limits<double>::min());
                        tmpWallZ[threadNum].push_back(std::numeric_limits<double>::min());
                    }
                }
            }
        }
    }

    for (int i = 0; i < numThreads; i++) {
        for (int j = 0; j < tmpPlayerAtTickId[i].size(); j++) {
            result.roundId.push_back(tmpRoundId[i][j]);
            result.playerAtTickId.push_back(tmpPlayerAtTickId[i][j]);
            result.playerId.push_back(tmpPlayerId[i][j]);
            result.posX.push_back(tmpPosX[i][j]);
            result.posY.push_back(tmpPosY[i][j]);
            result.posZ.push_back(tmpPosZ[i][j]);
            result.viewX.push_back(tmpViewX[i][j]);
            result.viewY.push_back(tmpViewY[i][j]);
            result.wallId.push_back(tmpWallId[i][j]);
            result.wallX.push_back(tmpWallX[i][j]);
            result.wallY.push_back(tmpWallY[i][j]);
            result.wallZ.push_back(tmpWallZ[i][j]);
            result.clusterId.push_back(-1);
        }
    }
    result.size = result.playerAtTickId.size();
    return result;
}

struct ACatPeekersSortableElement {
    int64_t roundId;
    int64_t playerAtTickId;
    int64_t playerId;
    int64_t aCatPeekersId;
    bool operator<(const ACatPeekersSortableElement & other) const {
        return (roundId < other.roundId) || (roundId == other.roundId && playerId < other.playerId) ||
               (roundId == other.roundId && playerId == other.playerId && playerAtTickId < other.playerAtTickId);
    }
};

ACatClusterSequence analyzeACatPeekersClusters(const PlayerAtTick & pat, ACatPeekers & aCatPeekers, const Cluster & clusters) {
    vector<ACatPeekersSortableElement> sortable;
    for (int64_t aCatPeekerIndex = 0; aCatPeekerIndex < aCatPeekers.size; aCatPeekerIndex++) {
        // create sortable array
        sortable.push_back({aCatPeekers.roundId[aCatPeekerIndex], aCatPeekers.playerAtTickId[aCatPeekerIndex],
                            aCatPeekers.playerId[aCatPeekerIndex], aCatPeekerIndex});
        // assuming first position is less than first kills
        vector<int64_t> candidateClusters;
        for (int i = 0; i < clusters.wallId.size(); i++) {
            if (clusters.wallId[i] == aCatPeekers.wallId[aCatPeekerIndex]) {
                candidateClusters.push_back(i);
            }
        }

        double minDistance = std::numeric_limits<double>::max();
        int minId = -1;
        for (int i = 0; i < candidateClusters.size(); i++) {
            double newDistance = computeDistance(
                    {clusters.x[candidateClusters[i]], clusters.y[candidateClusters[i]], clusters.z[candidateClusters[i]]},
                    {aCatPeekers.wallX[aCatPeekerIndex], aCatPeekers.wallY[aCatPeekerIndex], aCatPeekers.wallZ[aCatPeekerIndex]}
                    );
            if (newDistance < minDistance) {
                minDistance = newDistance;
                minId = candidateClusters[i];
            }
        }

        aCatPeekers.clusterId[aCatPeekerIndex] = clusters.id[minId];
    }
    std::sort(sortable.begin(), sortable.end());

    ACatClusterSequence result;
    vector<ClusterSequence> & clusterSequences = result.clusterSequences;
    for (const auto & sortableElement : sortable) {
        if (clusterSequences.empty()
            || clusterSequences[clusterSequences.size() - 1].roundId != sortableElement.roundId
            || clusterSequences[clusterSequences.size() - 1].playerId != sortableElement.playerId) {
            clusterSequences.push_back(ClusterSequence {});
        }
        ClusterSequence & curSequence = clusterSequences[clusterSequences.size() - 1];
        curSequence.roundId = sortableElement.roundId;
        curSequence.playerId = sortableElement.playerId;
        curSequence.playerAtTickIds.push_back(sortableElement.playerAtTickId);
        // if empty, start a new cluster
        // if not empty and old cluster doesn't match new one, add new cluster (and trust old timeInCluster.max was set before)
        // if not empty and old clsuter matches new one, update timeInCluster.max
        int curClusterId = aCatPeekers.clusterId[sortableElement.aCatPeekersId];
        int curTickId = pat.tickId[aCatPeekers.playerAtTickId[sortableElement.aCatPeekersId]];
        if (curSequence.clusterIds.empty() || curSequence.clusterIds[curSequence.clusterIds.size() - 1] != curClusterId) {
            curSequence.clusterIds.push_back(curClusterId);
            curSequence.tickIdsInCluster.push_back({curTickId, curTickId});
        }
        else {
            curSequence.tickIdsInCluster[curSequence.tickIdsInCluster.size() - 1].maxId = curTickId;
        }
    }
    return result;
}
