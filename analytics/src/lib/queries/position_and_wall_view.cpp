//
// Created by durst on 5/18/21.
//
#include "queries/position_and_wall_view.h"
#include "load_regions.h"
#include "geometry.h"
#include "indices/spotted.h"
#include <omp.h>
#include <set>
#include <map>
#include <limits>
#include <algorithm>
using std::string;
using std::map;

PositionsAndWallViews queryViewsFromRegion(const Rounds & rounds, const Ticks & ticks, const PlayerAtTick & playerAtTick,
                                           string standingFilePath, string wallsFilePath) {
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

    Regions standingPositionRegion = loadRegions(standingFilePath);
    AABB standingPosition = standingPositionRegion.aabb[0];
    Regions walls = loadRegions(wallsFilePath);
    /*
    AABB standingPosition{{190.0, 1327.34, -5.0}, {513.0, 2260.0, std::numeric_limits<double>::max()}};
    AABB frontCatWall{{507.0, 1353.0, std::numeric_limits<double>::min()}, {507.1, 2003.0, std::numeric_limits<double>::max()}};
    AABB oppositeElevatorWall{{507.0, 2003.0, std::numeric_limits<double>::min()}, {1215.0, 2003.1, std::numeric_limits<double>::max()}};
    AABB backCatWall{{224.0, 1301, std::numeric_limits<double>::min()}, {224.1, 2733.0, std::numeric_limits<double>::max()}};
    AABB ninjaWall{{224.0, 2733.0, std::numeric_limits<double>::min()}, {1010.0, 2733.1, std::numeric_limits<double>::max()}};
    AABB gooseWall{{1010.0, 2733.0, std::numeric_limits<double>::min()}, {1010.1, 3090.0, std::numeric_limits<double>::max()}};
    AABB topRampWall{{1010.0, 3090.0, std::numeric_limits<double>::min()}, {1588.0, 3090.1, std::numeric_limits<double>::max()}};
    AABB longWall{{1588.0, -1247.0, std::numeric_limits<double>::min()}, {1588.0, 3090.1, std::numeric_limits<double>::max()}};

    // these are wrapping containers to catch stray rays
    AABB leftContainer{{-2253.0, -900.0, std::numeric_limits<double>::min()}, {-2253.1, 3168.0, std::numeric_limits<double>::max()}};
    AABB topContainer{{-2253.0, 3168.0, std::numeric_limits<double>::min()}, {1897.0, 3168.1, std::numeric_limits<double>::max()}};
    AABB bottomContainer{{-2253.0, -900.0, std::numeric_limits<double>::min()}, {1897.0, -900.1, std::numeric_limits<double>::max()}};
    AABB rightContainer{{1897.0, -900.0, std::numeric_limits<double>::min()}, {1897.0, 3168.1, std::numeric_limits<double>::max()}};
    vector<AABB> walls{frontCatWall, oppositeElevatorWall, backCatWall, ninjaWall, gooseWall, topRampWall, longWall,
                       leftContainer, topContainer, bottomContainer, rightContainer};
                       */
    PositionsAndWallViews result(walls.aabb);

#pragma omp parallel for
    for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
        int threadNum = omp_get_thread_num();
        // assuming first position is less than first kills
        for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
             tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {
            for (int64_t patIndex = ticks.patPerTick[tickIndex].minId; patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
                Vec3 playerPosition = {playerAtTick.posX[patIndex], playerAtTick.posY[patIndex], playerAtTick.posZ[patIndex]};
                if (pointInRegion(standingPosition, playerPosition)) {
                    tmpRoundId[threadNum].push_back(roundIndex);
                    tmpPlayerAtTickId[threadNum].push_back(playerAtTick.id[patIndex]);
                    tmpPlayerId[threadNum].push_back(playerAtTick.playerId[patIndex]);
                    tmpPosX[threadNum].push_back(playerAtTick.posX[patIndex]);
                    tmpPosY[threadNum].push_back(playerAtTick.posY[patIndex]);
                    tmpPosZ[threadNum].push_back(playerAtTick.posZ[patIndex]);
                    tmpViewX[threadNum].push_back(playerAtTick.viewX[patIndex]);
                    tmpViewY[threadNum].push_back(playerAtTick.viewY[patIndex]);
                    bool hitAWall = false;
                    for (int wallIndex = 0; wallIndex < walls.aabb.size(); wallIndex++) {
                        Ray playerEyes = getEyeCoordinatesForPlayer(
                                playerPosition, {playerAtTick.viewX[patIndex], playerAtTick.viewY[patIndex]});
                        double hit0, hit1;
                        if (intersectP(walls.aabb[wallIndex], playerEyes, hit0, hit1)) {
                            tmpWallId[threadNum].push_back(walls.id[wallIndex]);
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

struct PosAndWallSortableElement {
    int64_t roundId;
    int64_t playerAtTickId;
    int64_t playerId;
    int64_t posAndWallId;
    bool operator<(const PosAndWallSortableElement & other) const {
        return (roundId < other.roundId) || (roundId == other.roundId && playerId < other.playerId) ||
               (roundId == other.roundId && playerId == other.playerId && playerAtTickId < other.playerAtTickId);
    }
};

ClusterSequencesByRound analyzeViewClusters(const Rounds & rounds, const Players & players,
                                            const PlayerAtTick & pat, PositionsAndWallViews & posAndWall, const Cluster & clusters) {
    vector<PosAndWallSortableElement> sortable;
    for (int64_t aCatPeekerIndex = 0; aCatPeekerIndex < posAndWall.size; aCatPeekerIndex++) {
        // create sortable array
        sortable.push_back({posAndWall.roundId[aCatPeekerIndex], posAndWall.playerAtTickId[aCatPeekerIndex],
                            posAndWall.playerId[aCatPeekerIndex], aCatPeekerIndex});

        double minDistance = std::numeric_limits<double>::max();
        int minId = -1;
        for (int i = 0; i < clusters.id.size(); i++) {
            double newDistance = computeDistance(
                    {clusters.x[i], clusters.y[i], clusters.z[i]},
                    {posAndWall.wallX[aCatPeekerIndex], posAndWall.wallY[aCatPeekerIndex], posAndWall.wallZ[aCatPeekerIndex]}
                    );
            if (newDistance < minDistance) {
                minDistance = newDistance;
                minId = i;
            }
        }

        if (minId == -1) {
            posAndWall.clusterId[aCatPeekerIndex] = -1;
        }
        else {
            posAndWall.clusterId[aCatPeekerIndex] = clusters.id[minId];
        }
    }
    std::sort(sortable.begin(), sortable.end());

    ClusterSequencesByRound result;
    for (int i = 0; i < rounds.id.size(); i++) {
        result.clusterSequencesPerRound.push_back({-1, -1});
    }

    vector<ClusterSequence> & clusterSequences = result.clusterSequences;
    int64_t idPerClusterInSequence = 0;
    for (const auto & sortableElement : sortable) {
        if (clusterSequences.empty()
            || clusterSequences[clusterSequences.size() - 1].roundId != sortableElement.roundId
            || clusterSequences[clusterSequences.size() - 1].playerId != sortableElement.playerId) {
            clusterSequences.push_back(ClusterSequence {});
            result.clusterSequencesPerRound[sortableElement.roundId].maxId = clusterSequences.size() - 1;
            if (result.clusterSequencesPerRound[sortableElement.roundId].minId == -1) {
                result.clusterSequencesPerRound[sortableElement.roundId].minId = clusterSequences.size() - 1;
            }
        }
        ClusterSequence & curSequence = clusterSequences[clusterSequences.size() - 1];
        curSequence.roundId = sortableElement.roundId;
        curSequence.playerId = sortableElement.playerId;
        curSequence.name = players.name[sortableElement.playerId + players.idOffset];
        // if empty, start a new cluster
        // if not empty and old cluster doesn't match new one, add new cluster (and trust old timeInCluster.max was set before)
        // if not empty and old clsuter matches new one, update timeInCluster.max
        int curClusterId = posAndWall.clusterId[sortableElement.posAndWallId];
        int curTickId = pat.tickId[posAndWall.playerAtTickId[sortableElement.posAndWallId]];
        if (curSequence.clusterIds.empty() || curSequence.clusterIds[curSequence.clusterIds.size() - 1] != curClusterId) {
            curSequence.ids.push_back(idPerClusterInSequence++);
            curSequence.clusterIds.push_back(curClusterId);
            curSequence.tickIdsInCluster.push_back({curTickId, curTickId});
            curSequence.playerAtTickIds.push_back({sortableElement.playerAtTickId});
        }
        else {
            curSequence.tickIdsInCluster[curSequence.tickIdsInCluster.size() - 1].maxId = curTickId;
            curSequence.playerAtTickIds[curSequence.playerAtTickIds.size() - 1].push_back(sortableElement.playerAtTickId);
        }
    }
    result.size = clusterSequences.size();

    return result;
}
