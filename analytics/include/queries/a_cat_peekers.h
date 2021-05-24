//
// Created by durst on 5/18/21.
//

#ifndef CSKNOW_A_CAT_PEEKERS_H
#define CSKNOW_A_CAT_PEEKERS_H

#include "load_clusters.h"
#include "queries/query.h"
#include "load_data.h"
#include "geometry.h"
#include <string>
#include <vector>
using std::string;
using std::vector;

class ACatPeekers : public QueryResult {
public:
    vector<int64_t> roundId;
    vector<int64_t> playerAtTickId;
    vector<int64_t> playerId;
    vector<double> posX;
    vector<double> posY;
    vector<double> posZ;
    vector<double> viewX;
    vector<double> viewY;
    vector<int64_t> wallId;
    vector<double> wallX;
    vector<double> wallY;
    vector<double> wallZ;
    vector<AABB> walls;
    // cluster data per ACatPeekers tick
    vector<int> clusterId;


    ACatPeekers(vector<AABB> walls) {
        this->startTickColumn = -1;
        this->ticksPerEvent = 1;
        this->walls = walls;
    }

    vector<int64_t> filterByForeignKey(int64_t otherTableIndex) {
        // no indexes on results
        return {};
    }

    void oneLineToCSV(int64_t index, stringstream & ss) {
        ss << index << "," << playerAtTickId[index] << "," << posX[index] << "," << posY[index] << "," << posZ[index]
            << "," << viewX[index] << "," << viewY[index] << "," << wallId[index] << ","
            << wallX[index] << "," << wallY[index] << "," << wallZ[index] << std::endl;
    }

    vector<string> getForeignKeyNames() {
        return {"player at tick id"};
    }

    vector<string> getOtherColumnNames() {
        return {"pos x", "pos y", "pos z", "view x", "wall id", "wall x", "wall y", "wall z"};
    }
};

class ACatClusterSequence : public QueryResult {
public:
    vector<ClusterSequence> clusterSequences;
    vector<RangeIndexEntry> clusterSequencesPerRound;

    ACatClusterSequence() {
        this->variableLength = true;
        this->startTickColumn = 4;
        this->ticksColumn = 6;
    }

    vector<int64_t> filterByForeignKey(int64_t otherTableIndex) {
        // no indexes on results
        vector<int64_t> result;
        for (int i = clusterSequencesPerRound[otherTableIndex].minId; i <= clusterSequencesPerRound[otherTableIndex].maxId; i++) {
            result.push_back(i);
        }
        return result;
    }

    void oneLineToCSV(int64_t index, stringstream & ss) {
        for (int i = 0; i < clusterSequences[index].playerAtTickIds.size(); i++) {
            ss << index << ",";
            ss << clusterSequences[index].roundId << "," << clusterSequences[index].playerId << ","
                << clusterSequences[index].clusterIds[i] << ","
                << clusterSequences[index].tickIdsInCluster[i].minId << "," << clusterSequences[index].tickIdsInCluster[i].maxId << ","
                << clusterSequences[index].tickIdsInCluster[i].maxId - clusterSequences[index].tickIdsInCluster[i].minId + 1
                << std::endl;
        }
    }

    vector<string> getForeignKeyNames() {
        return {"round id", "player id", "player at tick id",
                "cluster id", "min tick id in cluster", "max tick id in cluster", "cluster length"};
    }

    vector<string> getOtherColumnNames() {
        return {};
    }
};


ACatPeekers queryACatPeekers(const Rounds & rounds, const Ticks & ticks, const PlayerAtTick & playerAtTick);
ACatClusterSequence analyzeACatPeekersClusters(const Rounds & rounds, const PlayerAtTick & pat, ACatPeekers & aCatPeekers, const Cluster & clusters);


#endif //CSKNOW_A_CAT_PEEKERS_H
