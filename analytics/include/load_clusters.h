#ifndef CSKNOW_LOAD_CLUSTERS_H
#define CSKNOW_LOAD_CLUSTERS_H
#include <string>
#include <set>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <list>
#include "load_data.h"
#include "queries/query.h"
using std::string;
using std::vector;
using std::set;
using std::unordered_map;
using std::vector;

class Cluster : public QueryResult {
public:
    vector<int64_t> id;
    vector<int64_t> wallId;
    vector<double> x;
    vector<double> y;
    vector<double> z;

    vector<int64_t> filterByForeignKey(int64_t otherTableIndex) {
        // no indexes on results
        return {};
    }

    Cluster() {
        this->variableLength = false;
    };
    Cluster(string filePath);

    void oneLineToCSV(int64_t index, stringstream & ss) {
        ss << id[index] << "," << wallId[index] << "," << x[index] << "," << y[index] << "," << z[index] << std::endl;
    }

    vector<string> getForeignKeyNames() {
        return {"wall id"};
    }

    vector<string> getOtherColumnNames() {
        return {"x", "y", "z"};
    }
};

struct ClusterSequence {
    vector<int64_t> ids;
    int64_t roundId;
    int64_t playerId;
    string name;
    // separate playerAtTickIds for each time period in a cluster
    vector<vector<int64_t>> playerAtTickIds;
    vector<int> clusterIds;
    vector<RangeIndexEntry> tickIdsInCluster;
};


#endif //CSKNOW_LOAD_CLUSTERS_H
