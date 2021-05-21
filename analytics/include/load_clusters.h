#ifndef CSKNOW_LOAD_CLUSTERS_H
#define CSKNOW_LOAD_CLUSTERS_H
#include <string>
#include <set>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <list>
#include "load_data.h"
using std::string;
using std::vector;
using std::set;
using std::unordered_map;
using std::vector;

class Cluster {
public:
    vector<int64_t> id;
    vector<int64_t> wallId;
    vector<double> x;
    vector<double> y;
    vector<double> z;

    Cluster() {};
    Cluster(string filePath);
};

struct ClusterSequence {
    int64_t roundId;
    int64_t playerId;
    vector<int64_t> playerAtTickIds;
    vector<int> clusterIds;
    vector<RangeIndexEntry> tickIdsInCluster;
};


#endif //CSKNOW_LOAD_CLUSTERS_H
