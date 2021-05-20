#ifndef CSKNOW_LOAD_CLUSTERS_H
#define CSKNOW_LOAD_CLUSTERS_H
#include <string>
#include <set>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <list>
using std::string;
using std::vector;
using std::set;
using std::unordered_map;
using std::vector;

class Cluster {
public:
    vector<int64_t> index;
    vector<double> x;
    vector<double> y;
    vector<double> z;

    Cluster(string filePath);
};



#endif //CSKNOW_LOAD_CLUSTERS_H
