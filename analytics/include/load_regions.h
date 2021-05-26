#ifndef CSKNOW_LOAD_REGIONS_H
#define CSKNOW_LOAD_REGIONS_H
#include "file_helpers.h"
#include "geometry.h"
#include <vector>
#include <string>
using std::vector;
using std::string;

struct Regions {
    vector<int64_t> id;
    vector<string> name;
    vector<AABB> aabb;
};

Regions loadRegions(string filePath);

#endif //CSKNOW_LOAD_REGIONS_H
