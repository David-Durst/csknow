#ifndef CSKNOW_LOAD_WALLS_H
#define CSKNOW_LOAD_WALLS_H
#include "file_helpers.h"
#include "geometry.h"
#include <vector>
#include <string>
using std::vector;
using std::string;

struct Walls {
    vector<int64_t> id;
    vector<string> name;
    vector<AABB> aabb;
};

Walls loadWalls(string filePath);

#endif //CSKNOW_LOAD_WALLS_H
