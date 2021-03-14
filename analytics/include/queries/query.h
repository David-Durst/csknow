#ifndef CSKNOW_QUERY_H
#define CSKNOW_QUERY_H
#include <vector>
#include <sstream>
using std::vector;
using std::stringstream;

struct Vec3 {
    double x;
    double y;
    double z;
};

struct Vec2 {
    double x;
    double y;
};

struct Vec5 {
    Vec3 pos;
    Vec2 view;
};

struct AABB {
    Vec3 min;
    Vec3 max;
};

class QueryResult {
public:
    virtual string toCSV() = 0;
};

class PredicateResult : public QueryResult {
public:
    vector<int64_t> positionIndex;

    void collectResults(vector<int64_t> * tmpIndices, int numThreads) {
        for (int i = 0; i < numThreads; i++) {
            for (const auto & elem : tmpIndices[i]) {
                positionIndex.push_back(elem);
            }
        }
    }

    virtual string toCSV() {
        stringstream ss;
        ss << "position index" << std::endl;
        for (const auto & index : positionIndex) {
            ss << index << std::endl;
        }
        return ss.str();
    };
};

#endif //CSKNOW_QUERY_H
