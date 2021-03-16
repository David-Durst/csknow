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

    double operator[](size_t index) const {
        if (index == 0) {
            return x;
        } else if (index == 1) {
            return y;
        } else {
            return z;
        }
    }
};

struct Vec2 {
    double x;
    double y;
};

class Ray {
public:
    Ray() { }
    Ray(const Vec3 &orig, const Vec3 &dir) : orig(orig), dir(dir) {
        invdir.x = 1 / dir.x;
        invdir.y = 1 / dir.y;
        invdir.z = 1 / dir.y;
        sign[0] = invdir.x < 0;
        sign[1] = invdir.y < 0;
        sign[2] = invdir.z < 0;
    }

    Vec3 orig, dir, invdir;
    int sign[3];
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
    vector<string> demoFile;

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
