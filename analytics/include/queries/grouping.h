#ifndef CSKNOW_GROUPING_H
#define CSKNOW_GROUPING_H
#include "load_data.h"
#include "query.h"
#include <string>
#define GROUPING_WINDOW_SIZE 64
#define GROUPING_DISTANCE 700
#define MAX_GROUP_SIZE 5
using std::string;

class GroupingResult : public JustTargetResult {
public:
    vector<vector<int>> & teammates = targets;
    vector<double> maxX;
    vector<double> minX;
    vector<double> maxY;
    vector<double> minY;
    vector<double> maxZ;
    vector<double> minZ;

    GroupingResult() {
        targetNames = {"member 1", "member 2", "member 3", "member 4", "member 5"};
        ticksPerEvent = GROUPING_WINDOW_SIZE;
        keysForDiff = {0, 1, 2, 3, 4, 5, 6};
    }

    vector<string> getExtraColumnNames() {
        return {"max x", "min x", "max y", "min y"};
    }

    vector<string> getExtraRow(const Position & position, int64_t queryIndex, int64_t posIndex) {
        return {doubleToString(maxX[queryIndex]), doubleToString(minX[queryIndex]),
                doubleToString(maxY[queryIndex]), doubleToString(minY[queryIndex])};
    }
};

GroupingResult queryGrouping(const Position & position);


#endif //CSKNOW_GROUPING_H
