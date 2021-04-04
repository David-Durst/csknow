#ifndef CSKNOW_GROUPINSEQUENCEOFREGIONS_H
#define CSKNOW_GROUPINSEQUENCEOFREGIONS_H
#include "load_data.h"
#include "query.h"
#include "geometry.h"
#include <string>
using std::string;

class GroupInSequenceOfRegionsResult : public JustTargetResult {
public:
    vector<CompoundAABB> sequenceOfRegions;
    vector<bool> wantToReachRegions;
    vector<int64_t> gameStarts;
    vector<vector<int>> & teammates = targets;
    vector<int64_t> endTick;
    vector<vector<string>> memberInRegion;
    vector<vector<int64_t>> tickInRegion;
    vector<vector<double>> xInRegion;
    vector<vector<double>> yInRegion;
    vector<vector<double>> zInRegion;
    vector<string> colNames;

    GroupingResult(vector<CompoundAABB> sequenceOfRegions, vector<bool> wantToReachRegions) {
        targetNames = {"member 1", "member 2", "member 3"};
        variableLength = true;
        ticksColumn = 5;
        keysForDiff = {0, 1, 2, 3, 4, 5};
        this->sequenceOfRegions = sequenceOfRegions;
        this->wantToReachRegions = wantToReachRegions;

        colNames = {"end tick"};
        for (int i = 0; i < wantToReachRegions.size(); i++) {
            if (wantToReachRegions[i]) {
                colNames.push_back("player reaching region " + std::to_string(i));
                colNames.push_back("tick " + std::to_string(i));
                colNames.push_back("x " + std::to_string(i));
                colNames.push_back("y " + std::to_string(i));
                colNames.push_back("z " + std::to_string(i));
            }
        }
    }

    vector<string> getExtraColumnNames() {
        return colNames
    }

    vector<string> getExtraRow(const Position & position, int64_t queryIndex, int64_t posIndex) {
        vector<string> results = {std::to_string(endTick[queryIndex])};
        int posInResults = 0;
        for (int i = 0; i < wantToReachRegions.size(); i++) {
            if (wantToReachRegions[i]) {
                colNames.push_back(memberInRegion[queryIndex][j]);
                colNames.push_back(std::to_string(tickInRegion[queryIndex][j]));
                colNames.push_back(doubleToString(xInRegion[queryIndex][j]));
                colNames.push_back(doubleToString(yInRegion[queryIndex][j]));
                colNames.push_back(doubleToString(zInRegion[queryIndex][j]));
                j++;
            }
        }
        return results;
    }
};

GroupingResult queryGrouping(const Position & position);

#endif //CSKNOW_GROUPINSEQUENCEOFREGIONS_H
