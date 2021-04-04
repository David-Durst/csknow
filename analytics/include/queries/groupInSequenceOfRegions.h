#ifndef CSKNOW_GROUPINSEQUENCEOFREGIONS_H
#define CSKNOW_GROUPINSEQUENCEOFREGIONS_H
#include "load_data.h"
#include "query.h"
#include "geometry.h"
#include "queries/grouping.h"
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
    // this is requirement for all results, so only need to store once
    vector<bool> stillGrouped;
    vector<string> colNames;

    GroupInSequenceOfRegionsResult(vector<CompoundAABB> sequenceOfRegions, vector<bool> wantToReachRegions,
                                   vector<bool> stillGrouped) {
        targetNames = {"member 1", "member 2", "member 3"};
        variableLength = true;
        ticksColumn = 5;
        keysForDiff = {0, 1, 2, 3, 4, 5};
        this->sequenceOfRegions = sequenceOfRegions;
        this->wantToReachRegions = wantToReachRegions;
        this->stillGrouped = stillGrouped;

        colNames = {"end tick"};
        for (int i = 0; i < wantToReachRegions.size(); i++) {
            if (wantToReachRegions[i]) {
                colNames.push_back("player reaching region " + std::to_string(i));
                colNames.push_back("tick " + std::to_string(i));
                colNames.push_back("x " + std::to_string(i));
                colNames.push_back("y " + std::to_string(i));
                colNames.push_back("z " + std::to_string(i));
                colNames.push_back("still grouped " + std::to_string(i));
            }
        }
    }

    vector<string> getExtraColumnNames() {
        return colNames;
    }

    vector<string> getExtraRow(const Position & position, int64_t queryIndex, int64_t posIndex) {
        vector<string> results = {std::to_string(endTick[queryIndex])};
        int posInResults = 0;
        for (int i = 0; i < wantToReachRegions.size(); i++) {
            if (wantToReachRegions[i]) {
                colNames.push_back(memberInRegion[queryIndex][posInResults]);
                colNames.push_back(std::to_string(tickInRegion[queryIndex][posInResults]));
                colNames.push_back(doubleToString(xInRegion[queryIndex][posInResults]));
                colNames.push_back(doubleToString(yInRegion[queryIndex][posInResults]));
                colNames.push_back(doubleToString(zInRegion[queryIndex][posInResults]));
                // using different index here as store stillGrouped once for all regions
                // while per column results only store entries for entries you actually want to be in
                colNames.push_back(boolToString(stillGrouped[i]));
                posInResults++;
            }
        }
        return results;
    }
};

GroupInSequenceOfRegionsResult queryGroupingInSequenceOfRegions(const Position & position,
                                                                const GroupingResult & groupingResult,
                                                                vector<CompoundAABB> sequenceOfRegions,
                                                                vector<bool> wantToReachRegions,
                                                                vector<bool> stillGrouped,
                                                                set<int> teams);

#endif //CSKNOW_GROUPINSEQUENCEOFREGIONS_H
