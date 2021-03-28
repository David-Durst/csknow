#ifndef CSKNOW_LOOKING_H
#define CSKNOW_LOOKING_H
#include "load_data.h"
#include "query.h"
#include "indices/spotted.h"
#include <string>
#include <map>
using std::string;
using std::map;

class LookingResult : public SourceAndTargetResult {
public:
    vector<int> & lookers = sources;
    vector<vector<int>> & lookedAt = targets;
    vector<int64_t> gameStarts;
    vector<string> fileNames;

    LookingResult() {
        sourceName = "lookers";
        targetNames = {"lookedAt"};
        ticksPerEvent = 1;
        keysForDiff = {0, 1, 2, 3};
    }

    vector<string> getExtraColumnNames() {
        return {};
    }

    vector<string> getExtraRow(const Position & position, int64_t index) {
        return {};
    }
};

LookingResult queryLookers(const Position & position);

#endif //CSKNOW_LOOKING_H
