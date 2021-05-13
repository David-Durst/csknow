#ifndef CSKNOW_NONCONSECUTIVE_H
#define CSKNOW_NONCONSECUTIVE_H
#include "load_data.h"
#include "query.h"
#include <string>
#include <map>
#include <iomanip>
using std::string;
using std::map;
/*
class NonConsecutiveResult : public NoSourceTargetQuery {
public:
    vector<int64_t> gameStarts;
    vector<string> fileNames;
    vector<int64_t> nextTicks;

    NonConsecutiveResult() {
        ticksPerEvent = 1;
        keysForDiff = {0, 1};
    }

    vector<string> getExtraColumnNames() {
        return {"next tick"};
    }

    vector<string> getExtraRow(const Position & position, int64_t queryIndex, int64_t posIndex) {
        return {std::to_string(nextTicks[queryIndex])};
    }
};

NonConsecutiveResult queryNonConsecutive(const Position & position);
*/
#endif //CSKNOW_NONCONSECUTIVE_H
