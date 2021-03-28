#ifndef CSKNOW_BAITERS_H
#define CSKNOW_BAITERS_H
#include "load_data.h"
#include "query.h"
#include "indices/spotted.h"
#include <string>
#define BAIT_WINDOW_SIZE 64
using std::string;

class BaitersResult : public SourceAndTargetResult {
public:
    vector<int> & baiters = sources;
    vector<vector<int>> & victimsAndKillers = targets;
    vector<int64_t> allyDeathTicks;

    BaitersResult() {
        sourceName = "baiter";
        targetNames = {"victim", "killer"};
        ticksPerEvent = BAIT_WINDOW_SIZE;
    }

    vector<string> getExtraColumnNames() {
        return {"ally death tick"};
    }

    vector<string> getExtraRow(const Position & position, int64_t index) {
        return {std::to_string(allyDeathTicks[index])};
    }
};

BaitersResult queryBaiters(const Position & position, const Kills & kills, const SpottedIndex & spottedIndex);

#endif //CSKNOW_BAITERS_H
