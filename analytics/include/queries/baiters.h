#ifndef CSKNOW_BAITERS_H
#define CSKNOW_BAITERS_H
#include "load_data.h"
#include "query.h"
#include "indices.h"
#include <string>
using std::string;

class BaitersResult : public SingleSourceSingleTargetResult {
public:
    vector<int> & baiters = sources;
    vector<int> & victims = targets;
    vector<int64_t> allyDeathTicks;

    BaitersResult() {
        sourceName = "baiter";
        targetName = "victim";
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
