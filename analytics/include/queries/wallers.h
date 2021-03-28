#ifndef CSKNOW_WALLERS_H
#define CSKNOW_WALLERS_H
#include "load_data.h"
#include "query.h"
#include <string>
#define WALL_WINDOW_SIZE 64
using std::string;

class WallersResult : public SourceAndTargetResult {
public:
    vector<int> & cheaters = sources;
    vector<vector<int>> & victims = targets;

    WallersResult() {
        sourceName = "waller";
        targetNames = {"victim"};
        ticksPerEvent = WALL_WINDOW_SIZE;
        keysForDiff = {0, 1, 2, 3};
    }

    vector<string> getExtraColumnNames() {
        return {};
    }

    vector<string> getExtraRow(const Position & position, int64_t index) {
        return {};
    }
};

WallersResult queryWallers(const Position & position, const Spotted & spotted);

#endif //CSKNOW_WALLERS_H
