#ifndef CSKNOW_VELOCITY_H
#define CSKNOW_VELOCITY_H
#include "load_data.h"
#include "query.h"
#include <string>
#include <map>
using std::string;
using std::map;
/*
class NetcodeResult : public SourceAndTargetResult {
public:
    vector<int> & shooters = sources;
    vector<vector<int>> & luckys = targets;

    NetcodeResult() {
        sourceName = "shooter";
        targetNames = {"lucky"};
        ticksPerEvent = 32;
        keysForDiff = {0, 1, 2, 3};
    }

    vector<string> getExtraColumnNames() {
        return {};
    }

    vector<string> getExtraRow(const Position & position, int64_t index) {
        return {};
    }
};

NetcodeResult queryNetcode(const Position & position, const WeaponFire & weaponFire,
                           const PlayerHurt & playerHurt, const SpottedIndex & spottedIndex);

*/
#endif //CSKNOW_VELOCITY_H
