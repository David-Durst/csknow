#ifndef CSKNOW_NETCODE_H
#define CSKNOW_NETCODE_H
#include "load_data.h"
#include "query.h"
#include <string>
using std::string;

class NetcodeResult : public SourceAndTargetResult {
public:
    vector<int> & shooters = sources;
    vector<vector<int>> & luckys = targets;

    NetcodeResult() {
        sourceName = "shooter";
        targetNames = {"lucky"};
    }

    vector<string> getExtraColumnNames() {
        return {};
    }

    vector<string> getExtraRow(const Position & position, int64_t index) {
        return {};
    }
};

NetcodeResult queryNetcode(const Position & position, const WeaponFire & weaponFire,
                           const PlayerHurt & playerHurt);

#endif //CSKNOW_NETCODE_H
