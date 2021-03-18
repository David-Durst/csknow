#ifndef CSKNOW_NETCODE_H
#define CSKNOW_NETCODE_H
#include "load_data.h"
#include "query.h"
#include <string>
using std::string;

class NetcodeResult : public SingleSourceSingleTargetResult {
public:
    vector<int> & shooters = sources;
    vector<int> & luckys = targets;

    NetcodeResult() {
        sourceName = "shooter";
        targetName = "lucky";
    }

    string getExtraColumns() {
        return "";
    }

    string getExtraRow(const Position & position, int64_t index) {
        return "";
    }
};

NetcodeResult queryNetcode(const Position & position, const WeaponFire & weaponFire,
                           const PlayerHurt & playerHurt);

#endif //CSKNOW_NETCODE_H
