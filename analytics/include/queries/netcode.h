#ifndef CSKNOW_NETCODE_H
#define CSKNOW_NETCODE_H
#include "load_data.h"
#include "query.h"
#include "indices/spotted.h"
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

    vector<string> getExtraRow(const Position & position, int64_t queryIndex, int64_t posIndex) {
        return {};
    }
};

NetcodeResult queryNetcode(const Position & position, const WeaponFire & weaponFire,
                           const PlayerHurt & playerHurt, const SpottedIndex & spottedIndex);

// https://developer.valvesoftware.com/wiki/Dimensions#Map_Grid_Units:_quick_reference
const double UNITS_PER_METER = 52.49;

static inline __attribute__((always_inline))
double metersToUnits(double meters) {
    return meters * UNITS_PER_METER;
}

// the below are in meters
const map<string, double> weaponAccurateRanges {
    {"AK-47", metersToUnits(21.74)},
    {"AUG", metersToUnits(28.22)},
    {"AWP", metersToUnits(69.27)},
    {"Desert Eagle", metersToUnits(24.58)},
    {"FAMAS", metersToUnits(14.58)},
    {"G3SG1", metersToUnits(66.26)},
    {"Galil AR", metersToUnits(16.26)},
    {"Glock-18", metersToUnits(20.05)},
    {"M4A4", metersToUnits(27.71)},
    {"MAC-10", metersToUnits(10.96)},
    {"MAG-7", metersToUnits(3.24)},
    {"MP7", metersToUnits(14.38)},
    {"MP9", metersToUnits(15.88)},
    {"Nova", metersToUnits(3.24)},
    {"P2000", metersToUnits(22.09)},
    {"p250", metersToUnits(13.73)},
    {"P90", metersToUnits(10.40)},
    {"PP-Bizon", metersToUnits(10.16)},
    {"SCAR-20", metersToUnits(66.26)},
    {"SG 553", metersToUnits(23.78)},
    {"SSG 08", metersToUnits(23.78)},
    {"UMP-45", metersToUnits(10.56)},
    {"XM1014", metersToUnits(3.39)}
};
*/
#endif //CSKNOW_NETCODE_H
