#ifndef CSKNOW_VELOCITY_H
#define CSKNOW_VELOCITY_H
#include "load_data.h"
#include "query.h"
#include <string>
#include <map>
using std::string;
using std::map;

class VelocityResult : public AllPlayersQuery<double> {
public:
    vector<int64_t> gameStarts;
    vector<string> fileNames;

    VelocityResult() {
        ticksPerEvent = 1;
        keysForDiff = {0, 1};
        valueName = "velocity";
    }

    string perPlayerValueToString(double value) {
        return std::to_string(value);
    }

    vector<string> getExtraColumnNames() {
        return {};
    }

    vector<string> getExtraRow(const Position & position, int64_t index) {
        return {};
    }
};

VelocityResult queryVelocity(const Position & position);

#endif //CSKNOW_VELOCITY_H
