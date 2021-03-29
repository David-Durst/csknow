#ifndef CSKNOW_CONSECUTIVE_H
#define CSKNOW_CONSECUTIVE_H
#include "load_data.h"
#include "query.h"
#include <string>
#include <map>
#include <iomanip>
using std::string;
using std::map;

class VelocityResult : public NoSourceTargetQuery {
public:
    vector<int64_t> gameStarts;
    vector<string> fileNames;
    vector<double> resultsPerPlayer[NUM_PLAYERS];

    VelocityResult() {
        ticksPerEvent = 1;
        keysForDiff = {0, 1};
    }

    string perPlayerValueToString(double value) {
        std::stringstream ss;
        ss << std::setprecision(2) << std::fixed << value;
        return ss.str();
    }

    vector<string> getExtraColumnNames() {
        vector<string> result = {};
        for (int i = 0; i < NUM_PLAYERS; i++) {
            result.push_back(std::to_string(i) + " name");
            result.push_back(std::to_string(i) + " velocity");
        }
        return result;
    }

    vector<string> getExtraRow(const Position & position, int64_t index) {
        vector<string> result = {};
        for (int i = 0; i < NUM_PLAYERS; i++) {
            result.push_back(position.players[i].name[index]);
            result.push_back(perPlayerValueToString(resultsPerPlayer[i][index]));
        }
        return result;
    }
};

VelocityResult queryVelocity(const Position & position);

#endif //CSKNOW_CONSECUTIVE_H
