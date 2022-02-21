#ifndef CSKNOW_DATAST_GENERATION_H
#define CSKNOW_DATAST_GENERATION_H
#include "load_data.h"
#include "query.h"
#include "indices/spotted.h"
#include <string>
#include <map>
#include <sstream>
using std::string;
using std::map;
#define LOOKBACK_SECONDS 0.25

class TrainDatasetResult : public QueryResult {
public:
    struct NavmeshState {
        int32_t numFriends;
        int32_t numEnemies;
    };

    struct TimeStepState {
        vector<NavmeshState> navStates;
    };

    string timeStepToString(TimeStepState step) {
        std::stringstream result;
        bool firstStep = true;
        for (const auto & navState : step.navStates) {
            if (!firstStep) {
                result << ",";
            }
            result << navState.numFriends << "," << navState.numEnemies;
            firstStep = false;
        }
        return result.str();
    }

    vector<string> timeStepColumns(vector<TimeStepState> steps, string prefix = "") {
        vector<string> result;
        for (size_t i = 0; i < steps.front().navStates.size(); i++) {
            result.push_back(prefix + " nav aabb " + std::to_string(i) + " friends");
            result.push_back(prefix + " nav aabb " + std::to_string(i) + " enemies");
        }
        return result;
    }

    vector<int64_t> tickId;
    vector<int64_t> sourcePlayerId;
    vector<string> sourcePlayerName;
    vector<string> demoName;
    vector<TimeStepState> curState;
    vector<TimeStepState> lastState;
    vector<TimeStepState> oldState;

    TrainDatasetResult() {
        this->startTickColumn = -1;
        this->ticksPerEvent = 1;
    }

    vector<int64_t> filterByForeignKey(int64_t otherTableIndex) {
        // no filtering on dataset
        vector<int64_t> result;
        return result;
    }

    void oneLineToCSV(int64_t index, stringstream & ss) {
        ss << index << "," << tickId[index] << "," << sourcePlayerId[index] << "," << sourcePlayerName[index]
            << "," << demoName[index] << "," << timeStepToString(curState[index])
            << "," << timeStepToString(lastState[index]) << "," << timeStepToString(lastState[index]);
    }

    vector<string> getForeignKeyNames() {
        return {"tick id", "source player id"};
    }

    vector<string> getOtherColumnNames() {
        vector<string> result{"source player name", "demo name"};
        vector<string> curColumns = timeStepColumns(curState);
        vector<string> lastColumns = timeStepColumns(lastState);
        vector<string> oldColumns = timeStepColumns(oldState);
        result.insert(result.end(), curColumns.begin(), curColumns.end());
        result.insert(result.end(), lastColumns.begin(), lastColumns.end());
        result.insert(result.end(), oldColumns.begin(), oldColumns.end());
        return result;
    }
};

TrainDatasetResult queryTrainDataset(const Games & games, const Rounds & rounds, const Ticks & ticks,
                                     const Players & players, const PlayerAtTick & playerAtTick);

#endif //CSKNOW_DATAST_GENERATION_H
