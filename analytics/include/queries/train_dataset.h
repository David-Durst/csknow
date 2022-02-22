#ifndef CSKNOW_DATAST_GENERATION_H
#define CSKNOW_DATAST_GENERATION_H
#include "load_data.h"
#include "query.h"
#include "indices/spotted.h"
#include "navmesh/nav_file.h"
#include <string>
#include <map>
#include <sstream>
using std::string;
using std::map;
#define DECISION_SECONDS 0.25

class TrainDatasetResult : public QueryResult {
public:
    struct NavmeshState {
        uint32_t numFriends;
        uint32_t numEnemies;
    };

    struct TimeStepState {
        uint32_t curAABB;
        vector<NavmeshState> navStates;
        // these aren't printed, just used for bookkeeping during query
        int64_t gameId;
        int64_t tickId;
        int64_t patId;
        TimeStepState(int64_t numNavmeshAABBs) : navStates(numNavmeshAABBs, {0, 0}) { };
    };

    string timeStepToString(TimeStepState step) {
        std::stringstream result;
        result << step.curAABB;
        for (const auto & navState : step.navStates) {
            result << "," << navState.numFriends << "," << navState.numEnemies;
        }
        return result.str();
    }

    vector<string> timeStepColumns(vector<TimeStepState> steps, string prefix = "") {
        vector<string> result;
        result.push_back(prefix + " nav aabb");
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
        vector<string> curColumns = timeStepColumns(curState, "cur");
        vector<string> lastColumns = timeStepColumns(lastState, "last");
        vector<string> oldColumns = timeStepColumns(oldState, "old");
        result.insert(result.end(), curColumns.begin(), curColumns.end());
        result.insert(result.end(), lastColumns.begin(), lastColumns.end());
        result.insert(result.end(), oldColumns.begin(), oldColumns.end());
        return result;
    }
};

TrainDatasetResult queryTrainDataset(const Games & games, const Rounds & rounds, const Ticks & ticks,
                                     const Players & players, const PlayerAtTick & playerAtTick,
                                     const std::map<std::string, const nav_mesh::nav_file> & mapNavs);

#endif //CSKNOW_DATAST_GENERATION_H
