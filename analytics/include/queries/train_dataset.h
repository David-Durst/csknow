#ifndef CSKNOW_DATAST_GENERATION_H
#define CSKNOW_DATAST_GENERATION_H
#include "load_data.h"
#include "query.h"
#include "indices/spotted.h"
#include "navmesh/nav_file.h"
#include "geometry.h"
#include <array>
#include <string>
#include <map>
#include <sstream>
using std::string;
using std::map;
#define DECISION_SECONDS 0.25
#define COSINE_SIMILARITY_THRESHOLD 0.9

class TrainDatasetResult : public QueryResult {
public:
    struct NavmeshState {
        uint32_t numFriends;
        uint32_t numEnemies;
    };

    struct TimeStepState {
        size_t curArea;
        int32_t team;
        Vec3 pos;
        vector<NavmeshState> navStates;
        // these aren't printed, just used for bookkeeping during query
        int64_t gameId;
        int64_t tickId;
        int64_t patId;
        TimeStepState(int64_t numNavmeshAreas) : navStates(numNavmeshAreas, {0, 0}) { };
    };

    string timeStepStateToString(TimeStepState step) {
        std::stringstream result;
        result << step.curArea;
        //result << "," << step.team;
        for (const auto & navState : step.navStates) {
            result << "," << navState.numFriends << "," << navState.numEnemies;
        }
        return result.str();
    }

    void timeStepStateColumns(vector<TimeStepState> steps, string prefix, vector<string> & result) {
        result.push_back(prefix + " nav area");
        //result.push_back(prefix + " team");
        if (steps.size() > 0) {
            for (size_t i = 0; i < steps.front().navStates.size(); i++) {
                result.push_back(prefix + " nav " + std::to_string(i) + " friends");
                result.push_back(prefix + " nav " + std::to_string(i) + " enemies");
            }
        }
    }

    struct TimeStepPlan {
        // result data
        double deltaX, deltaY;
        bool shootDuringNextThink;
        bool crouchDuringNextThink;
        int32_t navTargetArea;
    };

    string timeStepPlanToString(TimeStepPlan plan) {
        std::stringstream result;
        result << plan.deltaX << "," << plan.deltaY
             << "," << (plan.shootDuringNextThink ? "1" : "0")
             << "," << (plan.crouchDuringNextThink ? "1" : "0")
             << "," << plan.navTargetArea;
        return result.str();
    }

    void timeStepPlanColumns(vector<TimeStepPlan> plans, vector<string> & result) {
        result.push_back("delta x");
        result.push_back("delta y");
        result.push_back("shoot next");
        result.push_back("crouch next");
        result.push_back("nav target");
    }

    vector<int64_t> tickId;
    vector<int64_t> sourcePlayerId;
    vector<string> sourcePlayerName;
    vector<string> demoName;
    vector<TimeStepState> curState;
    vector<TimeStepState> lastState;
    vector<TimeStepState> oldState;
    vector<TimeStepPlan> plan;

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
            << "," << demoName[index] << "," << timeStepStateToString(curState[index])
            << "," << timeStepStateToString(lastState[index])
            << "," << timeStepStateToString(lastState[index]) << "," << timeStepPlanToString(plan[index])
            << std::endl;
    }

    vector<string> getForeignKeyNames() {
        return {"tick id", "source player id"};
    }

    vector<string> getOtherColumnNames() {
        vector<string> result{"source player name", "demo name"};
        timeStepStateColumns(curState, "cur", result);
        timeStepStateColumns(lastState, "last", result);
        timeStepStateColumns(oldState, "old", result);
        timeStepPlanColumns(plan, result);
        return result;
    }

    string getDataLabelRanges() {
        std::stringstream result;
        vector<string> inputCols, outputCols;
        timeStepStateColumns(curState, "cur", inputCols);
        timeStepStateColumns(lastState, "last", inputCols);
        timeStepStateColumns(oldState, "old", inputCols);
        timeStepPlanColumns(plan, outputCols);
        result << "source player id\n";
        for (size_t i = 0; i < inputCols.size(); i++) {
            if (i != 0) {
                result << ",";
            }
            result << inputCols[i];
        }
        result << "\n";
        for (size_t i = 0; i < outputCols.size(); i++) {
            if (i != 0) {
                result << ",";
            }
            result << outputCols[i];
        }
        return result.str();
    }
};

TrainDatasetResult queryTrainDataset(const Games & games, const Rounds & rounds, const Ticks & ticks,
                                     const Players & players, const PlayerAtTick & playerAtTick,
                                     const std::map<std::string, const nav_mesh::nav_file> & mapNavs);

#endif //CSKNOW_DATAST_GENERATION_H
