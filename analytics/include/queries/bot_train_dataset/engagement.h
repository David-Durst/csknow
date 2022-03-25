//
// Created by durst on 3/24/22.
//
#ifndef CSKNOW_ENGAGEMENT_H
#define CSKNOW_ENGAGEMENT_H
#include "load_data.h"
#include "queries/query.h"
#include "indices/spotted.h"
#include "navmesh/nav_file.h"
#include "geometry.h"
#include <array>
#include <string>
#include <map>
#include <sstream>
using std::string;
using std::map;
using std::array;
#define SPRAY_LOOKBACK 30
#define VELOCITY_LOOKBACK 3

class EngagementResult : public QueryResult {
public:
    struct BulletResult {
        bool valid;
        double secondsSinceFirst;
        bool hit;
        int64_t hitGroup;
    };

    struct ShooterTrajectory {
        array<BulletResult, SPRAY_LOOKBACK> bulletResults;
        Vec2 viewAngleWithActualRecoil;
        Vec2 viewAngleWithVisualRecoil;
    };

    struct WeaponState {
        int16_t currentWeapon;
        bool haveAwp;
        bool haveScout;
        bool haveAutomatic;
    };

    struct PosState {
        array<Vec3, VELOCITY_LOOKBACK> eyePosRelativeToShooter;
        Vec3 velocityRelativeToShooter;
        array<Vec2, VELOCITY_LOOKBACK> viewAngleRelativeToShooter;
        bool isCrouching;
        bool isWalking;
        bool isScoped;
        bool isAirborne;
        double remainingFlashTime;
    };

    struct FriendlyPlayerState {
        bool slotFilled;
        bool alive;
        PosState posState;
        int64_t money;
        WeaponState weaponState;
        int32_t health;
        int32_t armor;
        double secondsSinceLastFootstep;
        double secondsSinceLastFire;
    };

    struct EnemyPlayerState {
        bool slotFilled;
        bool alive;
        PosState posState;
        bool startRoundMoneyLessThan4k;
        bool beenHeadshotThisRound;
        WeaponState weaponState;
        int32_t priorTimesEngagedThisRound;
        double secondsSinceLastRadar;
        double secondsSinceLastFootstep;
        double secondsSinceLastFire;
        /*
        int64_t ticksSinceLastRadar;
        int64_t ticksSinceLastFootstep;
        int64_t ticksSinceLastFire;
         */
    };

    struct TimeStepState {
        size_t curArea;
        int32_t team;
        array<BulletResult, SPRAY_LOOKBACK> sprayState;
        array<Vec3, VELOCITY_LOOKBACK> globalShooterEyePos;
        array<Vec2, VELOCITY_LOOKBACK> globalShooterViewAngle;
        FriendlyPlayerState shooter;
        EnemyPlayerState target;
        array<FriendlyPlayerState, NUM_PLAYERS/2> friendlyPlayerStates;
        array<EnemyPlayerState, NUM_PLAYERS/2> enemyPlayerStates;
        // these aren't printed, just used for bookkeeping during query
        int64_t gameId;
        int64_t roundId;
        int64_t tickId;
        int64_t patId;
    };

    string timeStepStateToString(TimeStepState step) {
        std::stringstream result;
        result << step.curArea;
        //result << "," << step.team;
        /*
        result << "," << step.pos.toCSV();
        for (const auto & playerState : step.playerStates) {
            result << "," << boolToString(playerState.slotFilled)
                << "," << boolToString(playerState.alive)
                << "," << boolToString(playerState.friendly)
                << "," << playerState.posRelativeToShooterViewAngle.toCSV()
                << "," << playerState.shooterRelativeToPlayerViewAngle.toCSV();
        }
         */
        return result.str();
    }

    void timeStepStateColumns(vector<TimeStepState> steps, string prefix, vector<string> & result,
                              bool onlyOneHot = false, bool onlyMinMaxScale = false) {
        if (!onlyMinMaxScale) {
            result.push_back(prefix + " nav area");
        }
        //result.push_back(prefix + " team");
        if (!onlyOneHot) {
            result.push_back(prefix + " pos x");
            result.push_back(prefix + " pos y");
            result.push_back(prefix + " pos z");
            if (steps.size() > 0) {
                for (size_t i = 0; i < steps.front().playerStates.size(); i++) {
                    result.push_back(prefix + " i " + std::to_string(i) + " slot filled");
                    result.push_back(prefix + " i " + std::to_string(i) + " alive");
                    result.push_back(prefix + " i " + std::to_string(i) + " friendly");
                    result.push_back(prefix + " i " + std::to_string(i) + " relative pos x");
                    result.push_back(prefix + " i " + std::to_string(i) + " relative pos y");
                    result.push_back(prefix + " i " + std::to_string(i) + " relative pos z");
                    result.push_back(prefix + " i " + std::to_string(i) + " relative view x");
                    result.push_back(prefix + " i " + std::to_string(i) + " relative view y");
                    result.push_back(prefix + " player " + std::to_string(i) + " enemies");
                }
            }
        }
    }

    void timeStepStateOneHotNumCategories(vector<string> & result) {
        result.push_back(std::to_string(numNavAreas));
    }

    struct TimeStepPlan {
        // result data
        Vec3 deltaPos;
        Vec2 deltaView;
        bool fire;
        bool crouch;
        bool walk;
        bool scope;
        bool newlyAirborne;
    };

    string timeStepPlanToString(TimeStepPlan plan) {
        std::stringstream result;
        result << plan.deltaX << "," << plan.deltaY
               << "," << (plan.shootDuringNextThink ? "1" : "0")
               << "," << (plan.crouchDuringNextThink ? "1" : "0")
               << "," << plan.navTargetArea;
        return result.str();
    }

    void timeStepPlanColumns(vector<TimeStepPlan> plans, vector<string> & result,
                             bool onlyOneHot = false, bool onlyMinMaxScale = false) {
        if (!onlyOneHot) {
            result.push_back("delta x");
            result.push_back("delta y");
        }
        if (!onlyMinMaxScale) {
            result.push_back("shoot next");
            result.push_back("crouch next");
            result.push_back("nav target");
        }
    }

    void timeStepPlanOneHotNumCategories(vector<string> & result) {
        result.push_back("2");
        result.push_back("2");
        result.push_back(std::to_string(numNavAreas));
    }

    vector<int64_t> tickId;
    vector<int64_t> roundId;
    vector<int64_t> sourcePlayerId;
    vector<string> sourcePlayerName;
    vector<string> demoName;
    vector<TimeStepState> curState;
    vector<TimeStepState> lastState;
    vector<TimeStepState> oldState;
    vector<TimeStepPlan> plan;
    int64_t numNavAreas;

    NextNavmeshResult() {
        this->startTickColumn = -1;
        this->ticksPerEvent = 1;
    }

    vector<int64_t> filterByForeignKey(int64_t otherTableIndex) {
        // no filtering on dataset
        vector<int64_t> result;
        return result;
    }

    void oneLineToCSV(int64_t index, stringstream & ss) {
        ss << index << "," << tickId[index] << "," << roundId[index]
           << "," << sourcePlayerId[index] << "," << sourcePlayerName[index]
           << "," << demoName[index] << "," << curState[index].team
           << "," << timeStepStateToString(curState[index])
           << "," << timeStepStateToString(lastState[index])
           << "," << timeStepStateToString(lastState[index])
           << "," << timeStepPlanToString(plan[index]) << std::endl;
    }

    vector<string> getForeignKeyNames() {
        return {"tick id", "round id", "source player id"};
    }

    vector<string> getOtherColumnNames() {
        vector<string> result{"source player name", "demo name", "team"};
        timeStepStateColumns(curState, "cur", result);
        timeStepStateColumns(lastState, "last", result);
        timeStepStateColumns(oldState, "old", result);
        timeStepPlanColumns(plan, result);
        return result;
    }

    string getDataLabelRanges() {
        std::stringstream result;
        vector<string> inputCols, outputCols, inputOneHot, inputOneHotNumCategories, outputOneHot, outputOneHotNumCategories,
                inputMinMaxScale, outputMinMaxScale;

        timeStepStateColumns(curState, "cur", inputCols);
        timeStepStateColumns(lastState, "last", inputCols);
        timeStepStateColumns(oldState, "old", inputCols);
        timeStepPlanColumns(plan, outputCols);

        timeStepStateColumns(curState, "cur", inputOneHot, true);
        timeStepStateColumns(lastState, "last", inputOneHot, true);
        timeStepStateColumns(oldState, "old", inputOneHot, true);
        timeStepPlanColumns(plan, outputOneHot, true);

        // repeat the below once for each state
        timeStepStateOneHotNumCategories(inputOneHotNumCategories);
        timeStepStateOneHotNumCategories(inputOneHotNumCategories);
        timeStepStateOneHotNumCategories(inputOneHotNumCategories);
        timeStepPlanOneHotNumCategories(outputOneHotNumCategories);

        timeStepStateColumns(curState, "cur", inputMinMaxScale, false, true);
        timeStepStateColumns(lastState, "last", inputMinMaxScale, false, true);
        timeStepStateColumns(oldState, "old", inputMinMaxScale, false, true);
        timeStepPlanColumns(plan, outputMinMaxScale, false, true);

        result << "source player id\n";
        commaSeparateList(result, inputCols);
        result << "\n";
        commaSeparateList(result, inputOneHot);
        result << "\n";
        commaSeparateList(result, inputOneHotNumCategories);
        result << "\n";
        commaSeparateList(result, inputMinMaxScale);
        result << "\n";
        commaSeparateList(result, outputCols);
        result << "\n";
        commaSeparateList(result, outputOneHot);
        result << "\n";
        commaSeparateList(result, outputOneHotNumCategories);
        result << "\n";
        commaSeparateList(result, outputMinMaxScale);
        return result.str();
    }
};

NextNavmeshResult queryTrainDataset(const Games & games, const Rounds & rounds, const Ticks & ticks,
                                    const Players & players, const PlayerAtTick & playerAtTick,
                                    const std::map<std::string, const nav_mesh::nav_file> & mapNavs);

#endif //CSKNOW_ENGAGEMENT_H