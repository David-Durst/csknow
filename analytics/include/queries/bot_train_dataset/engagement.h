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
    /*
    struct FireResult {
        bool valid;
        double secondsSinceFirst;
        bool hit;
        int64_t hitGroup;
    };
     */

    struct PosState {
        Vec3 eyePosRelativeToShooter;
        Vec3 velocityRelativeToShooter;
        Vec2 viewAngleRelativeToShooter;
        bool isCrouching;
        bool isWalking;
        bool isScoped;
        bool isAirborne;
        double remainingFlashTime;
    };

    string posStateToCSV(PosState posState) {
        std::stringstream result;
        result << posState.eyePosRelativeToShooter.toCSV();
        result << "," << posState.velocityRelativeToShooter.toCSV();
        result << "," << posState.viewAngleRelativeToShooter.toCSV();
        result << "," << boolToString(posState.isCrouching);
        result << "," << boolToString(posState.isWalking);
        result << "," << boolToString(posState.isScoped);
        result << "," << boolToString(posState.isAirborne);
        result << "," << posState.remainingFlashTime;
        return result.str();
    }

    void posStateColumns(string prefix, vector<string> & result, bool onlyOneHot = false, bool onlyMinMaxScale = false) {
        if (!onlyOneHot) {
            result.push_back(prefix + " eye pos rel x");
            result.push_back(prefix + " eye pos rel y");
            result.push_back(prefix + " eye pos rel z");
            result.push_back(prefix + " vel rel x");
            result.push_back(prefix + " vel rel y");
            result.push_back(prefix + " vel rel z");
            result.push_back(prefix + " view angle rel x");
            result.push_back(prefix + " view angle rel y");
            if (!onlyMinMaxScale) {
                result.push_back(prefix + " crouching");
                result.push_back(prefix + " walking");
                result.push_back(prefix + " scoped");
                result.push_back(prefix + " airborne");
            }
            result.push_back(prefix + " remaining flash time");
        }
    }

    struct FriendlyPlayerState {
        bool slotFilled;
        bool alive;
        PosState posState;
        int64_t money;
        int16_t currentWeapon;
        int16_t primaryWeapon;
        int16_t secondaryWeapon;
        int16_t currentClipBullets;
        int16_t primaryClipBullets;
        int16_t secondaryClipBullets;
        int32_t health;
        int32_t armor;
        double secondsSinceLastFootstep;
        double secondsSinceLastFire;
    };

    string friendlyPlayerStateToCSV(FriendlyPlayerState playerState) {
        std::stringstream result;
        result << boolToString(playerState.slotFilled);
        result << "," << boolToString(playerState.alive);
        result << "," << boolToString(playerState.alive);
        result << "," << posStateToCSV(playerState.posState);
        result << "," << playerState.money;
        result << "," << playerState.currentWeapon;
        result << "," << playerState.primaryWeapon;
        result << "," << playerState.secondaryWeapon;
        result << "," << playerState.currentClipBullets;
        result << "," << playerState.primaryClipBullets;
        result << "," << playerState.secondaryClipBullets;
        result << "," << playerState.health;
        result << "," << playerState.armor;
        result << "," << playerState.secondsSinceLastFootstep;
        result << "," << playerState.secondsSinceLastFire;
        return result.str();
    }

    void friendlyPlayerStateColumns(string prefix, vector<string> & result,
                                    bool onlyOneHot = false, bool onlyMinMaxScale = false) {
        if (!onlyMinMaxScale && !onlyOneHot) {
            result.push_back(prefix + " slot filled x");
            result.push_back(prefix + " alive");
        }
        if (!onlyOneHot) {
            posStateColumns(prefix, result, onlyOneHot, onlyMinMaxScale);
            result.push_back(prefix + " money");
        }
        if (!onlyMinMaxScale) {
            result.push_back(prefix + " current weapon");
            result.push_back(prefix + " primary weapon");
            result.push_back(prefix + " secondary weapon");
        }
        if (!onlyOneHot) {
            result.push_back(prefix + " current clip bullets");
            result.push_back(prefix + " primary clip bullets");
            result.push_back(prefix + " secondary clip bullets");
            result.push_back(prefix + " health");
            result.push_back(prefix + " armor");
            result.push_back(prefix + " seconds since last footstep");
            result.push_back(prefix + " seconds since last fire");
        }
    }

    void friendlyPlayerStateOneHotNumCategories(vector<string> & result, string equipmentIdList) {
        result.push_back(equipmentIdList);
        result.push_back(equipmentIdList);
        result.push_back(equipmentIdList);
    }

    struct EnemyPlayerState {
        bool slotFilled;
        bool alive;
        PosState posState;
        bool startRoundMoneyLessThan4k;
        bool beenHeadshotThisRound;
        int16_t currentWeapon;
        int32_t priorTimesEngagedThisRound;
        double secondsSinceLastSpotted;
        double secondsSinceLastFootstep;
        double secondsSinceLastFire;
        /*
        int64_t ticksSinceLastRadar;
        int64_t ticksSinceLastFootstep;
        int64_t ticksSinceLastFire;
         */
    };

    string enemyPlayerStateToCSV(EnemyPlayerState playerState) {
        std::stringstream result;
        result << boolToString(playerState.slotFilled);
        result << "," << boolToString(playerState.alive);
        result << "," << posStateToCSV(playerState.posState);
        result << "," << boolToString(playerState.startRoundMoneyLessThan4k);
        result << "," << boolToString(playerState.beenHeadshotThisRound);
        result << "," << playerState.currentWeapon;
        result << "," << playerState.priorTimesEngagedThisRound;
        result << "," << playerState.secondsSinceLastSpotted;
        result << "," << playerState.secondsSinceLastFootstep;
        result << "," << playerState.secondsSinceLastFire;
        return result.str();
    }

    void enemyPlayerStateColumns(string prefix, vector<string> & result,
                                    bool onlyOneHot = false, bool onlyMinMaxScale = false) {
        if (!onlyMinMaxScale && !onlyOneHot) {
            result.push_back(prefix + " slot filled x");
            result.push_back(prefix + " alive");
        }
        if (!onlyOneHot) {
            posStateColumns(prefix, result, onlyOneHot, onlyMinMaxScale);
        }
        if (!onlyMinMaxScale && !onlyOneHot) {
            result.push_back(prefix + " start money lt 4k");
            result.push_back(prefix + " been hs this round");
        }
        if (!onlyMinMaxScale) {
            result.push_back(prefix + " current weapon");
        }
        if (!onlyOneHot) {
            result.push_back(prefix + " prior times engaged this round");
            result.push_back(prefix + " seconds since last spotted");
            result.push_back(prefix + " seconds since last footstep");
            result.push_back(prefix + " seconds since last fire");
        }
    }

    void enemyPlayerStateOneHotNumCategories(vector<string> & result, string equipmentIdList) {
        result.push_back(equipmentIdList);
    }

    struct TimeStepState {
        int32_t team;
        Vec3 globalShooterEyePos;
        Vec3 globalShooterVelocity;
        Vec2 globalShooterViewAngle;
        Vec2 viewAngleWithActualRecoil;
        Vec2 viewAngleWithVisualRecoil;
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
        result << step.team;
        result << "," << step.globalShooterEyePos.toCSV();
        result << "," << step.globalShooterVelocity.toCSV();
        result << "," << step.globalShooterViewAngle.toCSV();
        result << "," << step.viewAngleWithActualRecoil.toCSV();
        result << "," << step.viewAngleWithVisualRecoil.toCSV();
        result << "," << friendlyPlayerStateToCSV(step.shooter);
        result << "," << enemyPlayerStateToCSV(step.target);
        for (const auto & friendlyPlayerState : step.friendlyPlayerStates) {
            result << "," << friendlyPlayerStateToCSV(friendlyPlayerState);
        }
        for (const auto & enemyPlayerState : step.enemyPlayerStates) {
            result << "," << enemyPlayerStateToCSV(enemyPlayerState);
        }
        return result.str();
    }

    void timeStepStateColumns(vector<string> & result, bool onlyOneHot = false, bool onlyMinMaxScale = false) {
        if (!onlyMinMaxScale && !onlyOneHot) {
            result.push_back("team");
        }
        if (!onlyOneHot) {
            result.push_back("global eye pos x");
            result.push_back("global eye pos y");
            result.push_back("global eye pos z");
            result.push_back("global vel x");
            result.push_back("global vel y");
            result.push_back("global vel z");
            result.push_back("global view angle x");
            result.push_back("global view angle y");
            result.push_back("view angle actual recoil x");
            result.push_back("view angle actual recoil y");
            result.push_back("view angle visual recoil x");
            result.push_back("view angle visual recoil y");
        }
        friendlyPlayerStateColumns("shooter", result, onlyOneHot, onlyMinMaxScale);
        enemyPlayerStateColumns("enemy", result, onlyOneHot, onlyMinMaxScale);
        for (int i = 0; i < NUM_PLAYERS/2; i++) {
            friendlyPlayerStateColumns("friendly " + std::to_string(i), result, onlyOneHot, onlyMinMaxScale);
        }
        for (int i = 0; i < NUM_PLAYERS/2; i++) {
            enemyPlayerStateColumns("enemy " + std::to_string(i), result, onlyOneHot, onlyMinMaxScale);
        }

    }

    void timeStepStateOneHotNumCategories(vector<string> & result, string equipmentIdList) {
        friendlyPlayerStateOneHotNumCategories(result, equipmentIdList);
        enemyPlayerStateOneHotNumCategories(result, equipmentIdList);
        for (int i = 0; i < NUM_PLAYERS/2; i++) {
            friendlyPlayerStateOneHotNumCategories(result, equipmentIdList);
        }
        for (int i = 0; i < NUM_PLAYERS/2; i++) {
            enemyPlayerStateOneHotNumCategories(result, equipmentIdList);
        }
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
        result << plan.deltaPos.toCSV()
               << "," << plan.deltaView.toCSV()
               << "," << boolToString(plan.fire)
               << "," << boolToString(plan.crouch)
               << "," << boolToString(plan.walk)
               << "," << boolToString(plan.scope)
               << "," << boolToString(plan.newlyAirborne);
        return result.str();
    }

    void timeStepPlanColumns(vector<string> & result, bool onlyOneHot = false, bool onlyMinMaxScale = false) {
        if (!onlyOneHot) {
            result.push_back("delta pos x");
            result.push_back("delta pos y");
            result.push_back("delta pos z");
            result.push_back("delta view x");
            result.push_back("delta view y");
            result.push_back("delta view z");
        }
        if (!onlyOneHot && !onlyMinMaxScale) {
            result.push_back("fire");
            result.push_back("crouch");
            result.push_back("walk");
            result.push_back("scope");
            result.push_back("newly airborne");
        }
    }

    void timeStepPlanOneHotNumCategories(vector<string> & result) { }

    vector<int64_t> tickId;
    vector<int64_t> roundId;
    vector<int64_t> sourcePlayerId;
    vector<string> sourcePlayerName;
    vector<string> demoName;
    vector<TimeStepState> states;
    vector<TimeStepPlan> plans;
    const Equipment & equipment;

    EngagementResult(const Equipment & equipment) : equipment(equipment) {
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
           << "," << demoName[index]
           << "," << timeStepStateToString(states[index])
           << "," << timeStepPlanToString(plans[index]) << std::endl;
    }

    vector<string> getForeignKeyNames() {
        return {"tick id", "round id", "source player id"};
    }

    vector<string> getOtherColumnNames() {
        vector<string> result{"source player name", "demo name", "team"};
        timeStepStateColumns(result);
        timeStepPlanColumns(result);
        return result;
    }

    string getDataLabelRanges() {
        std::stringstream result;
        vector<string> inputCols, outputCols, inputOneHot, inputOneHotNumCategories, outputOneHot, outputOneHotNumCategories,
                inputMinMaxScale, outputMinMaxScale;

        timeStepStateColumns(inputCols);
        timeStepPlanColumns(outputCols);

        timeStepStateColumns(inputOneHot, true);
        timeStepPlanColumns(outputOneHot, true);

        // repeat the below once for each state
        vector<string> equipmentIdListVec;
        std::stringstream equipmentStream;
        bool firstEquipment = true;
        for (const auto id : equipment.id) {
            equipmentIdListVec.push_back(std::to_string(id));
        }
        commaSeparateList(equipmentStream, equipmentIdListVec);
        timeStepStateOneHotNumCategories(inputOneHotNumCategories, equipmentStream.str());
        timeStepPlanOneHotNumCategories(outputOneHotNumCategories);

        timeStepStateColumns(inputMinMaxScale, false, true);
        timeStepPlanColumns(outputMinMaxScale, false, true);

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

EngagementResult queryEngagementDataset(const Games & games, const Rounds & rounds, const Ticks & ticks,
                                    const Players & players, const PlayerAtTick & playerAtTick,
                                    const std::map<std::string, const nav_mesh::nav_file> & mapNavs);

#endif //CSKNOW_ENGAGEMENT_H