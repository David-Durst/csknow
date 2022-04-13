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
#include "enum_helpers.h"
#include <array>
#include <string>
#include <map>
#include <sstream>
using std::string;
using std::map;
using std::array;
#define ENGAGEMENT_SECONDS_RADIUS 1.00
#define WEAPON_RECOIL_SCALE 2.0
#define VIEW_RECOIL_TRACKING 0.45

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

    enum class ColumnTypes {
        Boolean = 0,
        FloatMinMax = 1,
        FloatNonLinear = 2,
        Categorical = 3,
        Drop = 4,
    };

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
        result << "," << boolToInt(posState.isCrouching);
        result << "," << boolToInt(posState.isWalking);
        result << "," << boolToInt(posState.isScoped);
        result << "," << boolToInt(posState.isAirborne);
        result << "," << posState.remainingFlashTime;
        return result.str();
    }

    void posStateColumns(string prefix, vector<string> & resultNames, vector<ColumnTypes> & resultTypes) {
        resultNames.push_back(prefix + " eye pos rel x");
        resultTypes.push_back(ColumnTypes::FloatMinMax);
        resultNames.push_back(prefix + " eye pos rel y");
        resultTypes.push_back(ColumnTypes::FloatMinMax);
        resultNames.push_back(prefix + " eye pos rel z");
        resultTypes.push_back(ColumnTypes::FloatMinMax);
        resultNames.push_back(prefix + " vel rel x");
        resultTypes.push_back(ColumnTypes::FloatMinMax);
        resultNames.push_back(prefix + " vel rel y");
        resultTypes.push_back(ColumnTypes::FloatMinMax);
        resultNames.push_back(prefix + " vel rel z");
        resultTypes.push_back(ColumnTypes::FloatMinMax);
        resultNames.push_back(prefix + " view angle rel x");
        resultTypes.push_back(ColumnTypes::FloatMinMax);
        resultNames.push_back(prefix + " view angle rel y");
        resultTypes.push_back(ColumnTypes::FloatMinMax);
        resultNames.push_back(prefix + " crouching");
        resultTypes.push_back(ColumnTypes::Boolean);
        resultNames.push_back(prefix + " walking");
        resultTypes.push_back(ColumnTypes::Boolean);
        resultNames.push_back(prefix + " scoped");
        resultTypes.push_back(ColumnTypes::Boolean);
        resultNames.push_back(prefix + " airborne");
        resultTypes.push_back(ColumnTypes::Boolean);
        resultNames.push_back(prefix + " remaining flash time");
        resultTypes.push_back(ColumnTypes::FloatMinMax);
    }

    struct FriendlyPlayerState {
        bool slotFilled;
        bool alive;
        PosState posState;
        int64_t money;
        int16_t activeWeapon;
        int16_t primaryWeapon;
        int16_t secondaryWeapon;
        int16_t currentClipBullets;
        int16_t primaryClipBullets;
        int16_t secondaryClipBullets;
        int32_t health;
        int32_t armor;
        /*
        double secondsSinceLastFootstep;
        double secondsSinceLastFire;
         */
    };

    string friendlyPlayerStateToCSV(FriendlyPlayerState playerState) {
        std::stringstream result;
        result << boolToInt(playerState.slotFilled);
        result << "," << boolToInt(playerState.alive);
        result << "," << posStateToCSV(playerState.posState);
        result << "," << playerState.money;
        result << "," << playerState.activeWeapon;
        result << "," << playerState.primaryWeapon;
        result << "," << playerState.secondaryWeapon;
        result << "," << playerState.currentClipBullets;
        result << "," << playerState.primaryClipBullets;
        result << "," << playerState.secondaryClipBullets;
        result << "," << playerState.health;
        result << "," << playerState.armor;
        /*
        result << "," << playerState.secondsSinceLastFootstep;
        result << "," << playerState.secondsSinceLastFire;
         */
        return result.str();
    }

    void friendlyPlayerStateColumns(string prefix, vector<string> & resultNames,
                                    vector<ColumnTypes> & resultTypes) {
        resultNames.push_back(prefix + " slot filled");
        resultTypes.push_back(ColumnTypes::Boolean);
        resultNames.push_back(prefix + " alive");
        resultTypes.push_back(ColumnTypes::Boolean);
        posStateColumns(prefix, resultNames, resultTypes);
        resultNames.push_back(prefix + " money");
        resultTypes.push_back(ColumnTypes::FloatMinMax);
        resultNames.push_back(prefix + " active weapon");
        resultTypes.push_back(ColumnTypes::Categorical);
        resultNames.push_back(prefix + " primary weapon");
        resultTypes.push_back(ColumnTypes::Categorical);
        resultNames.push_back(prefix + " secondary weapon");
        resultTypes.push_back(ColumnTypes::Categorical);
        resultNames.push_back(prefix + " current clip bullets");
        resultTypes.push_back(ColumnTypes::FloatMinMax);
        resultNames.push_back(prefix + " primary clip bullets");
        resultTypes.push_back(ColumnTypes::FloatMinMax);
        resultNames.push_back(prefix + " secondary clip bullets");
        resultTypes.push_back(ColumnTypes::FloatMinMax);
        resultNames.push_back(prefix + " health");
        resultTypes.push_back(ColumnTypes::FloatMinMax);
        resultNames.push_back(prefix + " armor");
        resultTypes.push_back(ColumnTypes::FloatMinMax);
        /*
        result.push_back(prefix + " seconds since last footstep");
        resultTypes.push_back(ColumnTypes::FloatMinMax);
        result.push_back(prefix + " seconds since last fire");
        resultTypes.push_back(ColumnTypes::FloatMinMax);
         */
    }

    void friendlyPlayerStateOneHotNumCategories(vector<string> & result, string equipmentIdList) {
        result.push_back(equipmentIdList);
        result.push_back(equipmentIdList);
        result.push_back(equipmentIdList);
    }

    struct EnemyPlayerState {
        bool slotFilled;
        bool alive;
        bool engaged;
        PosState posState;
        bool saveRound;
        //bool beenHeadshotThisRound;
        int16_t activeWeapon;
        //int32_t priorTimesEngagedThisRound;
        /*
        double secondsSinceLastSpotted;
        double secondsSinceLastFootstep;
        double secondsSinceLastFire;
        int64_t ticksSinceLastRadar;
        int64_t ticksSinceLastFootstep;
        int64_t ticksSinceLastFire;
         */
    };

    string enemyPlayerStateToCSV(EnemyPlayerState playerState) {
        std::stringstream result;
        result << boolToInt(playerState.slotFilled);
        result << "," << boolToInt(playerState.alive);
        result << "," << boolToInt(playerState.engaged);
        result << "," << posStateToCSV(playerState.posState);
        result << "," << boolToInt(playerState.saveRound);
        //result << "," << boolToInt(playerState.beenHeadshotThisRound);
        result << "," << playerState.activeWeapon;
        //result << "," << playerState.priorTimesEngagedThisRound;
        /*
        result << "," << playerState.secondsSinceLastSpotted;
        result << "," << playerState.secondsSinceLastFootstep;
        result << "," << playerState.secondsSinceLastFire;
         */
        return result.str();
    }

    void enemyPlayerStateColumns(string prefix, vector<string> & resultNames,
                                 vector<ColumnTypes> & resultTypes) {
        resultNames.push_back(prefix + " slot filled");
        resultTypes.push_back(ColumnTypes::Boolean);
        resultNames.push_back(prefix + " alive");
        resultTypes.push_back(ColumnTypes::Boolean);
        resultNames.push_back(prefix + " engaged");
        resultTypes.push_back(ColumnTypes::Boolean);
        posStateColumns(prefix, resultNames, resultTypes);
        resultNames.push_back(prefix + " save round");
        resultTypes.push_back(ColumnTypes::Boolean);
        //resultNames.push_back(prefix + " been hs this round");
        resultNames.push_back(prefix + " active weapon");
        resultTypes.push_back(ColumnTypes::Categorical);
        /*
        if (!onlyOneHot) {
            result.push_back(prefix + " prior times engaged this round");
            result.push_back(prefix + " seconds since last spotted");
            result.push_back(prefix + " seconds since last footstep");
            result.push_back(prefix + " seconds since last fire");
        }
         */
    }

    void enemyPlayerStateOneHotNumCategories(vector<string> & result, string equipmentIdList) {
        result.push_back(equipmentIdList);
    }

    struct TimeStepState {
        int64_t bestEngagementId;
        int32_t team;
        Vec3 globalShooterEyePos;
        Vec3 globalShooterVelocity;
        Vec2 globalShooterViewAngle;
        Vec2 viewAngleWithActualRecoil;
        Vec2 viewAngleWithVisualRecoil;
        double secondsSinceLastFire;
        Vec2 priorDeltaView1;
        Vec2 priorDeltaView4;
        Vec2 priorDeltaView8;
        Vec2 priorDeltaView16;
        Vec2 priorDeltaView32;
        FriendlyPlayerState shooter;
        EnemyPlayerState target;
        array<FriendlyPlayerState, NUM_PLAYERS/2> friendlyPlayerStates;
        array<EnemyPlayerState, NUM_PLAYERS/2> enemyPlayerStates;
        // these aren't printed, just used for bookkeeping during query
        int64_t roundId;
        int64_t tickId;
        int64_t shooterPatId;
    };

    string timeStepStateToString(TimeStepState step) {
        std::stringstream result;
        result << step.bestEngagementId;
        result << "," << step.team;
        result << "," << step.globalShooterEyePos.toCSV();
        result << "," << step.globalShooterVelocity.toCSV();
        result << "," << step.globalShooterViewAngle.toCSV();
        result << "," << step.viewAngleWithActualRecoil.toCSV();
        result << "," << step.viewAngleWithVisualRecoil.toCSV();
        result << "," << step.secondsSinceLastFire;
        result << "," << step.priorDeltaView1.toCSV();
        result << "," << step.priorDeltaView4.toCSV();
        result << "," << step.priorDeltaView8.toCSV();
        result << "," << step.priorDeltaView16.toCSV();
        result << "," << step.priorDeltaView32.toCSV();
        result << "," << friendlyPlayerStateToCSV(step.shooter);
        result << "," << enemyPlayerStateToCSV(step.target);
        /*
        for (const auto & friendlyPlayerState : step.friendlyPlayerStates) {
            result << "," << friendlyPlayerStateToCSV(friendlyPlayerState);
        }
        for (const auto & enemyPlayerState : step.enemyPlayerStates) {
            result << "," << enemyPlayerStateToCSV(enemyPlayerState);
        }
         */
        return result.str();
    }

    void timeStepStateColumns(vector<string> & resultNames, vector<ColumnTypes> & resultTypes) {
        resultNames.push_back("engagement id");
        resultTypes.push_back(ColumnTypes::Drop);
        resultNames.push_back("team");
        resultTypes.push_back(ColumnTypes::Categorical);
        resultNames.push_back("global eye pos x");
        resultTypes.push_back(ColumnTypes::FloatMinMax);
        resultNames.push_back("global eye pos y");
        resultTypes.push_back(ColumnTypes::FloatMinMax);
        resultNames.push_back("global eye pos z");
        resultTypes.push_back(ColumnTypes::FloatMinMax);
        resultNames.push_back("global vel x");
        resultTypes.push_back(ColumnTypes::FloatMinMax);
        resultNames.push_back("global vel y");
        resultTypes.push_back(ColumnTypes::FloatMinMax);
        resultNames.push_back("global vel z");
        resultTypes.push_back(ColumnTypes::FloatMinMax);
        resultNames.push_back("global view angle x");
        resultTypes.push_back(ColumnTypes::FloatMinMax);
        resultNames.push_back("global view angle y");
        resultTypes.push_back(ColumnTypes::FloatMinMax);
        resultNames.push_back("view angle actual recoil x");
        resultTypes.push_back(ColumnTypes::FloatMinMax);
        resultNames.push_back("view angle actual recoil y");
        resultTypes.push_back(ColumnTypes::FloatMinMax);
        resultNames.push_back("view angle visual recoil x");
        resultTypes.push_back(ColumnTypes::FloatMinMax);
        resultNames.push_back("view angle visual recoil y");
        resultTypes.push_back(ColumnTypes::FloatMinMax);
        resultNames.push_back("seconds since last fire");
        resultTypes.push_back(ColumnTypes::FloatMinMax);
        resultNames.push_back("prior delta view x 1");
        resultTypes.push_back(ColumnTypes::FloatNonLinear);
        resultNames.push_back("prior delta view y 1");
        resultTypes.push_back(ColumnTypes::FloatNonLinear);
        resultNames.push_back("prior delta view x 4");
        resultTypes.push_back(ColumnTypes::FloatNonLinear);
        resultNames.push_back("prior delta view y 4");
        resultTypes.push_back(ColumnTypes::FloatNonLinear);
        resultNames.push_back("prior delta view x 8");
        resultTypes.push_back(ColumnTypes::FloatNonLinear);
        resultNames.push_back("prior delta view y 8");
        resultTypes.push_back(ColumnTypes::FloatNonLinear);
        resultNames.push_back("prior delta view x 16");
        resultTypes.push_back(ColumnTypes::FloatNonLinear);
        resultNames.push_back("prior delta view y 16");
        resultTypes.push_back(ColumnTypes::FloatNonLinear);
        resultNames.push_back("prior delta view x 32");
        resultTypes.push_back(ColumnTypes::FloatNonLinear);
        resultNames.push_back("prior delta view y 32");
        resultTypes.push_back(ColumnTypes::FloatNonLinear);
        friendlyPlayerStateColumns("shooter", resultNames, resultTypes);
        enemyPlayerStateColumns("enemy", resultNames, resultTypes);
        /*
        for (int i = 0; i < NUM_PLAYERS/2; i++) {
            friendlyPlayerStateColumns("friendly " + std::to_string(i), resultNames, onlyOneHot, onlyMinMaxScale);
        }
        for (int i = 0; i < NUM_PLAYERS/2; i++) {
            enemyPlayerStateColumns("enemy " + std::to_string(i), resultNames, onlyOneHot, onlyMinMaxScale);
        }
         */

    }

    void timeStepStateOneHotNumCategories(vector<string> & result, string equipmentIdList) {
        result.push_back("\"" + std::to_string(INTERNAL_TEAM_CT) + ";" + std::to_string(INTERNAL_TEAM_T) + "\"");
        friendlyPlayerStateOneHotNumCategories(result, equipmentIdList);
        enemyPlayerStateOneHotNumCategories(result, equipmentIdList);
        /*
        for (int i = 0; i < NUM_PLAYERS/2; i++) {
            friendlyPlayerStateOneHotNumCategories(result, equipmentIdList);
        }
        for (int i = 0; i < NUM_PLAYERS/2; i++) {
            enemyPlayerStateOneHotNumCategories(result, equipmentIdList);
        }
         */
    }

    /*
    enum class ActionResult {
        keepEngaging,
        changeTarget,
        run,
        switchWeapon,
        kill,
        die,
        NUM_ACTION_RESULTS,
    };
     */

    struct TimeStepAction {
        //ActionResult actionResult;
        double secondsUntilEngagementOver;
        Vec3 deltaPos;
        Vec2 deltaView1;
        Vec2 deltaView4;
        Vec2 deltaView8;
        double nextFireTimeSeconds;
        bool crouch;
        bool walk;
        bool scope;
        bool newlyAirborne;
    };

    string timeStepPlanToString(TimeStepAction action) {
        std::stringstream result;
        result //<< action.secondsUntilEngagementOver //enumAsInt(action.actionResult)
               //<< "," << action.deltaPos.toCSV()
               //<< "," << action.deltaView1.toCSV(); /*
               << action.deltaView1.toCSV()
               << "," << action.deltaView4.toCSV()
               << "," << action.deltaView8.toCSV(); /*
               << "," << boolToInt(action.nextFireTimeSeconds)
               << "," << boolToInt(action.crouch)
               << "," << boolToInt(action.walk)
               << "," << boolToInt(action.scope)
               << "," << boolToInt(action.newlyAirborne);*/
        return result.str();
    }

    void timeStepActionColumns(vector<string> & resultNames, vector<ColumnTypes> & resultTypes) {
        /*
        if (!onlyMinMaxScale) {
            resultNames.push_back("action result");
            resultTypes.push_back(ColumnTypes::Categorical);
        }
         */
        /*
        resultNames.push_back("seconds until engagement over");
        resultTypes.push_back(ColumnTypes::FloatMinMax);
        resultNames.push_back("delta pos x");
        resultTypes.push_back(ColumnTypes::FloatNonLinear);
        resultNames.push_back("delta pos y");
        resultTypes.push_back(ColumnTypes::FloatNonLinear);
        resultNames.push_back("delta pos z");
        resultTypes.push_back(ColumnTypes::FloatNonLinear);
         */
        resultNames.push_back("delta view x 1");
        resultTypes.push_back(ColumnTypes::FloatNonLinear);
        resultNames.push_back("delta view y 1");
        resultTypes.push_back(ColumnTypes::FloatNonLinear);
        resultNames.push_back("delta view x 4");
        resultTypes.push_back(ColumnTypes::FloatNonLinear);
        resultNames.push_back("delta view y 4");
        resultTypes.push_back(ColumnTypes::FloatNonLinear);
        resultNames.push_back("delta view x 8");
        resultTypes.push_back(ColumnTypes::FloatNonLinear);
        resultNames.push_back("delta view y 8");
        resultTypes.push_back(ColumnTypes::FloatNonLinear);
        /*
        resultNames.push_back("next fire time seconds");
        resultTypes.push_back(ColumnTypes::FloatMinMax);
         */
        /*
        resultNames.push_back("crouch");
        resultTypes.push_back(ColumnTypes::Boolean);
        resultNames.push_back("walk");
        resultTypes.push_back(ColumnTypes::Boolean);
        resultNames.push_back("scope");
        resultTypes.push_back(ColumnTypes::Boolean);
        resultNames.push_back("newly airborne");
        resultTypes.push_back(ColumnTypes::Boolean);
         */
    }

    /*
    enum class LossType {
        Binary,
        Float,
        Categorical
    };

    map<LossType, string> timeStepActionLossTypes() {
        map<LossType, string> result;
        result[LossType::Binary] = {};
        result[LossType::Float] = {};
        result[LossType::Categorical] = {};
        result[LossType::Categorical}]push_back(std::to_string(enumAsInt(LossType::Float)));
        result.push_back(std::to_string(enumAsInt(LossType::Float)));
        return result;
    }
     */

    void timeStepActionOneHotNumCategories(vector<string> & result) {
        //result.push_back(std::to_string(enumAsInt(ActionResult::NUM_ACTION_RESULTS)));
    }

    vector<int64_t> tickId;
    vector<int64_t> roundId;
    vector<int64_t> sourcePlayerId;
    vector<int64_t> gameTickNumber;
    vector<string> sourcePlayerName;
    vector<string> demoName;
    vector<TimeStepState> states;
    vector<TimeStepAction> actions;
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
           << "," << sourcePlayerId[index] << "," << gameTickNumber[index]
           << "," << sourcePlayerName[index] << "," << demoName[index]
           << "," << timeStepStateToString(states[index])
           << "," << timeStepPlanToString(actions[index]) << std::endl;
    }

    vector<string> getForeignKeyNames() {
        return {"tick id", "round id", "source player id"};
    }

    vector<string> getOtherColumnNames() {
        vector<string> result{"game tick number", "source player name", "demo name"};
        vector<ColumnTypes> columnTypes;
        timeStepStateColumns(result, columnTypes);
        timeStepActionColumns(result, columnTypes);
        return result;
    }

    string getDataLabelRanges() {
        std::stringstream result;
        vector<string> inputCols, outputCols, inputOneHotNumCategories, outputOneHotNumCategories;
        vector<ColumnTypes> inputColTypes, outputColTypes;

        timeStepStateColumns(inputCols, inputColTypes);
        timeStepActionColumns(outputCols, outputColTypes);

        vector<string> inputColTypesStrs, outputColTypesStrs;
        for (const auto & inputColType : inputColTypes) {
            inputColTypesStrs.push_back(std::to_string(enumAsInt(inputColType)));
        }
        for (const auto & outputColType : outputColTypes) {
            outputColTypesStrs.push_back(std::to_string(enumAsInt(outputColType)));
        }

        // repeat the below once for each state
        vector<string> equipmentIdListVec;
        std::stringstream equipmentStream;
        for (const auto id : equipment.id) {
            equipmentIdListVec.push_back(std::to_string(id));
        }
        commaSeparateList(equipmentStream, equipmentIdListVec, ";");
        string equipmentIdStr = "\"" + equipmentStream.str() + "\"";
        timeStepStateOneHotNumCategories(inputOneHotNumCategories, equipmentIdStr);
        timeStepActionOneHotNumCategories(outputOneHotNumCategories);

        result << "source player id\n";
        commaSeparateList(result, inputCols);
        result << "\n";
        commaSeparateList(result, inputColTypesStrs);
        result << "\n";
        commaSeparateList(result, inputOneHotNumCategories);
        result << "\n";
        commaSeparateList(result, outputCols);
        result << "\n";
        commaSeparateList(result, outputColTypesStrs);
        result << "\n";
        commaSeparateList(result, outputOneHotNumCategories);
        result << "\n";
        return result.str();
    }
};

EngagementResult queryEngagementDataset(const Equipment & equipment, const Games & games, const Rounds & rounds,
                                        const WeaponFire & weaponFire, const Hurt & hurt,
                                        const Ticks & ticks, const Players & players, const PlayerAtTick & playerAtTick);

#endif //CSKNOW_ENGAGEMENT_H